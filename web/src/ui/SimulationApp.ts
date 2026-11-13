import * as THREE from 'three';
import { BiomeSystem } from '../biosphere/BiomeSystem';
import {
  ImpactBiosphereCoupler,
  type BiosphereMetrics,
} from '../biosphere/ImpactBiosphereCoupler';
import { ProceduralFauna, type FaunaStimulus } from '../biosphere/ProceduralFauna';
import { ProceduralFlora } from '../biosphere/ProceduralFlora';
import {
  createParticleBufferPool,
  createTracerBufferPool,
  type ParticleBufferPool,
  type TracerBufferPool,
} from '../net/BinaryProtocol';
import {
  SimClient,
  type ConnectionState,
  type CustomScenarioDto,
  type TelemetrySnapshotDto,
} from '../net/SimClient';
import { AtmosphereShader } from '../render/AtmosphereShader';
import { CubeSphereLod } from '../render/CubeSphereLod';
import {
  cubeFaceUvToUnitNormal,
  OrbitalSurfaceCamera,
  rotateYUnitNormal,
  unitNormalToCubeFaceUv,
  type CameraMode,
  type SurfaceAnchorCoord,
} from '../render/OrbitalSurfaceCamera';
import { SphParticleRenderer, type VisMode } from '../render/SphParticleRenderer';
import { ControlPanel } from './ControlPanel';
import { TelemetryCharts } from './TelemetryCharts';
import { ensureHudStyles, getOwnerDoc, makeEl } from './hudUtils';

const MAX_PARTICLES = 250_000;
const MAX_TRACERS = 6 * 32 * 32;
const STARFIELD_RADIUS_KM = 2.5e7;
const STARFIELD_COUNT = 2400;

export interface AstraeaDebugState {
  visMode: VisMode;
  sliceEnabled: boolean;
  cameraMode: CameraMode;
  altitudeKm: number;
  particleCount: number;
  tracerCount: number;
  activePatches: number;
  culledDamagedPatches: number;
  floraVisible: boolean;
  faunaVisible: boolean;
  survivalPct: number;
  extinctionTier: string;
  frameSeq: number;
  simTimeSec: number;
  connectionState: ConnectionState;
  glErrors: number;
  rendererReady: boolean;
}

function resolveDefaultWsUrl(): string {
  const loc =
    typeof globalThis !== 'undefined' && 'location' in globalThis
      ? (globalThis as { location?: { protocol?: string; host?: string } })
          .location
      : undefined;
  const protocol = loc?.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = loc?.host && loc.host.length > 0 ? loc.host : '127.0.0.1:8080';
  return `${protocol}//${host}/ws`;
}

function createStarfield(): {
  points: THREE.Points;
  geometry: THREE.BufferGeometry;
  material: THREE.PointsMaterial;
} {
  const positions = new Float32Array(STARFIELD_COUNT * 3);
  const colors = new Float32Array(STARFIELD_COUNT * 3);

  let state = 1337;
  const rand = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };

  for (let i = 0; i < STARFIELD_COUNT; i++) {
    const u = rand() * 2.0 - 1.0;
    const phi = rand() * Math.PI * 2.0;
    const sinTheta = Math.sqrt(Math.max(0.0, 1.0 - u * u));
    const idx = i * 3;
    positions[idx] = sinTheta * Math.cos(phi) * STARFIELD_RADIUS_KM;
    positions[idx + 1] = u * STARFIELD_RADIUS_KM;
    positions[idx + 2] = sinTheta * Math.sin(phi) * STARFIELD_RADIUS_KM;

    const tint = rand();
    const brightness = 0.55 + 0.45 * rand();
    if (tint < 0.25) {
      // Blue-white O/B star
      colors[idx] = 0.72 * brightness;
      colors[idx + 1] = 0.84 * brightness;
      colors[idx + 2] = 1.0 * brightness;
    } else if (tint < 0.75) {
      // Solar white
      colors[idx] = 0.98 * brightness;
      colors[idx + 1] = 0.96 * brightness;
      colors[idx + 2] = 0.92 * brightness;
    } else {
      // Warm K/M star
      colors[idx] = 1.0 * brightness;
      colors[idx + 1] = 0.8 * brightness;
      colors[idx + 2] = 0.62 * brightness;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 1.8,
    sizeAttenuation: false,
    vertexColors: true,
    transparent: true,
    opacity: 0.88,
    depthWrite: false,
  });

  const points = new THREE.Points(geometry, material);
  points.name = 'ObservatoryStarfield';
  points.frustumCulled = false;
  points.renderOrder = -10;
  return { points, geometry, material };
}

function hashTileKey(key: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

export class SimulationApp {
  private readonly canvasContainer: HTMLElement;
  private readonly uiContainer: HTMLElement;

  private readonly particlePool: ParticleBufferPool;
  private readonly tracerPool: TracerBufferPool;
  private readonly client: SimClient;

  private readonly scene: THREE.Scene;
  private readonly renderer: THREE.WebGLRenderer | null;
  private readonly sunLight: THREE.DirectionalLight;
  private readonly hemiLight: THREE.HemisphereLight;
  private readonly starfield: ReturnType<typeof createStarfield>;

  private readonly cameraCtrl: OrbitalSurfaceCamera;
  private readonly sphRenderer: SphParticleRenderer;
  private readonly atmosphere: AtmosphereShader;
  private readonly biomeSystem: BiomeSystem;
  private readonly cubeLod: CubeSphereLod;
  private readonly flora: ProceduralFlora;
  private readonly fauna: ProceduralFauna;
  private readonly coupler: ImpactBiosphereCoupler;

  private readonly hudOverlay: HTMLElement | null;
  private readonly connectionPill: HTMLElement | null = null;
  private readonly topSimInfo: HTMLElement | null = null;
  private readonly errorToast: HTMLElement | null = null;
  private readonly controlPanel: ControlPanel;
  private readonly telemetryCharts: TelemetryCharts;

  private readonly sunDir = new THREE.Vector3(0.6, 0.5, 0.624).normalize();
  private readonly sliceNormal = new THREE.Vector3(0.0, 1.0, 0.0);
  private visMode: VisMode = 'NaturalBlackbody';
  private sliceEnabled = false;
  private sliceOffsetKm = 0.0;

  private latestTelemetry: TelemetrySnapshotDto | null = null;
  private latestBio: BiosphereMetrics;
  private lastPresetName = '';
  private lastFrameSeq = 0;
  private connectionState: ConnectionState = 'disconnected';
  private glErrors = 0;
  private glCheckCountdown = 2;

  private started = false;
  private disposed = false;
  private rafId: number | null = null;
  private lastAnimTimeMs = 0;
  private resizeObserver: ResizeObserver | null = null;
  private readonly cleanups: Array<() => void> = [];

  // Cached chunk state to avoid redundant flora/fauna rebuilds
  private lastFaunaTileKey = '';
  private lastFloraTileKey = '';
  private readonly lastFloraCam: [number, number, number] = [NaN, NaN, NaN];
  private readonly lastFloraCom: [number, number, number] = [NaN, NaN, NaN];
  private lastFloraSpin = NaN;

  // Scratch vectors to avoid per-frame allocations
  private readonly tmpAnchorRest = new THREE.Vector3();
  private readonly tmpAnchorWorld = new THREE.Vector3();
  private readonly tmpImpAzimuth = new THREE.Vector3(1.0, 0.0, 0.0);
  private readonly tmpBlastDir = new THREE.Vector3(-1.0, 0.0, 0.0);
  private readonly tmpCamVec = new THREE.Vector3();

  constructor(
    canvasContainer: HTMLElement,
    uiContainer: HTMLElement,
    wsUrl?: string
  ) {
    this.canvasContainer = canvasContainer;
    this.uiContainer = uiContainer;

    const resolvedUrl =
      wsUrl !== undefined && wsUrl.length > 0 ? wsUrl : resolveDefaultWsUrl();

    this.particlePool = createParticleBufferPool(MAX_PARTICLES);
    this.tracerPool = createTracerBufferPool(MAX_TRACERS);
    this.client = new SimClient(resolvedUrl, this.particlePool, this.tracerPool);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x02040b);

    this.sunLight = new THREE.DirectionalLight(0xfff6e8, 2.4);
    this.sunLight.position.copy(this.sunDir).multiplyScalar(1.0e5);
    this.hemiLight = new THREE.HemisphereLight(0x9ec5ff, 0x1b2030, 0.45);
    this.scene.add(this.sunLight);
    this.scene.add(this.hemiLight);

    this.starfield = createStarfield();
    this.scene.add(this.starfield.points);

    const dims = this.measureCanvasContainer();
    this.cameraCtrl = new OrbitalSurfaceCamera(
      dims.width / Math.max(dims.height, 1),
      18000.0
    );

    this.sphRenderer = new SphParticleRenderer(MAX_PARTICLES);
    this.sphRenderer.setViewportHeight(dims.height);
    this.scene.add(this.sphRenderer.mesh);

    this.atmosphere = new AtmosphereShader(64);
    this.scene.add(this.atmosphere.mesh);

    this.biomeSystem = new BiomeSystem(42);
    this.cubeLod = new CubeSphereLod(this.biomeSystem);
    this.cubeLod.setSunDirection(this.sunDir);
    this.scene.add(this.cubeLod.group);

    this.flora = new ProceduralFlora(this.biomeSystem, 4096);
    this.scene.add(this.flora.group);

    this.fauna = new ProceduralFauna(this.biomeSystem, 512);
    this.scene.add(this.fauna.group);

    this.coupler = new ImpactBiosphereCoupler(this.biomeSystem, 256);
    this.latestBio = this.coupler.getGlobalMetrics();

    this.renderer = this.createWebGLRenderer(dims.width, dims.height);

    // Mount HUD layout cleanly inside uiContainer
    const doc = getOwnerDoc(uiContainer);
    let leftMount = uiContainer;
    let rightMount = uiContainer;
    if (doc !== null) {
      ensureHudStyles(doc);
      const overlay = makeEl(doc, 'div', {
        className: 'astraea-hud-overlay',
        testId: 'hud-overlay',
      });

      const topbar = makeEl(doc, 'div', { className: 'astraea-topbar' });
      const brand = makeEl(doc, 'div', { className: 'astraea-brand' });
      brand.appendChild(
        makeEl(doc, 'span', {
          className: 'astraea-brand-title',
          text: 'ASTRAEA',
        })
      );
      brand.appendChild(
        makeEl(doc, 'span', {
          className: 'astraea-brand-sub',
          text: 'Planetary Collision & Biosphere Impact Simulator',
        })
      );

      const statusRow = makeEl(doc, 'div', { className: 'astraea-status-row' });
      this.errorToast = makeEl(doc, 'span', {
        className: 'astraea-error-toast',
        testId: 'error-toast',
      });
      this.topSimInfo = makeEl(doc, 'span', {
        text: '0 pts · 0 tracers',
        testId: 'topbar-sim-info',
      });
      this.connectionPill = makeEl(doc, 'span', {
        className: 'astraea-pill astraea-pill-info',
        text: 'disconnected',
        testId: 'connection-status',
      });
      statusRow.appendChild(this.errorToast);
      statusRow.appendChild(this.topSimInfo);
      statusRow.appendChild(this.connectionPill);

      topbar.appendChild(brand);
      topbar.appendChild(statusRow);

      const leftPanel = makeEl(doc, 'div', { className: 'astraea-panel-left' });
      const rightPanel = makeEl(doc, 'div', { className: 'astraea-panel-right' });

      overlay.appendChild(topbar);
      overlay.appendChild(leftPanel);
      overlay.appendChild(rightPanel);
      uiContainer.appendChild(overlay);

      this.hudOverlay = overlay;
      leftMount = leftPanel;
      rightMount = rightPanel;
    } else {
      this.hudOverlay = null;
    }

    this.controlPanel = new ControlPanel(leftMount, {
      onPreset: (preset, particles) => {
        this.coupler.resetBaseline();
        this.latestBio = this.coupler.getGlobalMetrics();
        this.resetChunkCaches();
        this.client.sendCommand({ type: 'set_preset', preset, particles });
      },
      onLaunchCustom: (cfg: CustomScenarioDto) => {
        this.coupler.resetBaseline();
        this.latestBio = this.coupler.getGlobalMetrics();
        this.resetChunkCaches();
        this.client.sendCommand({ type: 'launch_custom', config: cfg });
      },
      onPlayPause: (playing) => {
        this.client.sendCommand({ type: playing ? 'play' : 'pause' });
      },
      onStep: () => {
        this.client.sendCommand({ type: 'step', count: 1 });
      },
      onTimescale: (scale) => {
        this.client.sendCommand({ type: 'set_timescale', scale });
      },
      onVisMode: (mode) => {
        this.visMode = mode;
        this.sphRenderer.setVisMode(mode);
        this.glCheckCountdown = 2;
        this.client.sendCommand({ type: 'set_vis_mode', mode });
      },
      onSliceChange: (enabled, normal, offsetKm) => {
        this.sliceEnabled = enabled;
        this.sliceNormal.set(normal[0], normal[1], normal[2]);
        if (this.sliceNormal.lengthSq() > 1.0e-12) {
          this.sliceNormal.normalize();
        } else {
          this.sliceNormal.set(0.0, 1.0, 0.0);
        }
        this.sliceOffsetKm = offsetKm;
        this.glCheckCountdown = 2;
        const comKm = this.latestTelemetry?.livingBodyComKm ?? [0.0, 0.0, 0.0];
        this.sphRenderer.setSlicePlane(
          this.sliceEnabled,
          this.sliceNormal,
          this.sliceOffsetKm,
          comKm
        );
        this.client.sendCommand({
          type: 'set_slice_plane',
          enabled,
          normal: [this.sliceNormal.x, this.sliceNormal.y, this.sliceNormal.z],
          offsetKm,
        });
      },
      onCameraMode: (mode, anchor) => {
        this.cameraCtrl.setMode(mode, anchor);
        if (mode === 'SurfaceLock' && (anchor !== undefined || this.cameraCtrl.altitudeKm > 5.0)) {
          this.cameraCtrl.setTargetAltitudeKm(0.045);
        } else if (mode === 'OrbitalFree' && this.cameraCtrl.altitudeKm < 200.0) {
          this.cameraCtrl.setTargetAltitudeKm(18000.0);
        }
        this.resetChunkCaches();
        this.controlPanel.setCameraReadout(
          this.cameraCtrl.mode,
          this.cameraCtrl.altitudeKm
        );
      },
      onExport: (format) => {
        this.client.sendCommand({ type: 'request_export', format });
      },
    });

    this.telemetryCharts = new TelemetryCharts(rightMount);

    this.bindClientEvents();
    this.bindInteractionEvents();
    this.publishDebugState();
  }

  start(): void {
    if (this.disposed || this.started) {
      return;
    }
    this.started = true;
    this.lastAnimTimeMs =
      typeof performance !== 'undefined' ? performance.now() : Date.now();

    if (typeof WebSocket !== 'undefined') {
      this.client.connect();
    }

    this.stepFrame(this.lastAnimTimeMs);

    if (typeof requestAnimationFrame === 'function') {
      const loop = (nowMs: number): void => {
        if (this.disposed) {
          return;
        }
        this.stepFrame(nowMs);
        this.rafId = requestAnimationFrame(loop);
      };
      this.rafId = requestAnimationFrame(loop);
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    if (this.rafId !== null && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }

    if (this.resizeObserver !== null) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }

    for (const cleanup of this.cleanups) {
      cleanup();
    }
    this.cleanups.length = 0;

    this.client.disconnect();
    this.controlPanel.dispose();
    this.telemetryCharts.dispose();

    if (this.hudOverlay !== null && this.hudOverlay.parentNode !== null) {
      this.hudOverlay.parentNode.removeChild(this.hudOverlay);
    }

    this.sphRenderer.dispose();
    this.atmosphere.dispose();
    this.cubeLod.dispose();
    this.flora.dispose();
    this.fauna.dispose();
    this.starfield.geometry.dispose();
    this.starfield.material.dispose();

    if (this.renderer !== null) {
      if (this.renderer.domElement.parentNode !== null) {
        this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
      this.renderer.dispose();
    }
  }

  private createWebGLRenderer(
    width: number,
    height: number
  ): THREE.WebGLRenderer | null {
    try {
      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        logarithmicDepthBuffer: true,
        powerPreference: 'high-performance',
      });
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.05;
      const dpr =
        typeof window !== 'undefined' && Number.isFinite(window.devicePixelRatio)
          ? Math.min(Math.max(window.devicePixelRatio, 1), 2)
          : 1;
      renderer.setPixelRatio(dpr);
      renderer.setSize(width, height);
      if (typeof this.canvasContainer.appendChild === 'function') {
        this.canvasContainer.appendChild(renderer.domElement);
      }
      if (typeof renderer.domElement?.addEventListener === 'function') {
        const onContextLost = (ev: Event): void => {
          ev.preventDefault?.();
          this.glErrors += 1;
          this.publishDebugState();
        };
        renderer.domElement.addEventListener('webglcontextlost', onContextLost);
        this.cleanups.push(() => {
          if (typeof renderer.domElement?.removeEventListener === 'function') {
            renderer.domElement.removeEventListener(
              'webglcontextlost',
              onContextLost
            );
          }
        });
      }
      return renderer;
    } catch {
      return null;
    }
  }

  private measureCanvasContainer(): { width: number; height: number } {
    const fallbackW =
      typeof window !== 'undefined' && Number.isFinite(window.innerWidth) && window.innerWidth > 0
        ? window.innerWidth
        : 1280;
    const fallbackH =
      typeof window !== 'undefined' && Number.isFinite(window.innerHeight) && window.innerHeight > 0
        ? window.innerHeight
        : 720;
    const w =
      Number.isFinite(this.canvasContainer.clientWidth) &&
      this.canvasContainer.clientWidth > 0
        ? this.canvasContainer.clientWidth
        : fallbackW;
    const h =
      Number.isFinite(this.canvasContainer.clientHeight) &&
      this.canvasContainer.clientHeight > 0
        ? this.canvasContainer.clientHeight
        : fallbackH;
    return { width: w, height: Math.max(h, 1) };
  }

  private bindClientEvents(): void {
    this.cleanups.push(
      this.client.onConnectionState((state) => {
        this.connectionState = state;
        if (this.connectionPill !== null) {
          this.connectionPill.textContent = state;
          this.connectionPill.className =
            state === 'connected'
              ? 'astraea-pill astraea-pill-ok'
              : state === 'reconnecting'
                ? 'astraea-pill astraea-pill-warn'
                : 'astraea-pill astraea-pill-info';
        }
        this.publishDebugState();
      })
    );

    this.cleanups.push(
      this.client.onPresetLoaded((msg) => {
        const switchedPreset =
          this.lastPresetName !== '' && msg.preset !== this.lastPresetName;
        this.lastPresetName = msg.preset;
        this.latestTelemetry = null;
        this.particlePool.count = 0;
        this.tracerPool.count = 0;
        this.coupler.resetBaseline();
        this.latestBio = this.coupler.getGlobalMetrics();
        this.telemetryCharts.clear();
        this.resetChunkCaches();
        if (switchedPreset && this.cameraCtrl.mode === 'SurfaceLock') {
          this.cameraCtrl.setMode('OrbitalFree');
          this.cameraCtrl.setTargetAltitudeKm(18000.0);
          this.controlPanel.setCameraReadout('OrbitalFree', 18000.0);
        }
        if (this.errorToast !== null) {
          this.errorToast.style.display = 'none';
          this.errorToast.textContent = '';
        }
        if (this.topSimInfo !== null) {
          this.topSimInfo.textContent = `${msg.preset} · ${msg.particleCount.toLocaleString('en-US')} pts · ${msg.tracerCount} tracers`;
        }
        this.publishDebugState();
      })
    );

    this.cleanups.push(
      this.client.onFrame((header) => {
        this.lastFrameSeq = header.frameSeq;
        if (this.topSimInfo !== null) {
          const presetLabel =
            this.latestTelemetry?.preset ??
            (this.lastPresetName || 'Simulation');
          this.topSimInfo.textContent = `${presetLabel} · ${header.particleCount.toLocaleString('en-US')} pts · ${header.tracerCount} tracers`;
        }
        this.publishDebugState();
      })
    );

    this.cleanups.push(
      this.client.onTelemetry((dto) => {
        if (this.lastPresetName !== '' && dto.preset !== this.lastPresetName) {
          this.coupler.resetBaseline();
          this.telemetryCharts.clear();
          this.resetChunkCaches();
        }
        this.lastPresetName = dto.preset;
        this.latestTelemetry = dto;
        this.lastFrameSeq = dto.frameSeq;

        this.coupler.ingestSimulationState(
          dto,
          this.tracerPool,
          this.particlePool
        );
        const bio = this.coupler.getGlobalMetrics();
        this.latestBio = bio;

        this.telemetryCharts.pushSample(dto, bio);
        this.controlPanel.updateState(dto, bio);
        this.publishDebugState();
      })
    );

    this.cleanups.push(
      this.client.onExportData((msg) => {
        this.triggerExportDownload(msg.filename, msg.format, msg.content);
      })
    );

    this.cleanups.push(
      this.client.onError((err) => {
        if (this.errorToast !== null) {
          this.errorToast.textContent = `${err.code}: ${err.message}`;
          this.errorToast.style.display = 'inline-block';
        }
      })
    );
  }

  private bindInteractionEvents(): void {
    const pointerTarget: EventTarget =
      this.renderer?.domElement ?? this.canvasContainer;

    let dragging = false;
    let lastClientX = 0;
    let lastClientY = 0;

    const onWheel = (rawEv: Event): void => {
      const ev = rawEv as WheelEvent;
      ev.preventDefault?.();
      this.cameraCtrl.zoomByWheel(ev.deltaY);
      this.controlPanel.setCameraReadout(
        this.cameraCtrl.mode,
        this.cameraCtrl.altitudeKm
      );
    };

    const onPointerDown = (rawEv: Event): void => {
      const ev = rawEv as PointerEvent;
      dragging = true;
      lastClientX = Number.isFinite(ev.clientX) ? ev.clientX : 0;
      lastClientY = Number.isFinite(ev.clientY) ? ev.clientY : 0;
    };

    const onPointerUp = (): void => {
      dragging = false;
    };

    const onPointerMove = (rawEv: Event): void => {
      if (!dragging) {
        return;
      }
      const ev = rawEv as PointerEvent;
      const cx = Number.isFinite(ev.clientX) ? ev.clientX : lastClientX;
      const cy = Number.isFinite(ev.clientY) ? ev.clientY : lastClientY;
      const dx =
        Number.isFinite(ev.movementX) && ev.movementX !== 0
          ? ev.movementX
          : cx - lastClientX;
      const dy =
        Number.isFinite(ev.movementY) && ev.movementY !== 0
          ? ev.movementY
          : cy - lastClientY;
      lastClientX = cx;
      lastClientY = cy;
      this.cameraCtrl.orbitByDelta(dx, dy);
    };

    const onKeyDown = (rawEv: Event): void => {
      if (this.cameraCtrl.mode !== 'SurfaceLock') {
        return;
      }
      const ev = rawEv as KeyboardEvent;
      const tag = (ev.target as { tagName?: string } | null)?.tagName ?? '';
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') {
        return;
      }
      const panStep = Math.max(0.0002, Math.min(0.02, this.cameraCtrl.altitudeKm * 0.004));
      switch (ev.key) {
        case 'w':
        case 'W':
        case 'ArrowUp':
          this.cameraCtrl.panSurfaceAnchor(0, panStep);
          break;
        case 's':
        case 'S':
        case 'ArrowDown':
          this.cameraCtrl.panSurfaceAnchor(0, -panStep);
          break;
        case 'a':
        case 'A':
        case 'ArrowLeft':
          this.cameraCtrl.panSurfaceAnchor(-panStep, 0);
          break;
        case 'd':
        case 'D':
        case 'ArrowRight':
          this.cameraCtrl.panSurfaceAnchor(panStep, 0);
          break;
      }
    };

    const onResize = (): void => {
      const dims = this.measureCanvasContainer();
      this.cameraCtrl.setAspect(dims.width / Math.max(dims.height, 1));
      this.sphRenderer.setViewportHeight(dims.height);
      if (this.renderer !== null) {
        this.renderer.setSize(dims.width, dims.height);
      }
    };

    if (typeof pointerTarget.addEventListener === 'function') {
      pointerTarget.addEventListener('wheel', onWheel, { passive: false });
      pointerTarget.addEventListener('pointerdown', onPointerDown);
      this.cleanups.push(() => {
        if (typeof pointerTarget.removeEventListener === 'function') {
          pointerTarget.removeEventListener('wheel', onWheel);
          pointerTarget.removeEventListener('pointerdown', onPointerDown);
        }
      });
    }

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('pointerup', onPointerUp);
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('keydown', onKeyDown);
      window.addEventListener('resize', onResize);
      this.cleanups.push(() => {
        if (typeof window.removeEventListener === 'function') {
          window.removeEventListener('pointerup', onPointerUp);
          window.removeEventListener('pointermove', onPointerMove);
          window.removeEventListener('keydown', onKeyDown);
          window.removeEventListener('resize', onResize);
        }
      });
    }

    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => onResize());
      try {
        this.resizeObserver.observe(this.canvasContainer);
      } catch {
        // Ignore if canvasContainer is a minimal non-DOM Element mock
      }
    }
  }

  private resetChunkCaches(): void {
    this.lastFaunaTileKey = '';
    this.lastFloraTileKey = '';
    this.lastFloraCam[0] = NaN;
    this.lastFloraCam[1] = NaN;
    this.lastFloraCam[2] = NaN;
    this.lastFloraCom[0] = NaN;
    this.lastFloraCom[1] = NaN;
    this.lastFloraCom[2] = NaN;
    this.lastFloraSpin = NaN;
  }

  private stepFrame(nowMs: number): void {
    const dtSec = Math.min(0.1, Math.max(0.0, (nowMs - this.lastAnimTimeMs) * 0.001));
    this.lastAnimTimeMs = nowMs;

    const tel = this.latestTelemetry;
    const livingComKm: [number, number, number] = tel?.livingBodyComKm ?? [
      0.0, 0.0, 0.0,
    ];
    const livingRadiusKm =
      tel !== null && Number.isFinite(tel.livingBodyRadiusKm) && tel.livingBodyRadiusKm > 0
        ? tel.livingBodyRadiusKm
        : 6371.0;
    const simTimeSec = tel?.simTimeSec ?? 0.0;
    const spinAngleRad = (tel?.livingBodySpinRadS ?? 0.0) * simTimeSec;
    const livingPresent = tel?.livingBodyPresent ?? this.tracerPool.count > 0;

    this.cubeLod.setSimTimeSec(simTimeSec);
    this.cameraCtrl.update(
      dtSec,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      (face, u, v) =>
        this.cubeLod.sampleTracerField(face, u, v, this.tracerPool).dispKm,
      this.tracerPool.count > 0
        ? (face, u, v) =>
            this.cubeLod.sampleSurfacePointKm(
              face,
              u,
              v,
              livingComKm,
              livingRadiusKm,
              spinAngleRad,
              this.tracerPool
            )
        : undefined
    );

    this.controlPanel.setCameraReadout(
      this.cameraCtrl.mode,
      this.cameraCtrl.altitudeKm
    );

    this.sphRenderer.setSlicePlane(
      this.sliceEnabled,
      this.sliceNormal,
      this.sliceOffsetKm,
      livingComKm
    );

    const showLivingLod =
      livingPresent && this.visMode === 'NaturalBlackbody' && !this.sliceEnabled;

    if (showLivingLod) {
      this.cubeLod.group.visible = true;
      this.cubeLod.update(
        this.cameraCtrl,
        livingComKm,
        livingRadiusKm,
        spinAngleRad,
        this.tracerPool,
        this.coupler
      );
      const activePatches = this.cubeLod.getActivePatchCount();
      this.sphRenderer.setLivingLodActive(activePatches > 0);

      this.atmosphere.update(
        this.cameraCtrl.worldPositionKm,
        livingComKm,
        livingRadiusKm,
        this.sunDir,
        this.latestBio.globalOpticalDepthTau,
        tel?.maxTempK ?? 288.0
      );

      this.updateSurfaceBiosphere(
        dtSec,
        simTimeSec,
        livingComKm,
        livingRadiusKm,
        spinAngleRad,
        tel
      );
    } else {
      this.cubeLod.group.visible = false;
      this.flora.group.visible = false;
      this.fauna.group.visible = false;
      this.atmosphere.mesh.visible = false;
      this.sphRenderer.setLivingLodActive(false);
    }

    this.sphRenderer.updateFromPool(
      this.particlePool,
      this.cameraCtrl.worldPositionKm
    );

    if (this.renderer !== null) {
      this.renderer.render(this.scene, this.cameraCtrl.camera);
      if (this.glCheckCountdown > 0) {
        this.glCheckCountdown -= 1;
        if (this.glCheckCountdown === 0) {
          const gl = this.renderer.getContext?.();
          if (gl && typeof gl.getError === 'function') {
            const err = gl.getError();
            if (err !== 0) {
              this.glErrors += 1;
            }
          }
        }
      }
    }

    this.publishDebugState();
  }

  private updateSurfaceBiosphere(
    dtSec: number,
    simTimeSec: number,
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    tel: TelemetrySnapshotDto | null
  ): void {
    const camPos = this.cameraCtrl.worldPositionKm;
    let anchorCoord: SurfaceAnchorCoord;
    if (this.cameraCtrl.mode === 'SurfaceLock') {
      anchorCoord = this.cameraCtrl.surfaceAnchor;
    } else {
      this.tmpCamVec.set(
        camPos[0] - livingComKm[0],
        camPos[1] - livingComKm[1],
        camPos[2] - livingComKm[2]
      );
      if (this.tmpCamVec.lengthSq() > 1.0e-8) {
        this.tmpCamVec.normalize();
        rotateYUnitNormal(
          this.tmpCamVec.x,
          this.tmpCamVec.y,
          this.tmpCamVec.z,
          -spinAngleRad,
          this.tmpAnchorRest
        );
        anchorCoord = unitNormalToCubeFaceUv(
          this.tmpAnchorRest.x,
          this.tmpAnchorRest.y,
          this.tmpAnchorRest.z
        );
      } else {
        anchorCoord = this.cameraCtrl.surfaceAnchor;
      }
    }

    cubeFaceUvToUnitNormal(
      anchorCoord.face,
      anchorCoord.u,
      anchorCoord.v,
      this.tmpAnchorRest
    );
    rotateYUnitNormal(
      this.tmpAnchorRest.x,
      this.tmpAnchorRest.y,
      this.tmpAnchorRest.z,
      spinAngleRad,
      this.tmpAnchorWorld
    );

    const exposure = this.coupler.evaluateSiteExposure(
      this.tmpAnchorWorld,
      simTimeSec
    );
    const showBio =
      (this.cameraCtrl.mode === 'SurfaceLock' ||
        this.cameraCtrl.altitudeKm < 2500.0) &&
      exposure.crustDamage < 0.85;

    if (!showBio) {
      this.flora.group.visible = false;
      this.fauna.group.visible = false;
      return;
    }

    this.flora.group.visible = true;
    this.fauna.group.visible = true;

    const qx = Math.round(this.tmpAnchorRest.x * 1500.0);
    const qy = Math.round(this.tmpAnchorRest.y * 1500.0);
    const qz = Math.round(this.tmpAnchorRest.z * 1500.0);
    const tileKey = `${anchorCoord.face}:${qx}:${qy}:${qz}:${Math.round(livingRadiusKm)}`;

    if (tileKey !== this.lastFaunaTileKey) {
      this.lastFaunaTileKey = tileKey;
      this.fauna.spawnChunkAgents(
        this.tmpAnchorRest,
        livingRadiusKm,
        this.biomeSystem.seed + hashTileKey(tileKey)
      );
    }

    const floraNeedsUpdate =
      tileKey !== this.lastFloraTileKey ||
      Math.abs(camPos[0] - this.lastFloraCam[0]) > 1.0e-5 ||
      Math.abs(camPos[1] - this.lastFloraCam[1]) > 1.0e-5 ||
      Math.abs(camPos[2] - this.lastFloraCam[2]) > 1.0e-5 ||
      Math.abs(livingComKm[0] - this.lastFloraCom[0]) > 1.0e-5 ||
      Math.abs(livingComKm[1] - this.lastFloraCom[1]) > 1.0e-5 ||
      Math.abs(livingComKm[2] - this.lastFloraCom[2]) > 1.0e-5 ||
      Math.abs(spinAngleRad - this.lastFloraSpin) > 1.0e-7;

    if (floraNeedsUpdate) {
      this.lastFloraTileKey = tileKey;
      this.lastFloraCam[0] = camPos[0];
      this.lastFloraCam[1] = camPos[1];
      this.lastFloraCam[2] = camPos[2];
      this.lastFloraCom[0] = livingComKm[0];
      this.lastFloraCom[1] = livingComKm[1];
      this.lastFloraCom[2] = livingComKm[2];
      this.lastFloraSpin = spinAngleRad;
      this.flora.populateChunk(
        this.tmpAnchorRest,
        livingRadiusKm,
        camPos,
        livingComKm,
        spinAngleRad
      );
    }

    const impCom = tel?.impactorComKm ?? [0.0, 0.0, 0.0];
    const dx = impCom[0] - livingComKm[0];
    const dy = impCom[1] - livingComKm[1];
    const dz = impCom[2] - livingComKm[2];
    const sepKm = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (sepKm > 1.0) {
      this.tmpImpAzimuth.set(dx / sepKm, dy / sepKm, dz / sepKm);
    } else {
      this.tmpImpAzimuth.set(1.0, 0.0, 0.0);
    }

    this.tmpBlastDir.copy(this.tmpAnchorWorld).sub(this.tmpImpAzimuth);
    if (this.tmpBlastDir.lengthSq() > 1.0e-8) {
      this.tmpBlastDir.normalize();
    } else {
      this.tmpBlastDir.set(-1.0, 0.0, 0.0);
    }

    this.flora.updateImpactResponse(
      simTimeSec,
      exposure.pgaG,
      exposure.overpressureKpa,
      this.tmpBlastDir,
      exposure.effectiveTempK,
      exposure.radFluxKwM2
    );

    const impactorDistRadii =
      tel === null
        ? 10.0
        : sepKm > 1.0
          ? sepKm / Math.max(livingRadiusKm, 1.0)
          : tel.outcome === 'Approaching'
            ? 10.0
            : 0.5;
    const tImpactElapsedSec =
      exposure.pWaveArrived || exposure.blastArrived || exposure.crustDamage > 0.05
        ? simTimeSec
        : 0.0;

    const stim: FaunaStimulus = {
      tImpactElapsedSec,
      impactorDistRadii,
      pgaG: exposure.pgaG,
      overpressureKpa: exposure.overpressureKpa,
      localTempK: exposure.effectiveTempK,
      radFluxKwM2: exposure.radFluxKwM2,
      ejectaFluxKgM2: exposure.ejectaFluxKgM2,
      tsunamiHeightM: exposure.tsunamiHeightM,
      crustDamage: exposure.crustDamage,
      impactAzimuthLocal: this.tmpImpAzimuth,
    };

    const stepDt = tel?.state === 'paused' ? 0.0 : dtSec;
    this.fauna.step(
      stepDt,
      simTimeSec,
      stim,
      camPos,
      livingComKm,
      livingRadiusKm,
      spinAngleRad
    );
  }

  private triggerExportDownload(
    filename: string,
    format: 'csv' | 'json',
    content: string
  ): void {
    const doc = getOwnerDoc(this.uiContainer);
    if (
      doc === null ||
      typeof Blob === 'undefined' ||
      typeof URL === 'undefined' ||
      typeof URL.createObjectURL !== 'function'
    ) {
      return;
    }
    const mime = format === 'json' ? 'application/json' : 'text/csv';
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = doc.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    doc.body?.appendChild(link);
    link.click();
    if (link.parentNode !== null) {
      link.parentNode.removeChild(link);
    }
    URL.revokeObjectURL?.(url);
  }

  private publishDebugState(): void {
    if (typeof window === 'undefined') {
      return;
    }
    const debugState: AstraeaDebugState = {
      visMode: this.visMode,
      sliceEnabled: this.sliceEnabled,
      cameraMode: this.cameraCtrl.mode,
      altitudeKm: this.cameraCtrl.altitudeKm,
      particleCount: this.particlePool.count,
      tracerCount: this.tracerPool.count,
      activePatches: this.cubeLod.getActivePatchCount(),
      culledDamagedPatches: this.cubeLod.getCulledDamagedPatchCount(),
      floraVisible: this.flora.group.visible,
      faunaVisible: this.fauna.group.visible,
      survivalPct: this.latestBio.overallSurvivalPct,
      extinctionTier: this.latestBio.extinctionTier,
      frameSeq: this.lastFrameSeq,
      simTimeSec: this.latestTelemetry?.simTimeSec ?? 0.0,
      connectionState: this.connectionState,
      glErrors: this.glErrors,
      rendererReady: this.renderer !== null,
    };
    (window as unknown as { __ASTRAEA_DEBUG__?: AstraeaDebugState }).__ASTRAEA_DEBUG__ =
      debugState;
  }
}
