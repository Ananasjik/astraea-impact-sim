import * as THREE from 'three';
import {
  createParticleBufferPool,
  createTracerBufferPool,
} from './net/BinaryProtocol';
import { SimClient, type TelemetrySnapshotDto } from './net/SimClient';
import { OrbitalSurfaceCamera } from './render/OrbitalSurfaceCamera';
import { SphParticleRenderer } from './render/SphParticleRenderer';
import { AtmosphereShader } from './render/AtmosphereShader';
import {
  CubeSphereLod,
  type BiomeSurfaceSampler,
  type ImpactSiteSampler,
} from './render/CubeSphereLod';

const MAX_PARTICLES = 250_000;
const MAX_TRACERS = 6 * 32 * 32;

const fallbackBiomeSampler: BiomeSurfaceSampler = {
  getElevationM(unitNormal: THREE.Vector3): number {
    const continent =
      Math.sin(unitNormal.x * 3.1 + 0.4) *
      Math.cos(unitNormal.y * 2.7 - 0.2) *
      Math.sin(unitNormal.z * 3.5 + 0.8);
    return continent * 3200.0;
  },
  sampleSurface(unitNormal: THREE.Vector3) {
    const elevationM = this.getElevationM(unitNormal);
    const latitudeRad = Math.asin(Math.max(-1.0, Math.min(1.0, unitNormal.y)));
    const isOcean = elevationM < 0.0;
    return {
      elevationM,
      latitudeRad,
      moisture01: isOcean ? 1.0 : 0.55,
      climateTempK: 288.0 - 25.0 * Math.abs(Math.sin(latitudeRad)),
      biome: isOcean ? ('OceanReef' as const) : ('TemperateForest' as const),
      albedoRgb: isOcean
        ? ([0.06, 0.24, 0.48] as [number, number, number])
        : ([0.20, 0.42, 0.22] as [number, number, number]),
    };
  },
};

const fallbackSiteSampler: ImpactSiteSampler = {
  evaluateSiteExposure(unitNormal: THREE.Vector3) {
    return {
      siteNormal: unitNormal.clone(),
      arcDistanceM: 0.0,
      chordDistanceM: 0.0,
      tsunamiHeightM: 0.0,
      pWaveArrived: false,
      sWaveArrived: false,
      pgaG: 0.0,
      radFluxKwM2: 0.0,
      isWildfireIgnited: false,
      ejectaArrived: false,
      ejectaFluxKgM2: 0.0,
      blastArrived: false,
      overpressureKpa: 0.0,
      effectiveTempK: 288.0,
      crustDamage: 0.0,
    };
  },
};

function bootstrap(): void {
  const root = document.getElementById('app');
  if (root === null) {
    return;
  }

  const particlePool = createParticleBufferPool(MAX_PARTICLES);
  const tracerPool = createTracerBufferPool(MAX_TRACERS);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    logarithmicDepthBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(root.clientWidth || window.innerWidth, root.clientHeight || window.innerHeight);
  root.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const cameraCtrl = new OrbitalSurfaceCamera(
    (root.clientWidth || window.innerWidth) / Math.max(root.clientHeight || window.innerHeight, 1)
  );
  const sphRenderer = new SphParticleRenderer(MAX_PARTICLES);
  sphRenderer.setViewportHeight(root.clientHeight || window.innerHeight);
  const atmosphere = new AtmosphereShader(64);
  const cubeLod = new CubeSphereLod(fallbackBiomeSampler);

  scene.add(cubeLod.group);
  scene.add(sphRenderer.mesh);
  scene.add(atmosphere.mesh);

  const sunDir = new THREE.Vector3(0.6, 0.5, 0.624).normalize();
  let latestTelemetry: TelemetrySnapshotDto | null = null;
  let lastTimeMs = performance.now();

  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const client = new SimClient(
    `${wsProtocol}//${window.location.host}/ws`,
    particlePool,
    tracerPool
  );
  client.onTelemetry((dto) => {
    latestTelemetry = dto;
  });
  client.connect();

  const onWheel = (ev: WheelEvent): void => {
    ev.preventDefault();
    cameraCtrl.zoomByWheel(ev.deltaY);
  };
  let dragging = false;
  const onPointerDown = (): void => {
    dragging = true;
  };
  const onPointerUp = (): void => {
    dragging = false;
  };
  const onPointerMove = (ev: PointerEvent): void => {
    if (dragging) {
      cameraCtrl.orbitByDelta(ev.movementX, ev.movementY);
    }
  };
  const onResize = (): void => {
    const w = root.clientWidth || window.innerWidth;
    const h = Math.max(root.clientHeight || window.innerHeight, 1);
    renderer.setSize(w, h);
    cameraCtrl.setAspect(w / h);
    sphRenderer.setViewportHeight(h);
  };

  renderer.domElement.addEventListener('wheel', onWheel, { passive: false });
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointermove', onPointerMove);
  window.addEventListener('resize', onResize);

  const animate = (nowMs: number): void => {
    const dtSec = Math.min(0.1, Math.max(0.0, (nowMs - lastTimeMs) * 0.001));
    lastTimeMs = nowMs;

    const livingComKm: [number, number, number] =
      latestTelemetry?.livingBodyComKm ?? [0.0, 0.0, 0.0];
    const livingRadiusKm = latestTelemetry?.livingBodyRadiusKm ?? 6371.0;
    const simTimeSec = latestTelemetry?.simTimeSec ?? 0.0;
    const spinAngleRad = (latestTelemetry?.livingBodySpinRadS ?? 0.0) * simTimeSec;
    const livingPresent = latestTelemetry?.livingBodyPresent ?? true;

    cubeLod.setSimTimeSec(simTimeSec);
    cameraCtrl.update(
      dtSec,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      (face, u, v) => cubeLod.sampleTracerField(face, u, v, tracerPool).dispKm,
      tracerPool.count > 0
        ? (face, u, v) =>
            cubeLod.sampleSurfacePointKm(
              face,
              u,
              v,
              livingComKm,
              livingRadiusKm,
              spinAngleRad,
              tracerPool
            )
        : undefined
    );

    if (livingPresent) {
      cubeLod.group.visible = true;
      cubeLod.update(
        cameraCtrl,
        livingComKm,
        livingRadiusKm,
        spinAngleRad,
        tracerPool,
        fallbackSiteSampler
      );
      sphRenderer.setLivingLodActive(true);
      atmosphere.update(
        cameraCtrl.worldPositionKm,
        livingComKm,
        livingRadiusKm,
        sunDir,
        0.0,
        latestTelemetry?.maxTempK ?? 288.0
      );
    } else {
      cubeLod.group.visible = false;
      sphRenderer.setLivingLodActive(false);
      atmosphere.mesh.visible = false;
    }

    sphRenderer.updateFromPool(particlePool, cameraCtrl.worldPositionKm);
    renderer.render(scene, cameraCtrl.camera);
    requestAnimationFrame(animate);
  };

  requestAnimationFrame(animate);
}

bootstrap();
