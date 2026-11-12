import * as THREE from 'three';
import {
  TRACER_FLAG_OCEAN,
  type TracerBufferPool,
} from '../net/BinaryProtocol';
import {
  cubeFaceUvToUnitNormal,
  rotateYUnitNormal,
  unitNormalToCubeFaceUv,
  type OrbitalSurfaceCamera,
} from './OrbitalSurfaceCamera';

export type LodBiomeId =
  | 'TropicalRainforest'
  | 'Savanna'
  | 'TemperateForest'
  | 'BorealTaiga'
  | 'AlpineTundra'
  | 'AridSteppe'
  | 'OceanReef';

export interface BiomeSurfaceSample {
  elevationM: number;
  latitudeRad: number;
  moisture01: number;
  climateTempK: number;
  biome: LodBiomeId;
  albedoRgb: [number, number, number];
}

export interface BiomeSurfaceSampler {
  sampleSurface(unitNormal: THREE.Vector3): BiomeSurfaceSample;
  getElevationM(unitNormal: THREE.Vector3): number;
}

export interface LodSitePhysicalExposure {
  siteNormal: THREE.Vector3;
  arcDistanceM: number;
  chordDistanceM: number;
  tsunamiHeightM: number;
  pWaveArrived: boolean;
  sWaveArrived: boolean;
  pgaG: number;
  radFluxKwM2: number;
  isWildfireIgnited: boolean;
  ejectaArrived: boolean;
  ejectaFluxKgM2: number;
  blastArrived: boolean;
  overpressureKpa: number;
  effectiveTempK: number;
  crustDamage: number;
}

export interface ImpactSiteSampler {
  evaluateSiteExposure(
    unitNormal: THREE.Vector3,
    simTimeSec: number
  ): LodSitePhysicalExposure;
}

export interface TracerFieldSample {
  dispKm: number;
  tempK: number;
  damage: number;
  overpressureKpa: number;
  tsunamiHeightM: number;
  isOcean: boolean;
}

const PATCH_SEGMENTS = 16;
const PATCH_GRID = PATCH_SEGMENTS + 1;
const VERTICES_PER_PATCH = PATCH_GRID * PATCH_GRID;
const MAX_ACTIVE_LEAVES = 192;
const MAX_REST_CACHE_ENTRIES = 640;
const SPLIT_THRESHOLD = 0.24;
const CRACK_DISSOLVE_MIN_DAMAGE = 0.55;
const SHATTER_CULL_DAMAGE = 0.85;

const LOD_VERTEX_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>

attribute vec3 aAlbedo;
attribute float aDamage;
attribute float aTempK;
attribute vec2 aOceanTsunami;

uniform float uDissolveProgress;

varying vec3 vNormal;
varying vec3 vAlbedo;
varying float vDamage;
varying float vTempK;
varying vec2 vOceanTsunami;
varying vec2 vFaceUv;

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  vNormal = normal;
  vAlbedo = aAlbedo;
  vDamage = aDamage;
  vTempK = aTempK;
  vOceanTsunami = aOceanTsunami;
  vFaceUv = uv;

  vec3 displaced = position;
  if (uDissolveProgress > 0.0) {
    float cellJitter = hash21(floor(uv * 160.0));
    displaced += normal * (uDissolveProgress * 1.5 * cellJitter);
  }

  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);

  #include <logdepthbuf_vertex>
}
`;

const LOD_FRAGMENT_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>

uniform vec3 uSunDir;
uniform float uDissolveProgress;
uniform float uCrackProgress;
uniform float uWildfireGlow;
uniform float uSimTimeSec;

varying vec3 vNormal;
varying vec3 vAlbedo;
varying float vDamage;
varying float vTempK;
varying vec2 vOceanTsunami;
varying vec2 vFaceUv;

vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

float noise2D(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = fract(sin(dot(i, vec2(127.1, 311.7))) * 43758.5453);
  float b = fract(sin(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7))) * 43758.5453);
  float c = fract(sin(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);
  float d = fract(sin(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbmDetail(vec2 p) {
  float sum = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 4; i++) {
    sum += amp * noise2D(p);
    p *= 2.07;
    amp *= 0.5;
  }
  return sum;
}

// Returns vec2(edgeDistance, cellId)
vec2 voronoiCrack(vec2 x) {
  vec2 n = floor(x);
  vec2 f = fract(x);

  vec2 mg = vec2(0.0);
  vec2 mr = vec2(0.0);
  float md = 8.0;
  float cellId = 0.0;

  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 g = vec2(float(i), float(j));
      vec2 o = hash22(n + g);
      vec2 r = g + o - f;
      float d = dot(r, r);
      if (d < md) {
        md = d;
        mr = r;
        mg = g;
        cellId = o.x;
      }
    }
  }

  float edgeDist = 8.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 g = mg + vec2(float(i), float(j));
      vec2 o = hash22(n + g);
      vec2 r = g + o - f;
      if (dot(mr - r, mr - r) > 1.0e-5) {
        edgeDist = min(edgeDist, dot(0.5 * (mr + r), normalize(r - mr)));
      }
    }
  }

  return vec2(edgeDist, cellId);
}

void main() {
  #include <logdepthbuf_fragment>

  vec3 n = normalize(vNormal);
  float ndotl = max(dot(n, uSunDir), 0.0);
  float wrapLight = 0.14 + 0.86 * smoothstep(-0.15, 0.95, dot(n, uSunDir));

  float detail = fbmDetail(vFaceUv * 96.0);
  float microDetail = fbmDetail(vFaceUv * 1200.0);
  float detailMod = 0.86 + 0.18 * detail + 0.10 * microDetail;

  vec3 baseColor = vAlbedo * detailMod * wrapLight;

  // Ocean shading & animated wave crests
  if (vOceanTsunami.x > 0.5) {
    float wavePhase = noise2D(vFaceUv * 420.0 + vec2(uSimTimeSec * 0.25, -uSimTimeSec * 0.18));
    float tsunamiFoam = smoothstep(5.0, 80.0, vOceanTsunami.y) * (0.4 + 0.6 * wavePhase);
    vec3 oceanDeep = vec3(0.03, 0.16, 0.36);
    vec3 oceanShallow = vec3(0.08, 0.36, 0.58);
    vec3 waterCol = mix(oceanDeep, oceanShallow, 0.35 + 0.45 * wavePhase) * wrapLight;
    float spec = pow(max(ndotl, 0.0), 28.0) * 0.45;
    baseColor = mix(waterCol + vec3(0.35, 0.45, 0.55) * spec, vec3(0.88, 0.94, 0.98), tsunamiFoam * 0.75);
  }

  // Wildfire charring and ember glow
  if (uWildfireGlow > 0.05 && vOceanTsunami.x < 0.5) {
    vec3 charCol = vec3(0.07, 0.06, 0.05) * wrapLight;
    float emberMask = smoothstep(0.62, 0.92, microDetail) * uWildfireGlow;
    baseColor = mix(baseColor, charCol, clamp(uWildfireGlow * 0.85, 0.0, 0.9));
    baseColor += vec3(1.0, 0.36, 0.04) * emberMask * 1.4;
  }

  // Thermal crust heating glow
  float heatGlow = smoothstep(700.0, 2600.0, vTempK);
  if (heatGlow > 0.0) {
    vec3 magmaCol = mix(vec3(0.92, 0.18, 0.02), vec3(1.0, 0.82, 0.32), smoothstep(1400.0, 3200.0, vTempK));
    baseColor = mix(baseColor, magmaCol, heatGlow * 0.85);
  }

  // Voronoi crack-dissolve transition (D in [0.55, 0.85))
  float localDissolve = max(
    uDissolveProgress,
    vDamage >= 0.55 ? clamp((vDamage - 0.55) / 0.30, 0.0, 1.0) : 0.0
  );
  if (localDissolve > 0.0 || uCrackProgress > 0.0) {
    vec2 vor = voronoiCrack(vFaceUv * 128.0);
    float edgeDist = vor.x;
    float cellRand = vor.y;
    float fissureWidth = mix(0.03, 0.24, localDissolve);

    if (cellRand < localDissolve * 0.88 && edgeDist >= fissureWidth * 0.65) {
      discard;
    }

    float fissureMask = 1.0 - smoothstep(0.0, fissureWidth, edgeDist);
    float charFactor = smoothstep(0.0, 0.45, localDissolve);
    baseColor = mix(baseColor, vec3(0.08, 0.05, 0.04), charFactor * 0.75);

    vec3 fissureGlow = mix(vec3(1.0, 0.28, 0.03), vec3(1.0, 0.90, 0.48), fissureMask);
    baseColor = mix(baseColor, fissureGlow * 1.4, clamp(fissureMask * (0.45 + 0.55 * localDissolve), 0.0, 1.0));
  }

  gl_FragColor = vec4(baseColor, 1.0);
}
`;

interface QuadNode {
  face: number;
  depth: number;
  qx: number;
  qy: number;
  u0: number;
  v0: number;
  size: number;
  score: number;
  centerWorldKm: [number, number, number];
}

interface CachedPatchRest {
  unitNormals: Float32Array; // 3 * VERTICES_PER_PATCH
  elevKm: Float32Array;      // VERTICES_PER_PATCH
  albedoRgb: Float32Array;   // 3 * VERTICES_PER_PATCH
  faceUv: Float32Array;      // 2 * VERTICES_PER_PATCH
  centerNormal: THREE.Vector3;
}

interface PatchMeshSlot {
  mesh: THREE.Mesh;
  geometry: THREE.BufferGeometry;
  material: THREE.ShaderMaterial;
  posAttr: THREE.BufferAttribute;
  normalAttr: THREE.BufferAttribute;
  albedoAttr: THREE.BufferAttribute;
  damageAttr: THREE.BufferAttribute;
  tempAttr: THREE.BufferAttribute;
  oceanTsunamiAttr: THREE.BufferAttribute;
  uvAttr: THREE.BufferAttribute;
}

function createSharedPatchIndices(): Uint16Array {
  const indices = new Uint16Array(PATCH_SEGMENTS * PATCH_SEGMENTS * 6);
  let ptr = 0;
  for (let iy = 0; iy < PATCH_SEGMENTS; iy++) {
    for (let ix = 0; ix < PATCH_SEGMENTS; ix++) {
      const a = iy * PATCH_GRID + ix;
      const b = a + 1;
      const c = (iy + 1) * PATCH_GRID + ix;
      const d = c + 1;
      indices[ptr++] = a;
      indices[ptr++] = b;
      indices[ptr++] = d;
      indices[ptr++] = a;
      indices[ptr++] = d;
      indices[ptr++] = c;
    }
  }
  return indices;
}

const SHARED_PATCH_INDEX_ATTR = new THREE.BufferAttribute(
  createSharedPatchIndices(),
  1
);

function clamp(v: number, minVal: number, maxVal: number): number {
  return Math.max(minVal, Math.min(maxVal, v));
}

export class CubeSphereLod {
  readonly group: THREE.Group;
  readonly maxDepth = 10;

  private readonly biomeSystem: BiomeSurfaceSampler;
  private readonly restCache = new Map<number, CachedPatchRest>();
  private readonly meshPool: PatchMeshSlot[] = [];

  private activePatchCount = 0;
  private culledDamagedPatchCount = 0;
  private maxActiveDepth = 0;
  private simTimeSec = 0.0;
  private sunDir = new THREE.Vector3(0.6, 0.5, 0.624).normalize();

  // Reusable tracer anchor index remap when tracers are non-sequential
  private anchorSlotMap: Int32Array = new Int32Array(0);
  private anchorSlotDirect = true;

  // Scratch objects to avoid per-frame allocations
  private readonly tmpNormal = new THREE.Vector3();
  private readonly tmpOutPos: [number, number, number] = [0.0, 0.0, 0.0];
  private readonly tmpCenterPos: [number, number, number] = [0.0, 0.0, 0.0];
  private readonly tmpU0Pos: [number, number, number] = [0.0, 0.0, 0.0];
  private readonly tmpU1Pos: [number, number, number] = [0.0, 0.0, 0.0];
  private readonly tmpV0Pos: [number, number, number] = [0.0, 0.0, 0.0];
  private readonly tmpV1Pos: [number, number, number] = [0.0, 0.0, 0.0];
  private readonly tmpTracerSample: TracerFieldSample = {
    dispKm: 0.0,
    tempK: 288.0,
    damage: 0.0,
    overpressureKpa: 0.0,
    tsunamiHeightM: 0.0,
    isOcean: false,
  };
  private readonly tmpSeamCoord = new THREE.Vector3();

  constructor(biomeSystem: BiomeSurfaceSampler) {
    this.biomeSystem = biomeSystem;
    this.group = new THREE.Group();
    this.group.name = 'CubeSphereLodGroup';
  }

  setSunDirection(sunDir: THREE.Vector3): void {
    this.sunDir.copy(sunDir);
    if (this.sunDir.lengthSq() > 1.0e-12) {
      this.sunDir.normalize();
    } else {
      this.sunDir.set(0.6, 0.5, 0.624).normalize();
    }
  }

  setSimTimeSec(simTimeSec: number): void {
    if (Number.isFinite(simTimeSec)) {
      this.simTimeSec = simTimeSec;
    }
  }

  getActivePatchCount(): number {
    return this.activePatchCount;
  }

  getCulledDamagedPatchCount(): number {
    return this.culledDamagedPatchCount;
  }

  getMaxActiveDepth(): number {
    return this.maxActiveDepth;
  }

  sampleTracerField(
    face: number,
    u: number,
    v: number,
    tracers: TracerBufferPool
  ): TracerFieldSample {
    this.prepareTracerLookup(tracers);
    const out: TracerFieldSample = {
      dispKm: 0.0,
      tempK: 288.0,
      damage: 0.0,
      overpressureKpa: 0.0,
      tsunamiHeightM: 0.0,
      isOcean: false,
    };
    this.writeTracerFieldSample(face, u, v, tracers, out);
    return out;
  }

  sampleSurfacePointKm(
    face: number,
    u: number,
    v: number,
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    tracers: TracerBufferPool,
    outPos: [number, number, number] = [0.0, 0.0, 0.0]
  ): [number, number, number] {
    this.prepareTracerLookup(tracers);
    const cu = clamp(u, 0.0, 1.0);
    const cv = clamp(v, 0.0, 1.0);
    const restN = cubeFaceUvToUnitNormal(face, cu, cv, this.tmpNormal);
    const elevM = this.biomeSystem.getElevationM(restN);

    if (tracers.count > 0) {
      this.sampleTracerPosAndFields(
        face,
        cu,
        cv,
        tracers,
        outPos,
        this.tmpTracerSample
      );
      let ox = outPos[0] - livingComKm[0];
      let oy = outPos[1] - livingComKm[1];
      let oz = outPos[2] - livingComKm[2];
      const len = Math.sqrt(ox * ox + oy * oy + oz * oz);
      if (len > 1.0e-6) {
        const inv = 1.0 / len;
        ox *= inv;
        oy *= inv;
        oz *= inv;
      } else {
        rotateYUnitNormal(restN.x, restN.y, restN.z, spinAngleRad, restN);
        ox = restN.x;
        oy = restN.y;
        oz = restN.z;
      }
      const reliefKm =
        (this.tmpTracerSample.isOcean || elevM < 0.0
          ? this.tmpTracerSample.tsunamiHeightM * 0.001
          : Math.max(0.0, elevM) * 0.001);
      outPos[0] += ox * reliefKm;
      outPos[1] += oy * reliefKm;
      outPos[2] += oz * reliefKm;
      return outPos;
    }

    rotateYUnitNormal(restN.x, restN.y, restN.z, spinAngleRad, restN);
    const rKm = Math.max(livingRadiusKm, 1.0) + Math.max(0.0, elevM) * 0.001;
    outPos[0] = livingComKm[0] + restN.x * rKm;
    outPos[1] = livingComKm[1] + restN.y * rKm;
    outPos[2] = livingComKm[2] + restN.z * rKm;
    return outPos;
  }

  update(
    camera: OrbitalSurfaceCamera,
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    tracers: TracerBufferPool,
    coupler: ImpactSiteSampler
  ): void {
    const safeRadiusKm = Math.max(livingRadiusKm, 1.0);
    this.prepareTracerLookup(tracers);

    // Build priority-sorted quadtree leaves up to maxDepth = 10 and MAX_ACTIVE_LEAVES = 192
    const leaves: QuadNode[] = [];
    for (let f = 0; f < 6; f++) {
      leaves.push(
        this.makeQuadNode(
          f,
          0,
          0,
          0,
          camera.worldPositionKm,
          livingComKm,
          safeRadiusKm,
          spinAngleRad,
          tracers
        )
      );
    }

    while (leaves.length + 3 <= MAX_ACTIVE_LEAVES) {
      let bestIdx = -1;
      let bestScore = SPLIT_THRESHOLD;
      for (let i = 0; i < leaves.length; i++) {
        const node = leaves[i];
        if (node.depth < this.maxDepth && node.score > bestScore) {
          bestScore = node.score;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) {
        break;
      }

      const parent = leaves[bestIdx];
      // Replace parent with its 4 children
      const nextDepth = parent.depth + 1;
      const qx2 = parent.qx * 2;
      const qy2 = parent.qy * 2;
      leaves[bestIdx] = this.makeQuadNode(
        parent.face,
        nextDepth,
        qx2,
        qy2,
        camera.worldPositionKm,
        livingComKm,
        safeRadiusKm,
        spinAngleRad,
        tracers
      );
      leaves.push(
        this.makeQuadNode(
          parent.face,
          nextDepth,
          qx2 + 1,
          qy2,
          camera.worldPositionKm,
          livingComKm,
          safeRadiusKm,
          spinAngleRad,
          tracers
        ),
        this.makeQuadNode(
          parent.face,
          nextDepth,
          qx2,
          qy2 + 1,
          camera.worldPositionKm,
          livingComKm,
          safeRadiusKm,
          spinAngleRad,
          tracers
        ),
        this.makeQuadNode(
          parent.face,
          nextDepth,
          qx2 + 1,
          qy2 + 1,
          camera.worldPositionKm,
          livingComKm,
          safeRadiusKm,
          spinAngleRad,
          tracers
        )
      );
    }

    this.group.clear();
    let activeCount = 0;
    let culledCount = 0;
    let maxDepthSeen = 0;

    for (let i = 0; i < leaves.length; i++) {
      const node = leaves[i];
      if (node.depth > maxDepthSeen) {
        maxDepthSeen = node.depth;
      }

      const rest = this.getOrBuildPatchRest(node);
      const uMid = node.u0 + 0.5 * node.size;
      const vMid = node.v0 + 0.5 * node.size;

      let centerDamage = 0.0;
      if (tracers.count > 0) {
        this.writeTracerFieldSample(
          node.face,
          uMid,
          vMid,
          tracers,
          this.tmpTracerSample
        );
        centerDamage = this.tmpTracerSample.damage;
      }

      let exposure: LodSitePhysicalExposure | null = null;
      if (
        coupler !== undefined &&
        typeof coupler.evaluateSiteExposure === 'function'
      ) {
        rotateYUnitNormal(
          rest.centerNormal.x,
          rest.centerNormal.y,
          rest.centerNormal.z,
          spinAngleRad,
          this.tmpNormal
        );
        exposure = coupler.evaluateSiteExposure(this.tmpNormal, this.simTimeSec);
        if (tracers.count === 0) {
          centerDamage = exposure.crustDamage;
        }
      }

      if (centerDamage >= SHATTER_CULL_DAMAGE) {
        culledCount += 1;
        continue;
      }

      const slot = this.acquireSlot(activeCount);
      this.populatePatchSlot(
        slot,
        node,
        rest,
        centerDamage,
        exposure,
        camera.worldPositionKm,
        livingComKm,
        safeRadiusKm,
        spinAngleRad,
        tracers
      );
      this.group.add(slot.mesh);
      activeCount += 1;
    }

    this.activePatchCount = activeCount;
    this.culledDamagedPatchCount = culledCount;
    this.maxActiveDepth = maxDepthSeen;
  }

  dispose(): void {
    this.group.clear();
    for (const slot of this.meshPool) {
      slot.geometry.dispose();
      slot.material.dispose();
    }
    this.meshPool.length = 0;
    this.restCache.clear();
  }

  private getGridRes(tracers: TracerBufferPool): number {
    if (tracers.gridRes > 0) {
      return tracers.gridRes;
    }
    if (tracers.count > 0) {
      return Math.max(1, Math.round(Math.sqrt(tracers.count / 6)));
    }
    return 0;
  }

  private prepareTracerLookup(tracers: TracerBufferPool): void {
    const count = tracers.count;
    if (count === 0) {
      this.anchorSlotDirect = true;
      return;
    }
    const g = this.getGridRes(tracers);
    const totalBins = 6 * g * g;
    let direct = count === totalBins;
    if (direct) {
      for (let i = 0; i < count; i++) {
        if (tracers.anchorIndex[i] !== i) {
          direct = false;
          break;
        }
      }
    }

    this.anchorSlotDirect = direct;
    if (!direct) {
      if (this.anchorSlotMap.length < totalBins) {
        this.anchorSlotMap = new Int32Array(totalBins);
      }
      this.anchorSlotMap.fill(-1, 0, totalBins);
      for (let i = 0; i < count; i++) {
        const aIdx = tracers.anchorIndex[i];
        if (aIdx < totalBins) {
          this.anchorSlotMap[aIdx] = i;
        }
      }
    }
  }

  private resolveTracerSlot(binIdx: number, tracers: TracerBufferPool): number {
    if (this.anchorSlotDirect) {
      return binIdx < tracers.count ? binIdx : -1;
    }
    return binIdx < this.anchorSlotMap.length ? this.anchorSlotMap[binIdx] : -1;
  }

  private resolveGridCellSlot(
    face: number,
    ix: number,
    iy: number,
    g: number,
    tracers: TracerBufferPool
  ): number {
    let targetFace = face;
    let gx = ix;
    let gy = iy;

    if (ix < 0 || ix >= g || iy < 0 || iy >= g) {
      // Cross cube-face seam smoothly via 3D unit normal projection
      const clampedIx = clamp(ix, 0, g - 1);
      const clampedIy = clamp(iy, 0, g - 1);
      const uCross =
        ix < 0 || ix >= g ? (ix + 0.5) / g : (clampedIx + 0.5) / g;
      const vCross =
        iy < 0 || iy >= g ? (iy + 0.5) / g : (clampedIy + 0.5) / g;
      cubeFaceUvToUnitNormal(face, uCross, vCross, this.tmpSeamCoord);
      const mapped = unitNormalToCubeFaceUv(
        this.tmpSeamCoord.x,
        this.tmpSeamCoord.y,
        this.tmpSeamCoord.z
      );
      targetFace = mapped.face;
      gx = Math.min(g - 1, Math.max(0, Math.floor(mapped.u * g)));
      gy = Math.min(g - 1, Math.max(0, Math.floor(mapped.v * g)));
    }

    const binIdx = targetFace * g * g + gy * g + gx;
    return this.resolveTracerSlot(binIdx, tracers);
  }

  private sampleTracerPosAndFields(
    face: number,
    u: number,
    v: number,
    tracers: TracerBufferPool,
    outPosKm: [number, number, number],
    outField: TracerFieldSample
  ): void {
    const g = this.getGridRes(tracers);
    if (tracers.count === 0 || g <= 0) {
      outPosKm[0] = 0.0;
      outPosKm[1] = 0.0;
      outPosKm[2] = 0.0;
      outField.dispKm = 0.0;
      outField.tempK = 288.0;
      outField.damage = 0.0;
      outField.overpressureKpa = 0.0;
      outField.tsunamiHeightM = 0.0;
      outField.isOcean = false;
      return;
    }

    const fx = u * g - 0.5;
    const fy = v * g - 0.5;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const tx = fx - x0;
    const ty = fy - y0;

    const s00 = this.resolveGridCellSlot(face, x0, y0, g, tracers);
    const s10 = this.resolveGridCellSlot(face, x1, y0, g, tracers);
    const s01 = this.resolveGridCellSlot(face, x0, y1, g, tracers);
    const s11 = this.resolveGridCellSlot(face, x1, y1, g, tracers);

    const w00 = s00 >= 0 ? (1.0 - tx) * (1.0 - ty) : 0.0;
    const w10 = s10 >= 0 ? tx * (1.0 - ty) : 0.0;
    const w01 = s01 >= 0 ? (1.0 - tx) * ty : 0.0;
    const w11 = s11 >= 0 ? tx * ty : 0.0;
    const wSum = w00 + w10 + w01 + w11;

    if (wSum <= 1.0e-9) {
      outPosKm[0] = 0.0;
      outPosKm[1] = 0.0;
      outPosKm[2] = 0.0;
      outField.dispKm = 0.0;
      outField.tempK = 288.0;
      outField.damage = 0.0;
      outField.overpressureKpa = 0.0;
      outField.tsunamiHeightM = 0.0;
      outField.isOcean = false;
      return;
    }

    const invW = 1.0 / wSum;
    const tPos = tracers.positionsKm;
    const tDisp = tracers.dispKm;
    const tTemp = tracers.tempK;
    const tDam = tracers.damage;
    const tOverp = tracers.overpressureKpa;
    const tTsunami = tracers.tsunamiHeightM;
    const tFlags = tracers.flags;

    let px = 0.0;
    let py = 0.0;
    let pz = 0.0;
    let disp = 0.0;
    let temp = 0.0;
    let dam = 0.0;
    let overp = 0.0;
    let tsu = 0.0;
    let oceanWeight = 0.0;

    if (w00 > 0.0) {
      const p3 = s00 * 3;
      px += w00 * tPos[p3];
      py += w00 * tPos[p3 + 1];
      pz += w00 * tPos[p3 + 2];
      disp += w00 * tDisp[s00];
      temp += w00 * tTemp[s00];
      dam += w00 * tDam[s00];
      overp += w00 * tOverp[s00];
      tsu += w00 * tTsunami[s00];
      if ((tFlags[s00] & TRACER_FLAG_OCEAN) !== 0) {
        oceanWeight += w00;
      }
    }
    if (w10 > 0.0) {
      const p3 = s10 * 3;
      px += w10 * tPos[p3];
      py += w10 * tPos[p3 + 1];
      pz += w10 * tPos[p3 + 2];
      disp += w10 * tDisp[s10];
      temp += w10 * tTemp[s10];
      dam += w10 * tDam[s10];
      overp += w10 * tOverp[s10];
      tsu += w10 * tTsunami[s10];
      if ((tFlags[s10] & TRACER_FLAG_OCEAN) !== 0) {
        oceanWeight += w10;
      }
    }
    if (w01 > 0.0) {
      const p3 = s01 * 3;
      px += w01 * tPos[p3];
      py += w01 * tPos[p3 + 1];
      pz += w01 * tPos[p3 + 2];
      disp += w01 * tDisp[s01];
      temp += w01 * tTemp[s01];
      dam += w01 * tDam[s01];
      overp += w01 * tOverp[s01];
      tsu += w01 * tTsunami[s01];
      if ((tFlags[s01] & TRACER_FLAG_OCEAN) !== 0) {
        oceanWeight += w01;
      }
    }
    if (w11 > 0.0) {
      const p3 = s11 * 3;
      px += w11 * tPos[p3];
      py += w11 * tPos[p3 + 1];
      pz += w11 * tPos[p3 + 2];
      disp += w11 * tDisp[s11];
      temp += w11 * tTemp[s11];
      dam += w11 * tDam[s11];
      overp += w11 * tOverp[s11];
      tsu += w11 * tTsunami[s11];
      if ((tFlags[s11] & TRACER_FLAG_OCEAN) !== 0) {
        oceanWeight += w11;
      }
    }

    outPosKm[0] = px * invW;
    outPosKm[1] = py * invW;
    outPosKm[2] = pz * invW;
    outField.dispKm = disp * invW;
    outField.tempK = temp * invW;
    outField.damage = dam * invW;
    outField.overpressureKpa = overp * invW;
    outField.tsunamiHeightM = tsu * invW;
    outField.isOcean = oceanWeight * invW >= 0.5;
  }

  private writeTracerFieldSample(
    face: number,
    u: number,
    v: number,
    tracers: TracerBufferPool,
    outField: TracerFieldSample
  ): void {
    this.sampleTracerPosAndFields(
      face,
      clamp(u, 0.0, 1.0),
      clamp(v, 0.0, 1.0),
      tracers,
      this.tmpOutPos,
      outField
    );
  }

  private evalAnalyticalPosKm(
    face: number,
    u: number,
    v: number,
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    outPos: [number, number, number]
  ): void {
    cubeFaceUvToUnitNormal(face, u, v, this.tmpNormal);
    rotateYUnitNormal(
      this.tmpNormal.x,
      this.tmpNormal.y,
      this.tmpNormal.z,
      spinAngleRad,
      this.tmpNormal
    );
    outPos[0] = livingComKm[0] + this.tmpNormal.x * livingRadiusKm;
    outPos[1] = livingComKm[1] + this.tmpNormal.y * livingRadiusKm;
    outPos[2] = livingComKm[2] + this.tmpNormal.z * livingRadiusKm;
  }

  private evalNodePointKm(
    face: number,
    u: number,
    v: number,
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    tracers: TracerBufferPool,
    outPos: [number, number, number]
  ): void {
    if (tracers.count > 0) {
      this.sampleTracerPosAndFields(
        face,
        u,
        v,
        tracers,
        outPos,
        this.tmpTracerSample
      );
      return;
    }
    this.evalAnalyticalPosKm(
      face,
      u,
      v,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      outPos
    );
  }

  private makeQuadNode(
    face: number,
    depth: number,
    qx: number,
    qy: number,
    cameraPosKm: [number, number, number],
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    tracers: TracerBufferPool
  ): QuadNode {
    const size = 1.0 / (1 << depth);
    const u0 = qx * size;
    const v0 = qy * size;
    const uMid = u0 + 0.5 * size;
    const vMid = v0 + 0.5 * size;

    this.evalNodePointKm(
      face,
      uMid,
      vMid,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      tracers,
      this.tmpCenterPos
    );
    this.evalNodePointKm(
      face,
      u0,
      vMid,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      tracers,
      this.tmpU0Pos
    );
    this.evalNodePointKm(
      face,
      u0 + size,
      vMid,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      tracers,
      this.tmpU1Pos
    );
    this.evalNodePointKm(
      face,
      uMid,
      v0,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      tracers,
      this.tmpV0Pos
    );
    this.evalNodePointKm(
      face,
      uMid,
      v0 + size,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      tracers,
      this.tmpV1Pos
    );

    const eux = this.tmpU1Pos[0] - this.tmpU0Pos[0];
    const euy = this.tmpU1Pos[1] - this.tmpU0Pos[1];
    const euz = this.tmpU1Pos[2] - this.tmpU0Pos[2];
    const euLen2 = eux * eux + euy * euy + euz * euz;

    const evx = this.tmpV1Pos[0] - this.tmpV0Pos[0];
    const evy = this.tmpV1Pos[1] - this.tmpV0Pos[1];
    const evz = this.tmpV1Pos[2] - this.tmpV0Pos[2];
    const evLen2 = evx * evx + evy * evy + evz * evz;

    const patchSpanKm = Math.sqrt(Math.max(euLen2, evLen2));

    const dx = cameraPosKm[0] - this.tmpCenterPos[0];
    const dy = cameraPosKm[1] - this.tmpCenterPos[1];
    const dz = cameraPosKm[2] - this.tmpCenterPos[2];

    const projU = clamp(
      (dx * eux + dy * euy + dz * euz) / Math.max(euLen2, 1.0e-6),
      -0.5,
      0.5
    );
    const projV = clamp(
      (dx * evx + dy * evy + dz * evz) / Math.max(evLen2, 1.0e-6),
      -0.5,
      0.5
    );

    const cx = this.tmpCenterPos[0] + projU * eux + projV * evx;
    const cy = this.tmpCenterPos[1] + projU * euy + projV * evy;
    const cz = this.tmpCenterPos[2] + projU * euz + projV * evz;

    const distRaw = Math.sqrt(
      (cameraPosKm[0] - cx) * (cameraPosKm[0] - cx) +
        (cameraPosKm[1] - cy) * (cameraPosKm[1] - cy) +
        (cameraPosKm[2] - cz) * (cameraPosKm[2] - cz)
    );
    const sagittaKm =
      (patchSpanKm * patchSpanKm) / (8.0 * Math.max(livingRadiusKm, 1.0));
    const effectiveDistKm = Math.max(1.0e-4, distRaw - sagittaKm);
    const score = patchSpanKm / effectiveDistKm;

    return {
      face,
      depth,
      qx,
      qy,
      u0,
      v0,
      size,
      score,
      centerWorldKm: [
        this.tmpCenterPos[0],
        this.tmpCenterPos[1],
        this.tmpCenterPos[2],
      ],
    };
  }

  private getOrBuildPatchRest(node: QuadNode): CachedPatchRest {
    const key =
      (node.face << 24) |
      (node.depth << 20) |
      (node.qx << 10) |
      node.qy;
    const existing = this.restCache.get(key);
    if (existing !== undefined) {
      return existing;
    }

    if (this.restCache.size >= MAX_REST_CACHE_ENTRIES) {
      const oldestKey = this.restCache.keys().next().value;
      if (oldestKey !== undefined) {
        this.restCache.delete(oldestKey);
      }
    }

    const unitNormals = new Float32Array(VERTICES_PER_PATCH * 3);
    const elevKm = new Float32Array(VERTICES_PER_PATCH);
    const albedoRgb = new Float32Array(VERTICES_PER_PATCH * 3);
    const faceUv = new Float32Array(VERTICES_PER_PATCH * 2);

    const nVec = new THREE.Vector3();
    let idx = 0;
    for (let iy = 0; iy < PATCH_GRID; iy++) {
      const v = node.v0 + (iy / PATCH_SEGMENTS) * node.size;
      for (let ix = 0; ix < PATCH_GRID; ix++) {
        const u = node.u0 + (ix / PATCH_SEGMENTS) * node.size;
        cubeFaceUvToUnitNormal(node.face, u, v, nVec);
        const sample = this.biomeSystem.sampleSurface(nVec);

        const i3 = idx * 3;
        unitNormals[i3] = nVec.x;
        unitNormals[i3 + 1] = nVec.y;
        unitNormals[i3 + 2] = nVec.z;

        elevKm[idx] = sample.elevationM * 0.001;

        albedoRgb[i3] = sample.albedoRgb[0];
        albedoRgb[i3 + 1] = sample.albedoRgb[1];
        albedoRgb[i3 + 2] = sample.albedoRgb[2];

        const i2 = idx * 2;
        faceUv[i2] = u;
        faceUv[i2 + 1] = v;

        idx += 1;
      }
    }

    const centerNormal = cubeFaceUvToUnitNormal(
      node.face,
      node.u0 + 0.5 * node.size,
      node.v0 + 0.5 * node.size,
      new THREE.Vector3()
    );

    const cached: CachedPatchRest = {
      unitNormals,
      elevKm,
      albedoRgb,
      faceUv,
      centerNormal,
    };
    this.restCache.set(key, cached);
    return cached;
  }

  private acquireSlot(slotIndex: number): PatchMeshSlot {
    if (slotIndex < this.meshPool.length) {
      return this.meshPool[slotIndex];
    }

    const posAttr = new THREE.BufferAttribute(
      new Float32Array(VERTICES_PER_PATCH * 3),
      3
    );
    posAttr.setUsage(THREE.DynamicDrawUsage);

    const normalAttr = new THREE.BufferAttribute(
      new Float32Array(VERTICES_PER_PATCH * 3),
      3
    );
    normalAttr.setUsage(THREE.DynamicDrawUsage);

    const albedoAttr = new THREE.BufferAttribute(
      new Float32Array(VERTICES_PER_PATCH * 3),
      3
    );
    const damageAttr = new THREE.BufferAttribute(
      new Float32Array(VERTICES_PER_PATCH),
      1
    );
    damageAttr.setUsage(THREE.DynamicDrawUsage);

    const tempAttr = new THREE.BufferAttribute(
      new Float32Array(VERTICES_PER_PATCH),
      1
    );
    tempAttr.setUsage(THREE.DynamicDrawUsage);

    const oceanTsunamiAttr = new THREE.BufferAttribute(
      new Float32Array(VERTICES_PER_PATCH * 2),
      2
    );
    oceanTsunamiAttr.setUsage(THREE.DynamicDrawUsage);

    const uvAttr = new THREE.BufferAttribute(
      new Float32Array(VERTICES_PER_PATCH * 2),
      2
    );

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', posAttr);
    geometry.setAttribute('normal', normalAttr);
    geometry.setAttribute('aAlbedo', albedoAttr);
    geometry.setAttribute('aDamage', damageAttr);
    geometry.setAttribute('aTempK', tempAttr);
    geometry.setAttribute('aOceanTsunami', oceanTsunamiAttr);
    geometry.setAttribute('uv', uvAttr);
    geometry.setIndex(SHARED_PATCH_INDEX_ATTR);

    const material = new THREE.ShaderMaterial({
      vertexShader: LOD_VERTEX_SHADER,
      fragmentShader: LOD_FRAGMENT_SHADER,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      uniforms: {
        uSunDir: { value: this.sunDir.clone() },
        uDissolveProgress: { value: 0.0 },
        uCrackProgress: { value: 0.0 },
        uWildfireGlow: { value: 0.0 },
        uSimTimeSec: { value: 0.0 },
      },
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.renderOrder = 0;

    const slot: PatchMeshSlot = {
      mesh,
      geometry,
      material,
      posAttr,
      normalAttr,
      albedoAttr,
      damageAttr,
      tempAttr,
      oceanTsunamiAttr,
      uvAttr,
    };
    this.meshPool.push(slot);
    return slot;
  }

  private populatePatchSlot(
    slot: PatchMeshSlot,
    node: QuadNode,
    rest: CachedPatchRest,
    centerDamage: number,
    exposure: LodSitePhysicalExposure | null,
    cameraPosKm: [number, number, number],
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    tracers: TracerBufferPool
  ): void {
    const posArr = slot.posAttr.array as Float32Array;
    const normArr = slot.normalAttr.array as Float32Array;
    const albedoArr = slot.albedoAttr.array as Float32Array;
    const damageArr = slot.damageAttr.array as Float32Array;
    const tempArr = slot.tempAttr.array as Float32Array;
    const otArr = slot.oceanTsunamiAttr.array as Float32Array;
    const uvArr = slot.uvAttr.array as Float32Array;

    albedoArr.set(rest.albedoRgb);
    uvArr.set(rest.faceUv);

    const uMid = node.u0 + 0.5 * node.size;
    const vMid = node.v0 + 0.5 * node.size;
    this.sampleSurfacePointKm(
      node.face,
      uMid,
      vMid,
      livingComKm,
      livingRadiusKm,
      spinAngleRad,
      tracers,
      this.tmpCenterPos
    );
    const patchCenterX = this.tmpCenterPos[0];
    const patchCenterY = this.tmpCenterPos[1];
    const patchCenterZ = this.tmpCenterPos[2];

    const cosSpin = Math.cos(spinAngleRad);
    const sinSpin = Math.sin(spinAngleRad);
    const hasTracers = tracers.count > 0;
    const comX = livingComKm[0];
    const comY = livingComKm[1];
    const comZ = livingComKm[2];

    let maxPatchDamage = centerDamage;
    let sumTempK = 0.0;

    let idx = 0;
    for (let iy = 0; iy < PATCH_GRID; iy++) {
      const v = node.v0 + (iy / PATCH_SEGMENTS) * node.size;
      for (let ix = 0; ix < PATCH_GRID; ix++) {
        const u = node.u0 + (ix / PATCH_SEGMENTS) * node.size;
        const i3 = idx * 3;
        const i2 = idx * 2;

        const rnx = rest.unitNormals[i3];
        const rny = rest.unitNormals[i3 + 1];
        const rnz = rest.unitNormals[i3 + 2];
        const elevKm = rest.elevKm[idx];

        let nx = rnx * cosSpin + rnz * sinSpin;
        let ny = rny;
        let nz = -rnx * sinSpin + rnz * cosSpin;

        let wxKm: number;
        let wyKm: number;
        let wzKm: number;
        let vDam = centerDamage;
        let vTemp = exposure !== null ? exposure.effectiveTempK : 288.0;
        let vOcean = elevKm < 0.0 ? 1.0 : 0.0;
        let vTsunami = exposure !== null ? exposure.tsunamiHeightM : 0.0;

        if (hasTracers) {
          this.sampleTracerPosAndFields(
            node.face,
            u,
            v,
            tracers,
            this.tmpOutPos,
            this.tmpTracerSample
          );
          const ox = this.tmpOutPos[0] - comX;
          const oy = this.tmpOutPos[1] - comY;
          const oz = this.tmpOutPos[2] - comZ;
          const oLen = Math.sqrt(ox * ox + oy * oy + oz * oz);
          if (oLen > 1.0e-6) {
            const invO = 1.0 / oLen;
            nx = ox * invO;
            ny = oy * invO;
            nz = oz * invO;
          }
          vDam = this.tmpTracerSample.damage;
          vTemp = this.tmpTracerSample.tempK;
          vOcean = this.tmpTracerSample.isOcean || elevKm < 0.0 ? 1.0 : 0.0;
          vTsunami = Math.max(
            this.tmpTracerSample.tsunamiHeightM,
            exposure !== null ? exposure.tsunamiHeightM : 0.0
          );

          const reliefKm =
            vOcean > 0.5 ? vTsunami * 0.001 : Math.max(0.0, elevKm);
          wxKm = this.tmpOutPos[0] + nx * reliefKm;
          wyKm = this.tmpOutPos[1] + ny * reliefKm;
          wzKm = this.tmpOutPos[2] + nz * reliefKm;
        } else {
          const reliefKm =
            vOcean > 0.5 ? vTsunami * 0.001 : Math.max(0.0, elevKm);
          const rKm = livingRadiusKm + reliefKm;
          wxKm = comX + nx * rKm;
          wyKm = comY + ny * rKm;
          wzKm = comZ + nz * rKm;
        }

        posArr[i3] = wxKm - patchCenterX;
        posArr[i3 + 1] = wyKm - patchCenterY;
        posArr[i3 + 2] = wzKm - patchCenterZ;

        normArr[i3] = nx;
        normArr[i3 + 1] = ny;
        normArr[i3 + 2] = nz;

        damageArr[idx] = vDam;
        tempArr[idx] = vTemp;
        otArr[i2] = vOcean;
        otArr[i2 + 1] = vTsunami;

        if (vDam > maxPatchDamage) {
          maxPatchDamage = vDam;
        }
        sumTempK += vTemp;
        idx += 1;
      }
    }

    slot.posAttr.needsUpdate = true;
    slot.normalAttr.needsUpdate = true;
    slot.albedoAttr.needsUpdate = true;
    slot.damageAttr.needsUpdate = true;
    slot.tempAttr.needsUpdate = true;
    slot.oceanTsunamiAttr.needsUpdate = true;
    slot.uvAttr.needsUpdate = true;

    slot.mesh.position.set(
      patchCenterX - cameraPosKm[0],
      patchCenterY - cameraPosKm[1],
      patchCenterZ - cameraPosKm[2]
    );
    slot.mesh.updateMatrixWorld();

    const effectiveDamage = Math.max(centerDamage, 0.5 * (centerDamage + maxPatchDamage));
    let dissolveProgress = 0.0;
    if (effectiveDamage >= CRACK_DISSOLVE_MIN_DAMAGE) {
      dissolveProgress = clamp(
        Math.max(0.02, (effectiveDamage - CRACK_DISSOLVE_MIN_DAMAGE) / 0.3),
        0.02,
        1.0
      );
    }

    const uniforms = slot.material.uniforms;
    (uniforms.uSunDir.value as THREE.Vector3).copy(this.sunDir);
    uniforms.uDissolveProgress.value = dissolveProgress;
    uniforms.uCrackProgress.value = dissolveProgress;
    uniforms.uWildfireGlow.value =
      exposure !== null && exposure.isWildfireIgnited ? 1.0 : 0.0;
    uniforms.uSimTimeSec.value = this.simTimeSec;

    slot.mesh.userData = {
      face: node.face,
      depth: node.depth,
      damage: centerDamage,
      dissolveProgress,
      crackProgress: dissolveProgress,
      meanTempK: sumTempK / VERTICES_PER_PATCH,
      state: dissolveProgress > 0.0 ? 'crack-dissolve' : 'intact',
    };
  }
}
