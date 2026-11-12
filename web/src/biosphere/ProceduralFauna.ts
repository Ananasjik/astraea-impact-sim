import * as THREE from 'three';
import { BiomeSystem } from './BiomeSystem';

export type FaunaGuild =
  | 'HerbivoreHerd'
  | 'PursuitPredator'
  | 'AvianFlock'
  | 'PelagicMarine';

export type FaunaState =
  | 'Grazing'
  | 'Alert'
  | 'StampedeOrMassFlight'
  | 'ThermalEscapeOrBurrow'
  | 'Lethal';

export interface FaunaStimulus {
  tImpactElapsedSec: number;
  impactorDistRadii: number;
  pgaG: number;
  overpressureKpa: number;
  localTempK: number;
  radFluxKwM2: number;
  ejectaFluxKgM2: number;
  tsunamiHeightM: number;
  crustDamage: number;
  impactAzimuthLocal: THREE.Vector3;
}

export const ALL_FAUNA_GUILDS: readonly FaunaGuild[] = [
  'HerbivoreHerd',
  'PursuitPredator',
  'AvianFlock',
  'PelagicMarine',
];

const MAX_FAUNA_AGENTS_CAP = 512;

const STATE_TO_CODE: Record<FaunaState, number> = {
  Grazing: 0,
  Alert: 1,
  StampedeOrMassFlight: 2,
  ThermalEscapeOrBurrow: 3,
  Lethal: 4,
};

export function transitionFaunaState(
  current: FaunaState,
  guild: FaunaGuild,
  stim: FaunaStimulus,
  isLand: boolean
): FaunaState {
  void guild;

  if (current === 'Lethal') {
    return 'Lethal';
  }

  if (
    stim.localTempK > 420.0 ||
    stim.overpressureKpa > 80.0 ||
    stim.pgaG > 2.5 ||
    stim.ejectaFluxKgM2 > 50.0 ||
    (isLand && stim.tsunamiHeightM > 20.0) ||
    stim.crustDamage > 0.7
  ) {
    return 'Lethal';
  }

  if (stim.radFluxKwM2 > 12.0 || stim.localTempK > 340.0) {
    return 'ThermalEscapeOrBurrow';
  }

  if (
    stim.pgaG > 0.05 ||
    stim.overpressureKpa > 2.0 ||
    stim.radFluxKwM2 > 5.0 ||
    current === 'StampedeOrMassFlight'
  ) {
    return 'StampedeOrMassFlight';
  }

  if (
    stim.impactorDistRadii < 4.0 ||
    stim.pgaG > 0.01 ||
    current === 'Alert'
  ) {
    return 'Alert';
  }

  return 'Grazing';
}

function clamp(v: number, minVal: number, maxVal: number): number {
  return v < minVal ? minVal : v > maxVal ? maxVal : v;
}

function createMulberry32(seed: number): () => number {
  let state = (Math.trunc(seed) >>> 0) || 1;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };
}

interface FaunaGeomAcc {
  positions: number[];
  normals: number[];
  partIds: number[];
  limbWeights: number[];
  indices: number[];
}

function createFaunaAcc(): FaunaGeomAcc {
  return {
    positions: [],
    normals: [],
    partIds: [],
    limbWeights: [],
    indices: [],
  };
}

function pushFaunaVertex(
  acc: FaunaGeomAcc,
  px: number,
  py: number,
  pz: number,
  nx: number,
  ny: number,
  nz: number,
  partId: number,
  limbWeight: number
): number {
  const lenSq = nx * nx + ny * ny + nz * nz;
  let unx = 0.0;
  let uny = 1.0;
  let unz = 0.0;
  if (lenSq > 1.0e-12 && Number.isFinite(lenSq)) {
    const inv = 1.0 / Math.sqrt(lenSq);
    unx = nx * inv;
    uny = ny * inv;
    unz = nz * inv;
  }
  const idx = acc.positions.length / 3;
  acc.positions.push(px, py, pz);
  acc.normals.push(unx, uny, unz);
  acc.partIds.push(partId);
  acc.limbWeights.push(clamp(limbWeight, 0.0, 1.0));
  return idx;
}

function appendEllipsoid(
  acc: FaunaGeomAcc,
  cx: number,
  cy: number,
  cz: number,
  rx: number,
  ry: number,
  rz: number,
  partId: number,
  weightBase: number,
  weightTip: number,
  latSegs = 5,
  lonSegs = 8
): void {
  const rings: number[][] = [];
  for (let iy = 0; iy <= latSegs; iy++) {
    const phi = (iy / latSegs) * Math.PI;
    const sinP = Math.sin(phi);
    const cosP = Math.cos(phi);
    const ring: number[] = [];
    for (let ix = 0; ix <= lonSegs; ix++) {
      const theta = (ix / lonSegs) * Math.PI * 2.0;
      const dx = sinP * Math.cos(theta);
      const dy = cosP;
      const dz = sinP * Math.sin(theta);
      const w = weightBase + (weightTip - weightBase) * (0.5 + 0.5 * dz);
      ring.push(
        pushFaunaVertex(
          acc,
          cx + dx * rx,
          cy + dy * ry,
          cz + dz * rz,
          dx / Math.max(rx, 1.0e-4),
          dy / Math.max(ry, 1.0e-4),
          dz / Math.max(rz, 1.0e-4),
          partId,
          w
        )
      );
    }
    rings.push(ring);
  }
  for (let iy = 0; iy < latSegs; iy++) {
    for (let ix = 0; ix < lonSegs; ix++) {
      const a = rings[iy][ix];
      const b = rings[iy + 1][ix];
      const c = rings[iy][ix + 1];
      const d = rings[iy + 1][ix + 1];
      acc.indices.push(a, b, d, a, d, c);
    }
  }
}

function appendLimbTube(
  acc: FaunaGeomAcc,
  x0: number,
  y0: number,
  z0: number,
  x1: number,
  y1: number,
  z1: number,
  r0: number,
  r1: number,
  partId: number,
  w0: number,
  w1: number,
  segs = 6
): void {
  const dir = new THREE.Vector3(x1 - x0, y1 - y0, z1 - z0);
  if (dir.lengthSq() < 1.0e-8) {
    return;
  }
  dir.normalize();
  const ref =
    Math.abs(dir.y) < 0.9 ? new THREE.Vector3(0.0, 1.0, 0.0) : new THREE.Vector3(1.0, 0.0, 0.0);
  const u = new THREE.Vector3().crossVectors(ref, dir).normalize();
  const v = new THREE.Vector3().crossVectors(dir, u).normalize();

  const ring0: number[] = [];
  const ring1: number[] = [];
  for (let i = 0; i <= segs; i++) {
    const theta = (i / segs) * Math.PI * 2.0;
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const nx = u.x * c + v.x * s;
    const ny = u.y * c + v.y * s;
    const nz = u.z * c + v.z * s;
    ring0.push(pushFaunaVertex(acc, x0 + nx * r0, y0 + ny * r0, z0 + nz * r0, nx, ny, nz, partId, w0));
    ring1.push(pushFaunaVertex(acc, x1 + nx * r1, y1 + ny * r1, z1 + nz * r1, nx, ny, nz, partId, w1));
  }
  for (let i = 0; i < segs; i++) {
    const a = ring0[i];
    const b = ring0[i + 1];
    const c = ring1[i];
    const d = ring1[i + 1];
    acc.indices.push(a, b, d, a, d, c);
  }
}

function appendWingPanel(
  acc: FaunaGeomAcc,
  sideSign: number,
  span: number,
  chordRoot: number,
  chordTip: number,
  sweepZ: number,
  partId: number
): void {
  const segs = 4;
  const front: number[] = [];
  const back: number[] = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const x = sideSign * (0.12 + t * span);
    const zLead = 0.14 - t * sweepZ;
    const chord = chordRoot + (chordTip - chordRoot) * t;
    const yCamber = Math.sin(t * Math.PI) * 0.04;
    front.push(pushFaunaVertex(acc, x, yCamber, zLead, 0.0, 1.0, 0.15, partId, t));
    back.push(pushFaunaVertex(acc, x, yCamber - 0.02, zLead - chord, 0.0, 1.0, -0.15, partId, t));
  }
  for (let i = 0; i < segs; i++) {
    if (sideSign > 0) {
      acc.indices.push(front[i], back[i], back[i + 1], front[i], back[i + 1], front[i + 1]);
    } else {
      acc.indices.push(front[i], back[i + 1], back[i], front[i], front[i + 1], back[i + 1]);
    }
  }
}

function buildGuildGeometry(guild: FaunaGuild): THREE.BufferGeometry {
  const acc = createFaunaAcc();

  switch (guild) {
    case 'HerbivoreHerd': {
      // Articulated grazing ungulate (~2.2 m long, ~1.5 m tall)
      appendEllipsoid(acc, 0.0, 0.95, 0.0, 0.36, 0.38, 0.78, 0, 0.1, 0.2);
      // Neck + Head (part 5)
      appendLimbTube(acc, 0.0, 1.05, 0.55, 0.0, 1.38, 0.92, 0.18, 0.14, 5, 0.2, 0.7);
      appendEllipsoid(acc, 0.0, 1.4, 1.04, 0.14, 0.15, 0.25, 5, 0.6, 1.0, 4, 6);
      // 4 legs (FL=1, FR=2, BL=3, BR=4)
      appendLimbTube(acc, -0.22, 0.85, 0.45, -0.24, 0.0, 0.48, 0.09, 0.055, 1, 0.0, 1.0);
      appendLimbTube(acc, 0.22, 0.85, 0.45, 0.24, 0.0, 0.48, 0.09, 0.055, 2, 0.0, 1.0);
      appendLimbTube(acc, -0.22, 0.85, -0.45, -0.24, 0.0, -0.48, 0.1, 0.055, 3, 0.0, 1.0);
      appendLimbTube(acc, 0.22, 0.85, -0.45, 0.24, 0.0, -0.48, 0.1, 0.055, 4, 0.0, 1.0);
      // Short tail (part 6)
      appendLimbTube(acc, 0.0, 1.05, -0.72, 0.0, 0.82, -0.96, 0.05, 0.03, 6, 0.2, 0.8);
      break;
    }

    case 'PursuitPredator': {
      // Sleek muscular carnivore (~2.0 m long, ~1.05 m tall)
      appendEllipsoid(acc, 0.0, 0.72, 0.0, 0.26, 0.26, 0.74, 0, 0.1, 0.2);
      // Neck + tapered snout (part 5)
      appendLimbTube(acc, 0.0, 0.78, 0.55, 0.0, 0.94, 0.86, 0.15, 0.12, 5, 0.2, 0.6);
      appendEllipsoid(acc, 0.0, 0.94, 0.98, 0.12, 0.12, 0.24, 5, 0.6, 1.0, 4, 6);
      // 4 agile limbs
      appendLimbTube(acc, -0.18, 0.65, 0.42, -0.2, 0.0, 0.46, 0.075, 0.045, 1, 0.0, 1.0);
      appendLimbTube(acc, 0.18, 0.65, 0.42, 0.2, 0.0, 0.46, 0.075, 0.045, 2, 0.0, 1.0);
      appendLimbTube(acc, -0.18, 0.65, -0.44, -0.2, 0.0, -0.48, 0.085, 0.045, 3, 0.0, 1.0);
      appendLimbTube(acc, 0.18, 0.65, -0.44, 0.2, 0.0, -0.48, 0.085, 0.045, 4, 0.0, 1.0);
      // Long counterbalancing tail (part 6)
      appendLimbTube(acc, 0.0, 0.74, -0.68, 0.0, 0.66, -1.35, 0.06, 0.025, 6, 0.1, 1.0);
      break;
    }

    case 'AvianFlock': {
      // Streamlined bird (~1.0 m body, ~1.9 m wingspan)
      appendEllipsoid(acc, 0.0, 0.0, 0.0, 0.13, 0.12, 0.42, 0, 0.0, 0.2, 4, 7);
      // Head/beak (part 5)
      appendEllipsoid(acc, 0.0, 0.04, 0.44, 0.07, 0.07, 0.14, 5, 0.4, 0.8, 3, 5);
      appendLimbTube(acc, 0.0, 0.03, 0.54, 0.0, 0.01, 0.72, 0.03, 0.005, 5, 0.8, 1.0, 4);
      // Swept wings (left = 1, right = 2)
      appendWingPanel(acc, -1.0, 0.95, 0.28, 0.09, 0.22, 1);
      appendWingPanel(acc, 1.0, 0.95, 0.28, 0.09, 0.22, 2);
      // Tail fan (part 6)
      appendLimbTube(acc, 0.0, 0.0, -0.38, 0.0, 0.02, -0.72, 0.08, 0.14, 6, 0.2, 0.9, 4);
      break;
    }

    case 'PelagicMarine':
    default: {
      // Hydrofoil cetacean/marine silhouette (~3.2 m long)
      appendEllipsoid(acc, 0.0, 0.0, 0.0, 0.34, 0.32, 1.25, 0, 0.0, 0.25, 5, 8);
      // Dorsal fin (part 5)
      appendLimbTube(acc, 0.0, 0.24, 0.1, 0.0, 0.65, -0.18, 0.12, 0.03, 5, 0.1, 0.4, 4);
      // Pectoral hydrofoils (parts 1 & 2)
      appendWingPanel(acc, -1.0, 0.65, 0.32, 0.1, 0.26, 1);
      appendWingPanel(acc, 1.0, 0.65, 0.32, 0.1, 0.26, 2);
      // Tail peduncle & fluke (part 6)
      appendLimbTube(acc, 0.0, 0.0, -1.05, 0.0, 0.0, -1.65, 0.16, 0.06, 6, 0.2, 0.9, 6);
      appendWingPanel(acc, -1.0, 0.42, 0.24, 0.08, 0.15, 6);
      appendWingPanel(acc, 1.0, 0.42, 0.24, 0.08, 0.15, 6);
      break;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(acc.positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(acc.normals, 3));
  geometry.setAttribute('aPartId', new THREE.Float32BufferAttribute(acc.partIds, 1));
  geometry.setAttribute('aLimbWeight', new THREE.Float32BufferAttribute(acc.limbWeights, 1));
  geometry.setIndex(acc.indices);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

const FAUNA_VERTEX_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>

attribute float aPartId;
attribute float aLimbWeight;
attribute vec4 aAnimState; // [phaseOffset, strideFreqHz, stateCode, speedNorm]

uniform float uTimeSec;
uniform float uGuildMode; // 0 = quadruped, 1 = avian, 2 = marine

varying vec3 vWorldNormal;
varying vec3 vTint;
varying float vStateCode;

void main() {
  #ifdef USE_INSTANCING
    mat4 instMat = instanceMatrix;
  #else
    mat4 instMat = mat4(1.0);
  #endif

  #ifdef USE_INSTANCING_COLOR
    vTint = instanceColor;
  #else
    vTint = vec3(1.0);
  #endif

  float phase = aAnimState.x;
  float freq = aAnimState.y;
  float stateCode = aAnimState.z;
  float speedNorm = aAnimState.w;
  vStateCode = stateCode;

  vec3 pos = position;
  float animClock = uTimeSec * freq * PI2 + phase;

  if (stateCode > 3.5) {
    // Lethal: collapsed pose
    pos = vec3(pos.x, pos.z * 0.22 + 0.15, pos.y * 0.65);
  } else if (uGuildMode > 0.5 && uGuildMode < 1.5) {
    // AvianFlock: wingbeat flap on parts 1 (left) and 2 (right)
    if (aPartId > 0.5 && aPartId < 2.5) {
      float flap = sin(animClock) * (0.45 + 0.45 * speedNorm) * aLimbWeight;
      pos.y += flap;
    }
  } else if (uGuildMode >= 1.5) {
    // PelagicMarine: sinusoidal body/fluke swim undulation
    float swim = sin(animClock - pos.z * 2.1) * (0.14 + 0.22 * speedNorm);
    float tailMask = aPartId > 5.5 ? 1.0 : 0.25;
    pos.y += swim * tailMask * (0.35 + aLimbWeight);
  } else {
    // Quadruped gait: diagonal leg swing (FL=1, FR=2, BL=3, BR=4), head bob (5), tail (6)
    if (aPartId > 0.5 && aPartId < 4.5) {
      float legPhase = (aPartId < 1.5 || aPartId > 3.5) ? 0.0 : PI;
      float swing = sin(animClock + legPhase) * (0.28 + 0.35 * speedNorm) * aLimbWeight;
      pos.z += swing;
      pos.y += max(0.0, cos(animClock + legPhase)) * 0.08 * speedNorm * aLimbWeight;
    } else if (aPartId > 4.5 && aPartId < 5.5) {
      // Head raised in Alert (stateCode ~ 1), crouched in ThermalEscape (stateCode ~ 3)
      float alertLift = (stateCode > 0.5 && stateCode < 1.5) ? 0.22 : 0.0;
      float crouch = (stateCode > 2.5 && stateCode < 3.5) ? -0.28 : 0.0;
      pos.y += (alertLift + crouch) * aLimbWeight + sin(animClock * 0.5) * 0.05 * speedNorm;
    } else if (aPartId > 5.5) {
      pos.x += sin(animClock * 1.3) * 0.14 * aLimbWeight;
    }
  }

  vec4 localPos4 = vec4(pos, 1.0);
  #ifdef USE_INSTANCING
    localPos4 = instMat * localPos4;
  #endif

  vec3 worldNorm = normal;
  #ifdef USE_INSTANCING
    worldNorm = mat3(instMat) * worldNorm;
  #endif
  vWorldNormal = normalize(worldNorm);

  vec4 mvPosition = modelViewMatrix * localPos4;
  gl_Position = projectionMatrix * mvPosition;

  #include <logdepthbuf_vertex>
}
`;

const FAUNA_FRAGMENT_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>

uniform float uLocalTempK;
uniform vec3 uSunDir;

varying vec3 vWorldNormal;
varying vec3 vTint;
varying float vStateCode;

void main() {
  #include <logdepthbuf_fragment>

  vec3 n = normalize(vWorldNormal);
  vec3 sun = normalize(uSunDir);
  float light = 0.25 + 0.75 * smoothstep(-0.2, 0.95, dot(n, sun));

  vec3 col = vTint * light;

  // Panic / thermal stress tint & lethal darkening
  if (vStateCode > 3.5) {
    col *= 0.32;
  } else if (uLocalTempK > 340.0) {
    float heat = clamp((uLocalTempK - 340.0) / 140.0, 0.0, 1.0);
    col = mix(col, vec3(0.95, 0.35, 0.12) * light, heat * 0.45);
  }

  gl_FragColor = vec4(col, 1.0);
}
`;

interface FaunaAgent {
  guild: FaunaGuild;
  state: FaunaState;
  restNormal: THREE.Vector3;
  headingTangent: THREE.Vector3;
  altitudeOffsetM: number;
  baseScale: number;
  phaseOffset: number;
  color: THREE.Color;
}

interface GuildPool {
  guild: FaunaGuild;
  mesh: THREE.InstancedMesh;
  geometry: THREE.BufferGeometry;
  material: THREE.ShaderMaterial;
  animAttr: THREE.InstancedBufferAttribute;
  capacity: number;
  activeCount: number;
}

function getGuildBaseColor(guild: FaunaGuild, rng: () => number): THREE.Color {
  const s = 0.85 + 0.28 * rng();
  switch (guild) {
    case 'HerbivoreHerd':
      return new THREE.Color(0.58 * s, 0.44 * s, 0.28 * s);
    case 'PursuitPredator':
      return new THREE.Color(0.68 * s, 0.36 * s, 0.22 * s);
    case 'AvianFlock':
      return new THREE.Color(0.84 * s, 0.86 * s, 0.92 * s);
    case 'PelagicMarine':
      return new THREE.Color(0.22 * s, 0.48 * s, 0.66 * s);
  }
}

function getGuildSpeedMps(guild: FaunaGuild, state: FaunaState): number {
  if (state === 'Lethal') {
    return 0.0;
  }
  switch (guild) {
    case 'HerbivoreHerd':
      switch (state) {
        case 'Grazing':
          return 0.9;
        case 'Alert':
          return 0.4;
        case 'StampedeOrMassFlight':
          return 12.0;
        case 'ThermalEscapeOrBurrow':
          return 5.5;
      }
      break;
    case 'PursuitPredator':
      switch (state) {
        case 'Grazing':
          return 1.6;
        case 'Alert':
          return 0.7;
        case 'StampedeOrMassFlight':
          return 16.0;
        case 'ThermalEscapeOrBurrow':
          return 7.0;
      }
      break;
    case 'AvianFlock':
      switch (state) {
        case 'Grazing':
          return 6.5;
        case 'Alert':
          return 8.5;
        case 'StampedeOrMassFlight':
          return 22.0;
        case 'ThermalEscapeOrBurrow':
          return 25.0;
      }
      break;
    case 'PelagicMarine':
      switch (state) {
        case 'Grazing':
          return 2.8;
        case 'Alert':
          return 3.5;
        case 'StampedeOrMassFlight':
          return 13.0;
        case 'ThermalEscapeOrBurrow':
          return 15.0;
      }
      break;
  }
  return 0.0;
}

function buildTangentBasis(normal: THREE.Vector3): [THREE.Vector3, THREE.Vector3] {
  const ref =
    Math.abs(normal.y) < 0.92
      ? new THREE.Vector3(0.0, 1.0, 0.0)
      : new THREE.Vector3(1.0, 0.0, 0.0);
  const u = new THREE.Vector3().crossVectors(ref, normal).normalize();
  const v = new THREE.Vector3().crossVectors(normal, u).normalize();
  return [u, v];
}

export class ProceduralFauna {
  readonly group: THREE.Group;
  private readonly biomeSystem: BiomeSystem;
  private readonly maxAgents: number;
  private readonly agents: FaunaAgent[] = [];
  private readonly pools = new Map<FaunaGuild, GuildPool>();

  // Scratch vectors to avoid per-frame allocations
  private readonly tmpRotated = new THREE.Vector3();
  private readonly tmpFwd = new THREE.Vector3();
  private readonly tmpUp = new THREE.Vector3();
  private readonly tmpRight = new THREE.Vector3();
  private readonly tmpFlee = new THREE.Vector3();
  private readonly tmpSep = new THREE.Vector3();
  private readonly tmpSteer = new THREE.Vector3();
  private readonly tmpPos = new THREE.Vector3();
  private readonly tmpScale = new THREE.Vector3();
  private readonly tmpQuat = new THREE.Quaternion();
  private readonly tmpBasisMat = new THREE.Matrix4();
  private readonly tmpMat = new THREE.Matrix4();
  private readonly tmpColor = new THREE.Color(1.0, 1.0, 1.0);

  constructor(biomeSystem: BiomeSystem, maxAgents = MAX_FAUNA_AGENTS_CAP) {
    this.biomeSystem = biomeSystem;
    const validMax = Number.isFinite(maxAgents)
      ? Math.trunc(maxAgents)
      : MAX_FAUNA_AGENTS_CAP;
    this.maxAgents = clamp(validMax, ALL_FAUNA_GUILDS.length, MAX_FAUNA_AGENTS_CAP);

    this.group = new THREE.Group();
    this.group.name = 'ProceduralFaunaGroup';

    const baseCap = Math.floor(this.maxAgents / ALL_FAUNA_GUILDS.length);
    const remainder = this.maxAgents % ALL_FAUNA_GUILDS.length;

    for (let i = 0; i < ALL_FAUNA_GUILDS.length; i++) {
      const guild = ALL_FAUNA_GUILDS[i];
      const cap = baseCap + (i < remainder ? 1 : 0);
      const geometry = buildGuildGeometry(guild);

      const animData = new Float32Array(cap * 4);
      const animAttr = new THREE.InstancedBufferAttribute(animData, 4);
      animAttr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('aAnimState', animAttr);

      const guildMode =
        guild === 'AvianFlock' ? 1.0 : guild === 'PelagicMarine' ? 2.0 : 0.0;

      const material = new THREE.ShaderMaterial({
        vertexShader: FAUNA_VERTEX_SHADER,
        fragmentShader: FAUNA_FRAGMENT_SHADER,
        side: THREE.DoubleSide,
        uniforms: {
          uTimeSec: { value: 0.0 },
          uGuildMode: { value: guildMode },
          uLocalTempK: { value: 288.0 },
          uSunDir: { value: new THREE.Vector3(0.6, 0.5, 0.624).normalize() },
        },
      });

      const mesh = new THREE.InstancedMesh(geometry, material, cap);
      mesh.name = `Fauna_${guild}`;
      mesh.frustumCulled = false;
      mesh.count = 0;

      for (let k = 0; k < cap; k++) {
        mesh.setColorAt(k, this.tmpColor);
        mesh.setMatrixAt(k, this.tmpMat);
      }
      if (mesh.instanceColor !== null) {
        mesh.instanceColor.needsUpdate = true;
      }

      this.group.add(mesh);
      this.pools.set(guild, {
        guild,
        mesh,
        geometry,
        material,
        animAttr,
        capacity: cap,
        activeCount: 0,
      });
    }
  }

  spawnChunkAgents(
    chunkCenterUnit: THREE.Vector3,
    planetRadiusKm: number,
    seed: number
  ): void {
    const safeRadiusKm = Math.max(planetRadiusKm, 1.0);
    const cx = Number.isFinite(chunkCenterUnit.x) ? chunkCenterUnit.x : 0.0;
    const cy = Number.isFinite(chunkCenterUnit.y) ? chunkCenterUnit.y : 1.0;
    const cz = Number.isFinite(chunkCenterUnit.z) ? chunkCenterUnit.z : 0.0;
    const cLenSq = cx * cx + cy * cy + cz * cz;
    const centerUnit =
      cLenSq > 1.0e-12
        ? new THREE.Vector3(cx, cy, cz).multiplyScalar(1.0 / Math.sqrt(cLenSq))
        : new THREE.Vector3(0.0, 1.0, 0.0);

    this.agents.length = 0;
    for (const pool of this.pools.values()) {
      pool.activeCount = 0;
      pool.mesh.count = 0;
    }

    const rng = createMulberry32(seed);
    const [tanU, tanV] = buildTangentBasis(centerUnit);
    const spawnRadiusKm = 0.45;
    const goldenAngle = Math.PI * (3.0 - Math.sqrt(5.0));

    const guildCounts: Record<FaunaGuild, number> = {
      HerbivoreHerd: 0,
      PursuitPredator: 0,
      AvianFlock: 0,
      PelagicMarine: 0,
    };

    const candidateCount = this.maxAgents * 2;
    for (let i = 0; i < candidateCount && this.agents.length < this.maxAgents; i++) {
      const radialKm = 0.012 + spawnRadiusKm * Math.sqrt((i + 0.5) / candidateCount);
      const theta = i * goldenAngle + rng() * 0.4;
      const offU = (Math.cos(theta) * radialKm) / safeRadiusKm;
      const offV = (Math.sin(theta) * radialKm) / safeRadiusKm;

      const restNormal = new THREE.Vector3()
        .copy(centerUnit)
        .addScaledVector(tanU, offU)
        .addScaledVector(tanV, offV)
        .normalize();

      const sample = this.biomeSystem.sampleSurface(restNormal);
      const isLand = sample.elevationM >= 0.0;

      let guild: FaunaGuild;
      let altitudeOffsetM = 0.0;
      if (!isLand) {
        if (rng() < 0.72) {
          guild = 'PelagicMarine';
          altitudeOffsetM = -4.0 - 12.0 * rng();
        } else {
          guild = 'AvianFlock';
          altitudeOffsetM = 18.0 + 32.0 * rng();
        }
      } else {
        const roll = rng();
        if (roll < 0.52) {
          guild = 'HerbivoreHerd';
        } else if (roll < 0.72) {
          guild = 'PursuitPredator';
        } else {
          guild = 'AvianFlock';
          altitudeOffsetM = 14.0 + 28.0 * rng();
        }
      }

      const pool = this.pools.get(guild);
      if (pool === undefined || guildCounts[guild] >= pool.capacity) {
        continue;
      }

      const [tu, tv] = buildTangentBasis(restNormal);
      const hAng = rng() * Math.PI * 2.0;
      const headingTangent = new THREE.Vector3()
        .addScaledVector(tu, Math.cos(hAng))
        .addScaledVector(tv, Math.sin(hAng))
        .normalize();

      guildCounts[guild] += 1;
      this.agents.push({
        guild,
        state: 'Grazing',
        restNormal,
        headingTangent,
        altitudeOffsetM,
        baseScale: 0.8 + 0.4 * rng(),
        phaseOffset: rng() * Math.PI * 2.0,
        color: getGuildBaseColor(guild, rng),
      });
    }
  }

  step(
    dtSec: number,
    timeSec: number,
    stim: FaunaStimulus,
    cameraPosKm: [number, number, number],
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number
  ): void {
    const dt = Number.isFinite(dtSec) ? clamp(dtSec, 0.0, 0.25) : 0.0;
    const tSec = Number.isFinite(timeSec) ? timeSec : 0.0;
    const safeRadiusKm = Math.max(livingRadiusKm, 1.0);
    const cosSpin = Math.cos(spinAngleRad);
    const sinSpin = Math.sin(spinAngleRad);

    for (const pool of this.pools.values()) {
      pool.activeCount = 0;
      pool.material.uniforms.uTimeSec.value = tSec;
      pool.material.uniforms.uLocalTempK.value = Number.isFinite(stim.localTempK)
        ? stim.localTempK
        : 288.0;
    }

    const azValid =
      Number.isFinite(stim.impactAzimuthLocal.x) &&
      Number.isFinite(stim.impactAzimuthLocal.y) &&
      Number.isFinite(stim.impactAzimuthLocal.z) &&
      stim.impactAzimuthLocal.lengthSq() > 1.0e-12;

    const nAgents = this.agents.length;
    for (let i = 0; i < nAgents; i++) {
      const agent = this.agents[i];
      const elevM = this.biomeSystem.getElevationM(agent.restNormal);
      const isLand = agent.guild !== 'PelagicMarine' && elevM >= 0.0;

      agent.state = transitionFaunaState(agent.state, agent.guild, stim, isLand);

      const speedMps = getGuildSpeedMps(agent.guild, agent.state);
      if (dt > 0.0 && speedMps > 0.0) {
        // Compute local boids separation + impact flee steering on the tangent sphere plane
        this.tmpSep.set(0.0, 0.0, 0.0);
        const sepThreshRad = 0.006 / safeRadiusKm;
        const sepThreshSq = sepThreshRad * sepThreshRad;
        const neighborWindow = Math.min(nAgents, 24);
        for (let k = 1; k <= neighborWindow; k++) {
          const other = this.agents[(i + k) % nAgents];
          if (other.guild !== agent.guild) {
            continue;
          }
          const dx = agent.restNormal.x - other.restNormal.x;
          const dy = agent.restNormal.y - other.restNormal.y;
          const dz = agent.restNormal.z - other.restNormal.z;
          const dSq = dx * dx + dy * dy + dz * dz;
          if (dSq > 1.0e-18 && dSq < sepThreshSq) {
            const invD = 1.0 / Math.sqrt(dSq);
            this.tmpSep.x += dx * invD;
            this.tmpSep.y += dy * invD;
            this.tmpSep.z += dz * invD;
          }
        }

        // Remove radial component from separation
        const sepDot = this.tmpSep.dot(agent.restNormal);
        this.tmpSep.addScaledVector(agent.restNormal, -sepDot);

        this.tmpSteer.copy(agent.headingTangent);

        if (
          agent.state === 'StampedeOrMassFlight' ||
          agent.state === 'ThermalEscapeOrBurrow'
        ) {
          if (azValid) {
            // Rotate world/local impact azimuth back to rest frame if spinning, then project tangent flee away from impact
            const ax = stim.impactAzimuthLocal.x;
            const ay = stim.impactAzimuthLocal.y;
            const az = stim.impactAzimuthLocal.z;
            const rax = ax * cosSpin - az * sinSpin;
            const ray = ay;
            const raz = ax * sinSpin + az * cosSpin;
            this.tmpFlee.set(-rax, -ray, -raz);
          } else {
            this.tmpFlee.copy(agent.headingTangent);
          }
          const fDot = this.tmpFlee.dot(agent.restNormal);
          this.tmpFlee.addScaledVector(agent.restNormal, -fDot);
          if (this.tmpFlee.lengthSq() > 1.0e-12) {
            this.tmpFlee.normalize();
            this.tmpSteer.lerp(this.tmpFlee, clamp(dt * 4.5, 0.0, 1.0));
          }
        } else {
          // Gentle meandering wander while grazing/alert
          const wander = Math.sin(tSec * 0.7 + agent.phaseOffset) * 0.45 * dt;
          this.tmpRight.crossVectors(agent.restNormal, agent.headingTangent).normalize();
          this.tmpSteer.addScaledVector(this.tmpRight, wander);
        }

        if (this.tmpSep.lengthSq() > 1.0e-10) {
          this.tmpSep.normalize();
          this.tmpSteer.addScaledVector(this.tmpSep, clamp(dt * 2.5, 0.0, 0.8));
        }

        // Keep heading orthogonal to restNormal
        const hDot = this.tmpSteer.dot(agent.restNormal);
        this.tmpSteer.addScaledVector(agent.restNormal, -hDot);
        if (this.tmpSteer.lengthSq() > 1.0e-12) {
          agent.headingTangent.copy(this.tmpSteer).normalize();
        }

        const angularStep = (speedMps * 0.001 * dt) / safeRadiusKm;
        const candidateNormal = this.tmpRotated
          .copy(agent.restNormal)
          .addScaledVector(agent.headingTangent, angularStep)
          .normalize();

        const nextElevM = this.biomeSystem.getElevationM(candidateNormal);
        if (agent.guild === 'PelagicMarine' && nextElevM >= -1.0) {
          // Turn gently back toward deeper water
          agent.headingTangent.negate();
        } else if (
          (agent.guild === 'HerbivoreHerd' || agent.guild === 'PursuitPredator') &&
          nextElevM < 0.0
        ) {
          agent.headingTangent.negate();
        } else {
          agent.restNormal.copy(candidateNormal);
          const ortho = agent.headingTangent.dot(agent.restNormal);
          agent.headingTangent
            .addScaledVector(agent.restNormal, -ortho)
            .normalize();
        }
      }

      // Compute rotated world-frame position and orientation
      const pool = this.pools.get(agent.guild);
      if (pool === undefined || pool.activeCount >= pool.capacity) {
        continue;
      }

      const rnx = agent.restNormal.x;
      const rny = agent.restNormal.y;
      const rnz = agent.restNormal.z;
      this.tmpUp.set(
        rnx * cosSpin + rnz * sinSpin,
        rny,
        -rnx * sinSpin + rnz * cosSpin
      );

      const htx = agent.headingTangent.x;
      const hty = agent.headingTangent.y;
      const htz = agent.headingTangent.z;
      this.tmpFwd.set(
        htx * cosSpin + htz * sinSpin,
        hty,
        -htx * sinSpin + htz * cosSpin
      );
      this.tmpRight.crossVectors(this.tmpUp, this.tmpFwd).normalize();
      this.tmpFwd.crossVectors(this.tmpRight, this.tmpUp).normalize();

      const groundElevM =
        agent.guild === 'PelagicMarine'
          ? 0.0
          : Math.max(0.0, this.biomeSystem.getElevationM(agent.restNormal));
      const thermalDiveM =
        agent.state === 'ThermalEscapeOrBurrow' &&
        (agent.guild === 'HerbivoreHerd' ||
          agent.guild === 'PursuitPredator' ||
          agent.guild === 'PelagicMarine')
          ? -0.6
          : 0.0;
      const radialKm =
        safeRadiusKm +
        (groundElevM + agent.altitudeOffsetM + thermalDiveM) * 0.001;

      const wxKm = livingComKm[0] + this.tmpUp.x * radialKm;
      const wyKm = livingComKm[1] + this.tmpUp.y * radialKm;
      const wzKm = livingComKm[2] + this.tmpUp.z * radialKm;

      this.tmpPos.set(
        wxKm - cameraPosKm[0],
        wyKm - cameraPosKm[1],
        wzKm - cameraPosKm[2]
      );

      this.tmpBasisMat.makeBasis(this.tmpRight, this.tmpUp, this.tmpFwd);
      this.tmpQuat.setFromRotationMatrix(this.tmpBasisMat);
      const sKm = 0.001 * agent.baseScale;
      this.tmpScale.set(sKm, sKm, sKm);
      this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);

      const slot = pool.activeCount;
      pool.mesh.setMatrixAt(slot, this.tmpMat);
      pool.mesh.setColorAt(slot, agent.color);

      const strideFreqHz =
        agent.state === 'StampedeOrMassFlight'
          ? 3.8
          : agent.state === 'ThermalEscapeOrBurrow'
          ? 4.4
          : agent.state === 'Alert'
          ? 0.8
          : agent.state === 'Lethal'
          ? 0.0
          : 1.5;
      const speedNorm = clamp(speedMps / 22.0, 0.0, 1.0);
      const animArr = pool.animAttr.array as Float32Array;
      const off = slot * 4;
      animArr[off] = agent.phaseOffset;
      animArr[off + 1] = strideFreqHz;
      animArr[off + 2] = STATE_TO_CODE[agent.state];
      animArr[off + 3] = speedNorm;

      pool.activeCount += 1;
    }

    for (const pool of this.pools.values()) {
      pool.mesh.count = pool.activeCount;
      pool.mesh.instanceMatrix.needsUpdate = true;
      pool.animAttr.needsUpdate = true;
      if (pool.mesh.instanceColor !== null) {
        pool.mesh.instanceColor.needsUpdate = true;
      }
    }
  }

  getStateCounts(): Record<FaunaState, number> {
    const counts: Record<FaunaState, number> = {
      Grazing: 0,
      Alert: 0,
      StampedeOrMassFlight: 0,
      ThermalEscapeOrBurrow: 0,
      Lethal: 0,
    };
    for (let i = 0; i < this.agents.length; i++) {
      counts[this.agents[i].state] += 1;
    }
    return counts;
  }

  dispose(): void {
    this.group.clear();
    this.agents.length = 0;
    for (const pool of this.pools.values()) {
      pool.geometry.dispose();
      pool.material.dispose();
      pool.mesh.dispose();
    }
    this.pools.clear();
  }
}
