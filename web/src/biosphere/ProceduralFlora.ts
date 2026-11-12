import * as THREE from 'three';
import { BiomeSystem, type BiomeId } from './BiomeSystem';

export type FloraSpeciesId =
  | 'ButtressCanopyTree'
  | 'CoastalPalm'
  | 'AcaciaSavannaTree'
  | 'BroadleafOak'
  | 'BorealConifer'
  | 'GiantFern'
  | 'DesertSaguaroCactus'
  | 'CoralColony'
  | 'GiantKelp';

export const ALL_FLORA_SPECIES: readonly FloraSpeciesId[] = [
  'ButtressCanopyTree',
  'CoastalPalm',
  'AcaciaSavannaTree',
  'BroadleafOak',
  'BorealConifer',
  'GiantFern',
  'DesertSaguaroCactus',
  'CoralColony',
  'GiantKelp',
];

const MAX_FLORA_INSTANCES_CAP = 4096;

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

function hashSpeciesSeed(species: FloraSpeciesId, seed: number): number {
  let h = (Math.trunc(seed) >>> 0) ^ 0x9e3779b9;
  for (let i = 0; i < species.length; i++) {
    h = Math.imul(h ^ species.charCodeAt(i), 0x85ebca6b) >>> 0;
  }
  return h || 1;
}

interface GeometryAccumulator {
  positions: number[];
  normals: number[];
  branchWeights: number[];
  partTypes: number[];
  indices: number[];
}

function createAccumulator(): GeometryAccumulator {
  return {
    positions: [],
    normals: [],
    branchWeights: [],
    partTypes: [],
    indices: [],
  };
}

function pushVertex(
  acc: GeometryAccumulator,
  px: number,
  py: number,
  pz: number,
  nx: number,
  ny: number,
  nz: number,
  branchWeight: number,
  partType: number
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
  acc.positions.push(
    Number.isFinite(px) ? px : 0.0,
    Number.isFinite(py) ? py : 0.0,
    Number.isFinite(pz) ? pz : 0.0
  );
  acc.normals.push(unx, uny, unz);
  acc.branchWeights.push(clamp(Number.isFinite(branchWeight) ? branchWeight : 0.0, 0.0, 1.0));
  acc.partTypes.push(Number.isFinite(partType) ? partType : 0.0);
  return idx;
}

function buildOrthonormalBasis(dir: THREE.Vector3): [THREE.Vector3, THREE.Vector3, THREE.Vector3] {
  const d = dir.clone();
  if (d.lengthSq() < 1.0e-12) {
    d.set(0.0, 1.0, 0.0);
  } else {
    d.normalize();
  }
  const ref =
    Math.abs(d.y) < 0.92 ? new THREE.Vector3(0.0, 1.0, 0.0) : new THREE.Vector3(1.0, 0.0, 0.0);
  const u = new THREE.Vector3().crossVectors(ref, d).normalize();
  const v = new THREE.Vector3().crossVectors(d, u).normalize();
  return [u, v, d];
}

function appendTaperedLimb(
  acc: GeometryAccumulator,
  p0: THREE.Vector3,
  p1: THREE.Vector3,
  r0: number,
  r1: number,
  radialSegs: number,
  w0: number,
  w1: number,
  partType: number,
  ribAmp = 0.0,
  ribFreq = 8
): void {
  const dir = new THREE.Vector3().subVectors(p1, p0);
  if (dir.lengthSq() < 1.0e-8) {
    return;
  }
  const [u, v] = buildOrthonormalBasis(dir);
  const safeR0 = Math.max(r0, 0.015);
  const safeR1 = Math.max(r1, 0.01);
  const segs = Math.max(4, Math.trunc(radialSegs));

  const ring0: number[] = [];
  const ring1: number[] = [];

  for (let i = 0; i <= segs; i++) {
    const theta = (i / segs) * Math.PI * 2.0;
    const ribMod = 1.0 + ribAmp * Math.cos(theta * ribFreq);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    const nx = u.x * c + v.x * s;
    const ny = u.y * c + v.y * s;
    const nz = u.z * c + v.z * s;

    ring0.push(
      pushVertex(
        acc,
        p0.x + nx * safeR0 * ribMod,
        p0.y + ny * safeR0 * ribMod,
        p0.z + nz * safeR0 * ribMod,
        nx,
        ny,
        nz,
        w0,
        partType
      )
    );
    ring1.push(
      pushVertex(
        acc,
        p1.x + nx * safeR1 * ribMod,
        p1.y + ny * safeR1 * ribMod,
        p1.z + nz * safeR1 * ribMod,
        nx,
        ny,
        nz,
        w1,
        partType
      )
    );
  }

  for (let i = 0; i < segs; i++) {
    const a = ring0[i];
    const b = ring0[i + 1];
    const c = ring1[i];
    const d = ring1[i + 1];
    acc.indices.push(a, b, d, a, d, c);
  }
}

function appendCanopyBlob(
  acc: GeometryAccumulator,
  center: THREE.Vector3,
  rx: number,
  ry: number,
  rz: number,
  branchWeight: number,
  partType: number,
  rng: () => number,
  latSegs = 4,
  lonSegs = 6
): void {
  const safeRx = Math.max(rx, 0.05);
  const safeRy = Math.max(ry, 0.05);
  const safeRz = Math.max(rz, 0.05);

  const rings: number[][] = [];
  for (let iy = 0; iy <= latSegs; iy++) {
    const phi = (iy / latSegs) * Math.PI;
    const sinP = Math.sin(phi);
    const cosP = Math.cos(phi);
    const ring: number[] = [];
    for (let ix = 0; ix <= lonSegs; ix++) {
      const theta = (ix / lonSegs) * Math.PI * 2.0;
      const jitter = 0.88 + 0.24 * rng();
      const ox = sinP * Math.cos(theta) * safeRx * jitter;
      const oy = cosP * safeRy * jitter;
      const oz = sinP * Math.sin(theta) * safeRz * jitter;

      const weight = clamp(branchWeight + 0.12 * (0.5 + 0.5 * cosP), 0.0, 1.0);
      ring.push(
        pushVertex(
          acc,
          center.x + ox,
          center.y + oy,
          center.z + oz,
          ox / safeRx,
          oy / safeRy,
          oz / safeRz,
          weight,
          partType
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

function appendLeafBlade(
  acc: GeometryAccumulator,
  origin: THREE.Vector3,
  forward: THREE.Vector3,
  side: THREE.Vector3,
  up: THREE.Vector3,
  length: number,
  maxWidth: number,
  archDrop: number,
  segments: number,
  wBase: number,
  wTip: number,
  partType: number,
  waveAmp = 0.0
): void {
  const segs = Math.max(2, segments);
  const leftVerts: number[] = [];
  const midVerts: number[] = [];
  const rightVerts: number[] = [];

  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const widthProfile = Math.sin(Math.pow(t, 0.72) * Math.PI) * maxWidth;
    const archY = t * (1.0 - t) * archDrop * 1.4 - t * t * archDrop;
    const wave = waveAmp * Math.sin(t * Math.PI * 3.0);

    const cx = origin.x + forward.x * (t * length) + up.x * archY;
    const cy = origin.y + forward.y * (t * length) + up.y * archY;
    const cz = origin.z + forward.z * (t * length) + up.z * archY;

    const w = wBase + (wTip - wBase) * t;
    const ridgeLift = 0.06 * widthProfile;

    leftVerts.push(
      pushVertex(
        acc,
        cx - side.x * widthProfile + up.x * wave,
        cy - side.y * widthProfile + up.y * wave,
        cz - side.z * widthProfile + up.z * wave,
        up.x - side.x * 0.25,
        up.y - side.y * 0.25,
        up.z - side.z * 0.25,
        w,
        partType
      )
    );
    midVerts.push(
      pushVertex(
        acc,
        cx + up.x * (ridgeLift + wave),
        cy + up.y * (ridgeLift + wave),
        cz + up.z * (ridgeLift + wave),
        up.x,
        up.y,
        up.z,
        w,
        partType
      )
    );
    rightVerts.push(
      pushVertex(
        acc,
        cx + side.x * widthProfile + up.x * wave,
        cy + side.y * widthProfile + up.y * wave,
        cz + side.z * widthProfile + up.z * wave,
        up.x + side.x * 0.25,
        up.y + side.y * 0.25,
        up.z + side.z * 0.25,
        w,
        partType
      )
    );
  }

  for (let i = 0; i < segs; i++) {
    acc.indices.push(
      leftVerts[i],
      midVerts[i],
      midVerts[i + 1],
      leftVerts[i],
      midVerts[i + 1],
      leftVerts[i + 1],
      midVerts[i],
      rightVerts[i],
      rightVerts[i + 1],
      midVerts[i],
      rightVerts[i + 1],
      midVerts[i + 1]
    );
  }
}

function expandLSystem(
  axiom: string,
  rules: Record<string, string>,
  iterations: number
): string {
  let current = axiom;
  for (let i = 0; i < iterations; i++) {
    let next = '';
    for (let c = 0; c < current.length; c++) {
      const ch = current[c];
      next += rules[ch] ?? ch;
    }
    current = next;
  }
  return current;
}

interface TurtleState {
  pos: THREE.Vector3;
  dir: THREE.Vector3;
  radius: number;
  stepLen: number;
  depth: number;
}

function interpretBranchingLSystem(
  acc: GeometryAccumulator,
  program: string,
  startPos: THREE.Vector3,
  startDir: THREE.Vector3,
  startRadius: number,
  stepLen: number,
  yawRad: number,
  pitchRad: number,
  totalHeightEst: number,
  leafRx: number,
  leafRy: number,
  leafRz: number,
  rng: () => number,
  branchPartType = 0,
  leafPartType = 1,
  blossomPartType = 2,
  blossomChance = 0.18
): void {
  const stack: TurtleState[] = [];
  let state: TurtleState = {
    pos: startPos.clone(),
    dir: startDir.clone().normalize(),
    radius: startRadius,
    stepLen,
    depth: 0,
  };

  const qRot = new THREE.Quaternion();
  const axisVec = new THREE.Vector3();

  for (let i = 0; i < program.length; i++) {
    const cmd = program[i];
    switch (cmd) {
      case 'F': {
        const nextPos = state.pos
          .clone()
          .addScaledVector(state.dir, state.stepLen);
        const nextRadius = state.radius * 0.78;
        const w0 = clamp(state.pos.y / totalHeightEst, 0.0, 1.0);
        const w1 = clamp(nextPos.y / totalHeightEst, 0.0, 1.0);
        appendTaperedLimb(
          acc,
          state.pos,
          nextPos,
          state.radius,
          nextRadius,
          state.depth === 0 ? 7 : 5,
          w0,
          w1,
          branchPartType
        );
        state.pos.copy(nextPos);
        state.radius = nextRadius;
        break;
      }
      case '+': {
        const [u] = buildOrthonormalBasis(state.dir);
        qRot.setFromAxisAngle(u, pitchRad * (0.85 + 0.3 * rng()));
        state.dir.applyQuaternion(qRot).normalize();
        break;
      }
      case '-': {
        const [u] = buildOrthonormalBasis(state.dir);
        qRot.setFromAxisAngle(u, -pitchRad * (0.85 + 0.3 * rng()));
        state.dir.applyQuaternion(qRot).normalize();
        break;
      }
      case '&': {
        const [, v] = buildOrthonormalBasis(state.dir);
        qRot.setFromAxisAngle(v, pitchRad * (0.8 + 0.4 * rng()));
        state.dir.applyQuaternion(qRot).normalize();
        break;
      }
      case '/': {
        axisVec.copy(state.dir);
        qRot.setFromAxisAngle(axisVec, yawRad);
        state.dir.applyQuaternion(qRot).normalize();
        break;
      }
      case '[': {
        stack.push({
          pos: state.pos.clone(),
          dir: state.dir.clone(),
          radius: state.radius,
          stepLen: state.stepLen,
          depth: state.depth,
        });
        state.stepLen *= 0.76;
        state.radius *= 0.74;
        state.depth += 1;
        // Golden-angle twist per branch fork for 3D volumetric distribution
        const twist = (state.depth * 2.39996 + rng() * 0.6) % (Math.PI * 2.0);
        qRot.setFromAxisAngle(new THREE.Vector3(0.0, 1.0, 0.0), twist);
        state.dir.applyQuaternion(qRot).normalize();
        break;
      }
      case ']': {
        const popped = stack.pop();
        if (popped !== undefined) {
          state = popped;
        }
        break;
      }
      case 'L': {
        const w = clamp(state.pos.y / totalHeightEst, 0.25, 1.0);
        appendCanopyBlob(
          acc,
          state.pos,
          leafRx * (0.85 + 0.3 * rng()),
          leafRy * (0.85 + 0.3 * rng()),
          leafRz * (0.85 + 0.3 * rng()),
          w,
          leafPartType,
          rng
        );
        if (rng() < blossomChance) {
          const bPos = state.pos.clone().add(new THREE.Vector3(0.0, leafRy * 0.6, 0.0));
          appendCanopyBlob(
            acc,
            bPos,
            leafRx * 0.28,
            leafRy * 0.28,
            leafRz * 0.28,
            w,
            blossomPartType,
            rng,
            3,
            4
          );
        }
        break;
      }
      default:
        break;
    }
  }
}

export function buildFloraSpeciesGeometry(
  species: FloraSpeciesId,
  seed: number
): THREE.BufferGeometry {
  const rng = createMulberry32(hashSpeciesSeed(species, seed));
  const acc = createAccumulator();

  switch (species) {
    case 'ButtressCanopyTree': {
      // Flared tropical buttress root fins
      for (let i = 0; i < 5; i++) {
        const ang = (i / 5) * Math.PI * 2.0 + rng() * 0.4;
        const rBase = new THREE.Vector3(Math.cos(ang) * 1.45, 0.0, Math.sin(ang) * 1.45);
        const rTop = new THREE.Vector3(Math.cos(ang) * 0.22, 3.8 + rng() * 0.8, Math.sin(ang) * 0.22);
        appendTaperedLimb(acc, rBase, rTop, 0.32, 0.14, 5, 0.0, 0.16, 0);
      }
      // Tall columnar bole up to 14 m, then L-system emergent rainforest crown up to ~22 m
      const crownBase = new THREE.Vector3(0.0, 12.5, 0.0);
      appendTaperedLimb(acc, new THREE.Vector3(0.0, 0.0, 0.0), new THREE.Vector3(0.15, 6.2, -0.1), 0.72, 0.56, 8, 0.0, 0.28, 0);
      appendTaperedLimb(acc, new THREE.Vector3(0.15, 6.2, -0.1), crownBase, 0.56, 0.44, 8, 0.28, 0.56, 0);

      const prog = expandLSystem(
        'F[+FL][-FL][&FL]',
        { F: 'F[+FL][-FL]', L: 'L' },
        2
      );
      interpretBranchingLSystem(
        acc,
        prog,
        crownBase,
        new THREE.Vector3(0.0, 1.0, 0.0),
        0.42,
        3.2,
        2.39996,
        0.65,
        22.0,
        2.6,
        1.4,
        2.6,
        rng,
        0,
        1,
        2,
        0.25
      );
      break;
    }

    case 'CoastalPalm': {
      // Curved segmented ringed trunk reaching ~11.5 m
      const segs = 7;
      const leanX = (rng() - 0.5) * 1.6;
      const leanZ = (rng() - 0.5) * 1.6;
      let prev = new THREE.Vector3(0.0, 0.0, 0.0);
      let r = 0.34;
      for (let i = 1; i <= segs; i++) {
        const t = i / segs;
        const next = new THREE.Vector3(
          leanX * t * t,
          t * 11.2,
          leanZ * t * t
        );
        appendTaperedLimb(acc, prev, next, r, r * 0.91, 7, (i - 1) / segs, t, 0);
        prev = next;
        r *= 0.91;
      }
      // Coconut / blossom cluster under crown
      appendCanopyBlob(acc, prev.clone().add(new THREE.Vector3(0.0, -0.35, 0.0)), 0.45, 0.38, 0.45, 0.85, 2, rng, 3, 5);
      // 10 arching pinnate palm fronds
      const frondCount = 10;
      const up = new THREE.Vector3(0.0, 1.0, 0.0);
      for (let i = 0; i < frondCount; i++) {
        const ang = (i / frondCount) * Math.PI * 2.0 + rng() * 0.25;
        const elev = 0.25 + (i % 2) * 0.32;
        const fwd = new THREE.Vector3(Math.cos(ang), elev, Math.sin(ang)).normalize();
        const side = new THREE.Vector3(-Math.sin(ang), 0.0, Math.cos(ang));
        appendLeafBlade(acc, prev, fwd, side, up, 3.8 + rng() * 0.8, 0.72, 1.9, 6, 0.85, 1.0, 1);
      }
      break;
    }

    case 'AcaciaSavannaTree': {
      // Multi-forked thorny trunk + flat-topped umbrella savanna crown
      const fork = new THREE.Vector3(0.0, 4.2, 0.0);
      appendTaperedLimb(acc, new THREE.Vector3(0.0, 0.0, 0.0), fork, 0.48, 0.36, 7, 0.0, 0.4, 0);
      const prog = expandLSystem(
        '[+F[+FL][-FL]][-F[+FL][-FL]][&F[+FL]]',
        { F: 'F[+L][-L]' },
        1
      );
      interpretBranchingLSystem(
        acc,
        prog,
        fork,
        new THREE.Vector3(0.0, 1.0, 0.0),
        0.32,
        2.9,
        2.1,
        0.82,
        9.5,
        2.7,
        0.55,
        2.7,
        rng,
        0,
        1,
        2,
        0.3
      );
      break;
    }

    case 'BroadleafOak': {
      // Thick sturdy oak trunk + recursive spreading limbs
      const trunkTop = new THREE.Vector3(0.0, 5.0, 0.0);
      appendTaperedLimb(acc, new THREE.Vector3(0.0, 0.0, 0.0), trunkTop, 0.65, 0.45, 8, 0.0, 0.34, 0);
      const prog = expandLSystem(
        'F[+FL][-FL][&FL]',
        { F: 'F[+F][-FL]', L: 'L' },
        2
      );
      interpretBranchingLSystem(
        acc,
        prog,
        trunkTop,
        new THREE.Vector3(0.0, 1.0, 0.0),
        0.42,
        2.8,
        2.39996,
        0.62,
        14.5,
        2.1,
        1.7,
        2.1,
        rng,
        0,
        1,
        2,
        0.1
      );
      break;
    }

    case 'BorealConifer': {
      // Straight tapering spire trunk + whorled tiered needle boughs
      const height = 15.5 + rng() * 2.5;
      appendTaperedLimb(
        acc,
        new THREE.Vector3(0.0, 0.0, 0.0),
        new THREE.Vector3(0.0, height, 0.0),
        0.42,
        0.05,
        7,
        0.0,
        1.0,
        0
      );
      const tiers = 7;
      for (let t = 0; t < tiers; t++) {
        const frac = 0.2 + (t / (tiers - 1)) * 0.72;
        const y = height * frac;
        const tierRadius = (1.0 - frac * 0.86) * 3.4;
        const boughs = 5 + (t % 2);
        for (let b = 0; b < boughs; b++) {
          const ang = (b / boughs) * Math.PI * 2.0 + t * 0.7;
          const p0 = new THREE.Vector3(0.0, y, 0.0);
          const p1 = new THREE.Vector3(
            Math.cos(ang) * tierRadius,
            y - 0.55 * (1.0 - frac),
            Math.sin(ang) * tierRadius
          );
          appendTaperedLimb(acc, p0, p1, 0.1, 0.03, 4, frac, clamp(frac + 0.12, 0.0, 1.0), 0);
          appendCanopyBlob(
            acc,
            p1.clone().lerp(p0, 0.28),
            tierRadius * 0.48,
            0.55,
            tierRadius * 0.48,
            clamp(frac + 0.1, 0.2, 1.0),
            1,
            rng,
            3,
            5
          );
        }
      }
      // Conical spire tip
      appendCanopyBlob(acc, new THREE.Vector3(0.0, height + 0.35, 0.0), 0.55, 1.1, 0.55, 1.0, 1, rng, 3, 5);
      break;
    }

    case 'GiantFern': {
      // Short fibrous caudex trunk + 9 arching prehistoric fronds + fiddlehead buds
      const crown = new THREE.Vector3(0.0, 0.9, 0.0);
      appendTaperedLimb(acc, new THREE.Vector3(0.0, 0.0, 0.0), crown, 0.26, 0.2, 6, 0.0, 0.25, 0);
      appendCanopyBlob(acc, crown.clone().add(new THREE.Vector3(0.0, 0.25, 0.0)), 0.28, 0.32, 0.28, 0.35, 2, rng, 3, 4);
      const fronds = 9;
      const up = new THREE.Vector3(0.0, 1.0, 0.0);
      for (let i = 0; i < fronds; i++) {
        const ang = (i / fronds) * Math.PI * 2.0 + rng() * 0.25;
        const pitch = 0.55 + (i % 2) * 0.28;
        const fwd = new THREE.Vector3(Math.cos(ang), pitch, Math.sin(ang)).normalize();
        const side = new THREE.Vector3(-Math.sin(ang), 0.0, Math.cos(ang));
        appendLeafBlade(acc, crown, fwd, side, up, 2.6 + rng() * 0.6, 0.56, 1.35, 6, 0.35, 1.0, 1);
      }
      break;
    }

    case 'DesertSaguaroCactus': {
      // Ribbed columnar stem + 3 upward-curving articulated cactus arms + apical blossoms
      const h = 5.4 + rng() * 1.0;
      const pMid = new THREE.Vector3(0.0, h * 0.45, 0.0);
      const pTop = new THREE.Vector3(0.0, h, 0.0);
      appendTaperedLimb(acc, new THREE.Vector3(0.0, 0.0, 0.0), pMid, 0.38, 0.36, 10, 0.0, 0.45, 1, 0.14, 8);
      appendTaperedLimb(acc, pMid, pTop, 0.36, 0.28, 10, 0.45, 1.0, 1, 0.14, 8);
      appendCanopyBlob(acc, pTop, 0.24, 0.22, 0.24, 1.0, 2, rng, 3, 5);

      const armCount = 3;
      for (let a = 0; a < armCount; a++) {
        const ang = (a / armCount) * Math.PI * 2.0 + rng() * 0.6;
        const armBase = new THREE.Vector3(0.0, h * (0.35 + a * 0.13), 0.0);
        const elbow = new THREE.Vector3(
          Math.cos(ang) * 1.15,
          armBase.y + 0.12,
          Math.sin(ang) * 1.15
        );
        const armTip = new THREE.Vector3(
          elbow.x,
          elbow.y + 1.5 + rng() * 0.7,
          elbow.z
        );
        appendTaperedLimb(acc, armBase, elbow, 0.24, 0.22, 8, 0.45, 0.65, 1, 0.12, 6);
        appendTaperedLimb(acc, elbow, armTip, 0.22, 0.18, 8, 0.65, 0.95, 1, 0.12, 6);
        appendCanopyBlob(acc, armTip, 0.18, 0.16, 0.18, 0.98, 2, rng, 3, 4);
      }
      break;
    }

    case 'CoralColony': {
      // Calcified reef base + L-system staghorn bifurcations + bright reef polyps
      const base = new THREE.Vector3(0.0, 0.25, 0.0);
      appendTaperedLimb(acc, new THREE.Vector3(0.0, 0.0, 0.0), base, 0.34, 0.24, 7, 0.0, 0.15, 0);
      const prog = expandLSystem(
        '[+FL][-FL][&FL]',
        { F: 'F[+F][-FL]' },
        2
      );
      interpretBranchingLSystem(
        acc,
        prog,
        base,
        new THREE.Vector3(0.0, 1.0, 0.0),
        0.18,
        0.55,
        2.15,
        0.58,
        2.2,
        0.26,
        0.24,
        0.26,
        rng,
        0,
        2,
        2,
        0.45
      );
      break;
    }

    case 'GiantKelp':
    default: {
      // Holdfast + rising sinuous kelp stipes + pneumatocyst bladders + undulating blades
      const holdfast = new THREE.Vector3(0.0, 0.35, 0.0);
      appendCanopyBlob(acc, new THREE.Vector3(0.0, 0.2, 0.0), 0.55, 0.28, 0.55, 0.0, 0, rng, 3, 6);
      const stipes = 3;
      const totalH = 13.5 + rng() * 2.5;
      const upVec = new THREE.Vector3(0.0, 1.0, 0.0);
      for (let s = 0; s < stipes; s++) {
        const baseAng = (s / stipes) * Math.PI * 2.0 + rng() * 0.4;
        let prev = holdfast.clone();
        const nodes = 8;
        for (let k = 1; k <= nodes; k++) {
          const t = k / nodes;
          const swayR = 0.45 * Math.sin(t * Math.PI * 2.0 + s);
          const next = new THREE.Vector3(
            Math.cos(baseAng + t * 1.4) * (0.28 + swayR),
            t * totalH,
            Math.sin(baseAng + t * 1.4) * (0.28 + swayR)
          );
          appendTaperedLimb(acc, prev, next, 0.08, 0.06, 5, (k - 1) / nodes, t, 0);
          if (k >= 2) {
            const bladeAng = baseAng + k * 2.2;
            const fwd = new THREE.Vector3(Math.cos(bladeAng), 0.25, Math.sin(bladeAng)).normalize();
            const side = new THREE.Vector3(-Math.sin(bladeAng), 0.0, Math.cos(bladeAng));
            // Gas bladder
            appendCanopyBlob(acc, next, 0.11, 0.14, 0.11, t, 2, rng, 2, 4);
            // Undulating kelp ribbon
            appendLeafBlade(acc, next, fwd, side, upVec, 2.2 + rng() * 0.6, 0.42, -0.35, 5, t, clamp(t + 0.08, 0.0, 1.0), 1, 0.14);
          }
          prev = next;
        }
      }
      break;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(acc.positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(acc.normals, 3));
  geometry.setAttribute('aBranchWeight', new THREE.Float32BufferAttribute(acc.branchWeights, 1));
  geometry.setAttribute('aPartType', new THREE.Float32BufferAttribute(acc.partTypes, 1));
  geometry.setIndex(acc.indices);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

const FLORA_VERTEX_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>

attribute float aBranchWeight;
attribute float aPartType;
attribute vec3 aInstanceOrigin;

uniform float uTimeSec;
uniform float uPgaG;
uniform float uOverpressureKpa;
uniform vec3 uBlastDirLocal;

varying vec3 vWorldNormal;
varying vec3 vInstanceTint;
varying float vBranchWeight;
varying float vPartType;

void main() {
  #ifdef USE_INSTANCING
    mat4 instMat = instanceMatrix;
  #else
    mat4 instMat = mat4(1.0);
  #endif

  #ifdef USE_INSTANCING_COLOR
    vInstanceTint = instanceColor;
  #else
    vInstanceTint = vec3(1.0);
  #endif

  vBranchWeight = aBranchWeight;
  vPartType = aPartType;

  // Transform world-space blast direction into local meter frame via scale-normalized basis transpose
  mat3 inst3 = mat3(instMat);
  vec3 c0 = length(inst3[0]) > 1.0e-6 ? normalize(inst3[0]) : vec3(1.0, 0.0, 0.0);
  vec3 c1 = length(inst3[1]) > 1.0e-6 ? normalize(inst3[1]) : vec3(0.0, 1.0, 0.0);
  vec3 c2 = length(inst3[2]) > 1.0e-6 ? normalize(inst3[2]) : vec3(0.0, 0.0, 1.0);
  mat3 rot3 = mat3(c0, c1, c2);
  float blastLen = length(uBlastDirLocal);
  vec3 normBlast = blastLen > 1.0e-5 ? (uBlastDirLocal / blastLen) : vec3(1.0, 0.0, 0.0);
  vec3 localBlastRaw = transposeMat3(rot3) * normBlast;
  vec2 blastXZ = localBlastRaw.xz;
  float blastXZLen = length(blastXZ);
  vec3 localBlastDir = blastXZLen > 1.0e-5
    ? vec3(blastXZ.x / blastXZLen, 0.0, blastXZ.y / blastXZLen)
    : vec3(1.0, 0.0, 0.0);

  float instSeed = dot(fract(aInstanceOrigin * 1.73), vec3(12.9898, 78.233, 37.719));

  // Wind sway (height-weighted via aBranchWeight)
  float windPhase = uTimeSec * 1.85 + instSeed + position.y * 0.42;
  vec3 windOffset = vec3(sin(windPhase), 0.0, cos(windPhase * 0.83)) * (0.24 * aBranchWeight);

  // Seismic shaking from PGA
  float shakeAmp = clamp(uPgaG, 0.0, 6.0) * 0.30 * aBranchWeight;
  vec3 shakeOffset = vec3(
    sin(uTimeSec * 29.0 + instSeed * 3.1),
    0.15 * cos(uTimeSec * 37.0 + instSeed),
    cos(uTimeSec * 23.0 + instSeed * 2.7)
  ) * shakeAmp;

  // Aerodynamic blast bending and canopy flattening
  float bend01 = clamp(uOverpressureKpa / 65.0, 0.0, 1.0);
  float bendCurve = pow2(aBranchWeight);
  vec3 deformedPos = position + windOffset + shakeOffset;
  deformedPos += localBlastDir * (bend01 * bendCurve * 6.5);
  deformedPos.y *= mix(1.0, 0.28, bend01 * aBranchWeight);

  vec4 localPos4 = vec4(deformedPos, 1.0);
  #ifdef USE_INSTANCING
    localPos4 = instMat * localPos4;
  #endif

  vec3 worldNorm = normal;
  #ifdef USE_INSTANCING
    worldNorm = inst3 * worldNorm;
  #endif
  vWorldNormal = normalize(worldNorm);

  vec4 mvPosition = modelViewMatrix * localPos4;
  gl_Position = projectionMatrix * mvPosition;

  #include <logdepthbuf_vertex>
}
`;

const FLORA_FRAGMENT_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>

uniform float uLocalTempK;
uniform float uRadFluxKwM2;
uniform vec3 uSunDir;

varying vec3 vWorldNormal;
varying vec3 vInstanceTint;
varying float vBranchWeight;
varying float vPartType;

void main() {
  #include <logdepthbuf_fragment>

  vec3 n = normalize(vWorldNormal);
  vec3 sun = normalize(uSunDir);
  float wrapLight = 0.22 + 0.78 * smoothstep(-0.2, 0.95, dot(n, sun));

  vec3 baseCol;
  if (vPartType < 0.5) {
    // Bark / trunk / holdfast
    baseCol = vec3(0.34, 0.23, 0.14) * vInstanceTint;
  } else if (vPartType < 1.5) {
    // Foliage / fronds / kelp
    baseCol = vec3(0.18, 0.46, 0.16) * vInstanceTint;
  } else {
    // Reef polyp / canopy blossom
    baseCol = vec3(0.92, 0.45, 0.34) * vInstanceTint;
  }

  vec3 litColor = baseCol * wrapLight;

  // Thermal charring under intense flash or heat
  float charLevel = clamp(
    max((uLocalTempK - 360.0) / 220.0, (uRadFluxKwM2 - 14.0) / 36.0),
    0.0,
    1.0
  );
  vec3 charredCol = vec3(0.06, 0.05, 0.045) * (0.5 + 0.5 * wrapLight);
  litColor = mix(litColor, charredCol, charLevel * 0.92);

  // Incandescent ember glow along canopy edges
  float emberThermal = clamp(
    max((uLocalTempK - 490.0) / 350.0, (uRadFluxKwM2 - 26.0) / 40.0),
    0.0,
    1.0
  );
  float canopyEdge = smoothstep(0.30, 0.95, vBranchWeight) * (vPartType > 0.5 ? 1.0 : 0.35);
  vec3 emberColor = mix(vec3(1.0, 0.28, 0.04), vec3(1.0, 0.78, 0.26), emberThermal);
  litColor += emberColor * (emberThermal * canopyEdge * 1.45);

  gl_FragColor = vec4(litColor, 1.0);
}
`;

function getBiomeSpeciesList(biome: BiomeId, elevationM: number, climateTempK: number): FloraSpeciesId[] {
  switch (biome) {
    case 'TropicalRainforest':
      return ['ButtressCanopyTree', 'CoastalPalm', 'GiantFern'];
    case 'Savanna':
      return ['AcaciaSavannaTree', 'CoastalPalm'];
    case 'TemperateForest':
      return ['BroadleafOak', 'GiantFern'];
    case 'BorealTaiga':
      return ['BorealConifer'];
    case 'AridSteppe':
      return climateTempK >= 278.0 ? ['DesertSaguaroCactus'] : ['BorealConifer'];
    case 'AlpineTundra':
      return elevationM < 3850.0 && climateTempK >= 262.0 ? ['BorealConifer'] : [];
    case 'OceanReef':
      return elevationM >= -150.0 && elevationM <= -1.0 ? ['CoralColony', 'GiantKelp'] : [];
    default:
      return [];
  }
}

function getSpeciesTint(species: FloraSpeciesId, rng: () => number): THREE.Color {
  const shade = 0.85 + 0.3 * rng();
  switch (species) {
    case 'ButtressCanopyTree':
      return new THREE.Color(0.75 * shade, 1.05 * shade, 0.72 * shade);
    case 'CoastalPalm':
      return new THREE.Color(0.88 * shade, 1.06 * shade, 0.68 * shade);
    case 'AcaciaSavannaTree':
      return new THREE.Color(1.08 * shade, 0.96 * shade, 0.58 * shade);
    case 'BroadleafOak':
      return new THREE.Color(0.84 * shade, 1.0 * shade, 0.66 * shade);
    case 'BorealConifer':
      return new THREE.Color(0.62 * shade, 0.86 * shade, 0.74 * shade);
    case 'GiantFern':
      return new THREE.Color(0.76 * shade, 1.12 * shade, 0.62 * shade);
    case 'DesertSaguaroCactus':
      return new THREE.Color(0.78 * shade, 0.94 * shade, 0.62 * shade);
    case 'CoralColony':
      return new THREE.Color(0.88 + 0.25 * rng(), 0.55 + 0.4 * rng(), 0.75 + 0.3 * rng());
    case 'GiantKelp':
      return new THREE.Color(0.55 * shade, 0.86 * shade, 0.52 * shade);
  }
}

interface SpeciesPool {
  species: FloraSpeciesId;
  mesh: THREE.InstancedMesh;
  geometry: THREE.BufferGeometry;
  originAttr: THREE.InstancedBufferAttribute;
  capacity: number;
  activeCount: number;
}

export class ProceduralFlora {
  readonly group: THREE.Group;
  private readonly biomeSystem: BiomeSystem;
  private readonly maxInstances: number;
  private readonly material: THREE.ShaderMaterial;
  private readonly pools = new Map<FloraSpeciesId, SpeciesPool>();

  // Scratch objects to avoid per-frame allocations
  private readonly tmpNormal = new THREE.Vector3();
  private readonly tmpRotated = new THREE.Vector3();
  private readonly tmpUp = new THREE.Vector3(0.0, 1.0, 0.0);
  private readonly tmpQuatAlign = new THREE.Quaternion();
  private readonly tmpQuatYaw = new THREE.Quaternion();
  private readonly tmpScale = new THREE.Vector3();
  private readonly tmpPos = new THREE.Vector3();
  private readonly tmpMat = new THREE.Matrix4();
  private readonly tmpColor = new THREE.Color(1.0, 1.0, 1.0);

  constructor(biomeSystem: BiomeSystem, maxInstances = MAX_FLORA_INSTANCES_CAP) {
    this.biomeSystem = biomeSystem;
    const validMax = Number.isFinite(maxInstances) ? Math.trunc(maxInstances) : MAX_FLORA_INSTANCES_CAP;
    this.maxInstances = clamp(validMax, ALL_FLORA_SPECIES.length, MAX_FLORA_INSTANCES_CAP);

    this.group = new THREE.Group();
    this.group.name = 'ProceduralFloraGroup';

    this.material = new THREE.ShaderMaterial({
      vertexShader: FLORA_VERTEX_SHADER,
      fragmentShader: FLORA_FRAGMENT_SHADER,
      side: THREE.DoubleSide,
      uniforms: {
        uTimeSec: { value: 0.0 },
        uPgaG: { value: 0.0 },
        uOverpressureKpa: { value: 0.0 },
        uBlastDirLocal: { value: new THREE.Vector3(1.0, 0.0, 0.0) },
        uLocalTempK: { value: 288.0 },
        uRadFluxKwM2: { value: 0.0 },
        uSunDir: { value: new THREE.Vector3(0.6, 0.5, 0.624).normalize() },
      },
    });

    const baseCap = Math.floor(this.maxInstances / ALL_FLORA_SPECIES.length);
    const remainder = this.maxInstances % ALL_FLORA_SPECIES.length;

    for (let i = 0; i < ALL_FLORA_SPECIES.length; i++) {
      const species = ALL_FLORA_SPECIES[i];
      const cap = baseCap + (i < remainder ? 1 : 0);
      const geometry = buildFloraSpeciesGeometry(species, biomeSystem.seed + i * 97);
      const originArr = new Float32Array(cap * 3);
      const originAttr = new THREE.InstancedBufferAttribute(originArr, 3);
      originAttr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('aInstanceOrigin', originAttr);
      const mesh = new THREE.InstancedMesh(geometry, this.material, cap);
      mesh.name = `Flora_${species}`;
      mesh.frustumCulled = false;
      mesh.count = 0;

      // Pre-allocate instanceColor buffer cleanly so USE_INSTANCING_COLOR compiles cleanly
      for (let k = 0; k < cap; k++) {
        mesh.setColorAt(k, this.tmpColor);
        mesh.setMatrixAt(k, this.tmpMat);
      }
      if (mesh.instanceColor !== null) {
        mesh.instanceColor.needsUpdate = true;
      }

      this.group.add(mesh);
      this.pools.set(species, {
        species,
        mesh,
        geometry,
        originAttr,
        capacity: cap,
        activeCount: 0,
      });
    }
  }

  getSpeciesCount(): number {
    return this.pools.size;
  }

  populateChunk(
    chunkCenterUnit: THREE.Vector3,
    planetRadiusKm: number,
    cameraPosKm: [number, number, number],
    livingComKm: [number, number, number],
    spinAngleRad: number
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

    for (const pool of this.pools.values()) {
      pool.activeCount = 0;
    }

    const [tanU, tanV] = buildOrthonormalBasis(centerUnit);
    const centerSample = this.biomeSystem.sampleSurface(centerUnit);
    const cosSpin = Math.cos(spinAngleRad);
    const sinSpin = Math.sin(spinAngleRad);

    const centerRotX = centerUnit.x * cosSpin + centerUnit.z * sinSpin;
    const centerRotY = centerUnit.y;
    const centerRotZ = -centerUnit.x * sinSpin + centerUnit.z * cosSpin;
    const centerSurfR = safeRadiusKm + centerSample.elevationM * 0.001;
    const dxCam = livingComKm[0] + centerRotX * centerSurfR - cameraPosKm[0];
    const dyCam = livingComKm[1] + centerRotY * centerSurfR - cameraPosKm[1];
    const dzCam = livingComKm[2] + centerRotZ * centerSurfR - cameraPosKm[2];
    const camDistKm = Math.sqrt(dxCam * dxCam + dyCam * dyCam + dzCam * dzCam);

    // Local patch scatter radius scales smoothly from ~120 m at 5 m altitude up to 1.8 km
    const patchRadiusKm = clamp(camDistKm * 1.35, 0.12, 1.8);

    // Quantized deterministic seed per ~1 km patch so panning does not flicker
    const qx = Math.round(centerUnit.x * 4000.0);
    const qy = Math.round(centerUnit.y * 4000.0);
    const qz = Math.round(centerUnit.z * 4000.0);
    const patchSeed =
      (this.biomeSystem.seed ^
        Math.imul(qx, 73856093) ^
        Math.imul(qy, 19349663) ^
        Math.imul(qz, 83492791)) >>>
      0;
    const rng = createMulberry32(patchSeed);

    const candidateCount = Math.min(this.maxInstances, 1024);
    const goldenAngle = Math.PI * (3.0 - Math.sqrt(5.0));

    let totalPlaced = 0;
    for (let i = 0; i < candidateCount && totalPlaced < this.maxInstances; i++) {
      const frac = (i + 0.35) / candidateCount;
      const radialKm = Math.max(0.006, patchRadiusKm * Math.sqrt(frac));
      const theta = i * goldenAngle + (rng() - 0.5) * 0.45;
      const offU = (Math.cos(theta) * radialKm) / safeRadiusKm;
      const offV = (Math.sin(theta) * radialKm) / safeRadiusKm;

      this.tmpNormal
        .copy(centerUnit)
        .addScaledVector(tanU, offU)
        .addScaledVector(tanV, offV)
        .normalize();

      const sample = this.biomeSystem.sampleSurface(this.tmpNormal);
      const allowedSpecies = getBiomeSpeciesList(
        sample.biome,
        sample.elevationM,
        sample.climateTempK
      );
      if (allowedSpecies.length === 0) {
        continue;
      }

      const species = allowedSpecies[Math.floor(rng() * allowedSpecies.length)];
      const pool = this.pools.get(species);
      if (pool === undefined || pool.activeCount >= pool.capacity) {
        continue;
      }

      // Rotate rest-frame normal by planetary spin into authoritative world frame
      this.tmpRotated.set(
        this.tmpNormal.x * cosSpin + this.tmpNormal.z * sinSpin,
        this.tmpNormal.y,
        -this.tmpNormal.x * sinSpin + this.tmpNormal.z * cosSpin
      );

      const surfaceRKm = safeRadiusKm + sample.elevationM * 0.001;
      const wxKm = livingComKm[0] + this.tmpRotated.x * surfaceRKm;
      const wyKm = livingComKm[1] + this.tmpRotated.y * surfaceRKm;
      const wzKm = livingComKm[2] + this.tmpRotated.z * surfaceRKm;

      this.tmpPos.set(
        wxKm - cameraPosKm[0],
        wyKm - cameraPosKm[1],
        wzKm - cameraPosKm[2]
      );

      this.tmpQuatAlign.setFromUnitVectors(this.tmpUp, this.tmpRotated);
      this.tmpQuatYaw.setFromAxisAngle(this.tmpUp, rng() * Math.PI * 2.0);
      this.tmpQuatAlign.multiply(this.tmpQuatYaw);

      const baseScale =
        sample.biome === 'AlpineTundra'
          ? 0.45 + 0.3 * rng()
          : 0.8 + 0.55 * rng();
      const scaleKm = 0.001 * baseScale;
      this.tmpScale.set(scaleKm, scaleKm, scaleKm);

      this.tmpMat.compose(this.tmpPos, this.tmpQuatAlign, this.tmpScale);
      const slot = pool.activeCount;
      pool.mesh.setMatrixAt(slot, this.tmpMat);
      pool.mesh.setColorAt(slot, getSpeciesTint(species, rng));
      const originArr = pool.originAttr.array as Float32Array;
      const oIdx = slot * 3;
      originArr[oIdx] = this.tmpNormal.x * surfaceRKm;
      originArr[oIdx + 1] = this.tmpNormal.y * surfaceRKm;
      originArr[oIdx + 2] = this.tmpNormal.z * surfaceRKm;
      pool.activeCount += 1;
      totalPlaced += 1;
    }

    for (const pool of this.pools.values()) {
      pool.mesh.count = pool.activeCount;
      pool.mesh.instanceMatrix.needsUpdate = true;
      pool.originAttr.needsUpdate = true;
      if (pool.mesh.instanceColor !== null) {
        pool.mesh.instanceColor.needsUpdate = true;
      }
    }
  }

  updateImpactResponse(
    timeSec: number,
    pgaG: number,
    overpressureKpa: number,
    blastDirLocal: THREE.Vector3,
    localTempK: number,
    radFluxKwM2: number
  ): void {
    const uniforms = this.material.uniforms;
    uniforms.uTimeSec.value = Number.isFinite(timeSec) ? timeSec : 0.0;
    uniforms.uPgaG.value = Number.isFinite(pgaG) ? Math.max(0.0, pgaG) : 0.0;
    uniforms.uOverpressureKpa.value = Number.isFinite(overpressureKpa)
      ? Math.max(0.0, overpressureKpa)
      : 0.0;

    const uBlast = uniforms.uBlastDirLocal.value as THREE.Vector3;
    if (
      Number.isFinite(blastDirLocal.x) &&
      Number.isFinite(blastDirLocal.y) &&
      Number.isFinite(blastDirLocal.z) &&
      blastDirLocal.lengthSq() > 1.0e-12
    ) {
      uBlast.copy(blastDirLocal).normalize();
    } else {
      uBlast.set(1.0, 0.0, 0.0);
    }

    uniforms.uLocalTempK.value = Number.isFinite(localTempK) ? localTempK : 288.0;
    uniforms.uRadFluxKwM2.value = Number.isFinite(radFluxKwM2)
      ? Math.max(0.0, radFluxKwM2)
      : 0.0;
  }

  dispose(): void {
    this.group.clear();
    for (const pool of this.pools.values()) {
      pool.geometry.dispose();
      pool.mesh.dispose();
    }
    this.pools.clear();
    this.material.dispose();
  }
}
