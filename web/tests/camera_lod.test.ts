import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import {
  OrbitalSurfaceCamera,
  type CameraMode,
  type SurfaceAnchorCoord,
} from '../src/render/OrbitalSurfaceCamera';
import { SphParticleRenderer, type VisMode } from '../src/render/SphParticleRenderer';
import { AtmosphereShader } from '../src/render/AtmosphereShader';
import { CubeSphereLod } from '../src/render/CubeSphereLod';
import type { ParticleBufferPool, TracerBufferPool } from '../src/net/BinaryProtocol';

function makeParticlePool(capacity: number): ParticleBufferPool {
  return {
    capacity,
    count: 0,
    positionsKm: new Float32Array(3 * capacity),
    props: new Float32Array(4 * capacity),
    meta: new Float32Array(4 * capacity),
  };
}

function makeTracerPool(capacity: number, gridRes = 16): TracerBufferPool {
  return {
    capacity,
    count: 0,
    gridRes,
    anchorIndex: new Uint16Array(capacity),
    flags: new Uint16Array(capacity),
    positionsKm: new Float32Array(3 * capacity),
    dispKm: new Float32Array(capacity),
    normalDelta: new Float32Array(capacity),
    tempK: new Float32Array(capacity),
    damage: new Float32Array(capacity),
    overpressureKpa: new Float32Array(capacity),
    tsunamiHeightM: new Float32Array(capacity),
  };
}

// Shared Cube-Sphere Face Orientation from Spec Section 4.4:
// face 0 (+X): ( 1,  sy, -sx), face 1 (-X): (-1,  sy,  sx)
// face 2 (+Y): ( sx,  1, -sy), face 3 (-Y): ( sx, -1,  sy)
// face 4 (+Z): ( sx, sy,   1), face 5 (-Z): (-sx, sy,  -1)
function cubeFaceUvToUnitNormal(face: number, u: number, v: number): [number, number, number] {
  const sx = 2 * u - 1;
  const sy = 2 * v - 1;
  let x = 0;
  let y = 0;
  let z = 0;
  switch (face) {
    case 0:
      x = 1;
      y = sy;
      z = -sx;
      break;
    case 1:
      x = -1;
      y = sy;
      z = sx;
      break;
    case 2:
      x = sx;
      y = 1;
      z = -sy;
      break;
    case 3:
      x = sx;
      y = -1;
      z = sy;
      break;
    case 4:
      x = sx;
      y = sy;
      z = 1;
      break;
    case 5:
    default:
      x = -sx;
      y = sy;
      z = -1;
      break;
  }
  const invLen = 1 / Math.hypot(x, y, z);
  return [x * invLen, y * invLen, z * invLen];
}

// Structural mock conforming to BiomeSystem (Stage E5 does not exist yet on e4-tests)
function createMockBiomeSystem(elevationM = 120) {
  return {
    sampleSurface(unitNormal: THREE.Vector3) {
      const lat = Math.asin(Math.max(-1, Math.min(1, unitNormal.y)));
      return {
        elevationM,
        latitudeRad: lat,
        moisture01: 0.65,
        climateTempK: 294,
        biome: 'TropicalRainforest' as const,
        albedoRgb: [0.14, 0.42, 0.16] as [number, number, number],
      };
    },
    getElevationM(_unitNormal: THREE.Vector3): number {
      return elevationM;
    },
  };
}

// Structural mock conforming to ImpactBiosphereCoupler (Stage E5 does not exist yet on e4-tests)
function createMockCoupler() {
  return {
    resetBaseline(): void {},
    ingestSimulationState(): void {},
    evaluateSiteExposure(unitNormal: THREE.Vector3, _simTimeSec: number) {
      return {
        siteNormal: unitNormal.clone(),
        arcDistanceM: 0,
        chordDistanceM: 0,
        tsunamiHeightM: 0,
        pWaveArrived: false,
        sWaveArrived: false,
        pgaG: 0,
        radFluxKwM2: 0,
        isWildfireIgnited: false,
        ejectaArrived: false,
        ejectaFluxKgM2: 0,
        blastArrived: false,
        overpressureKpa: 0,
        effectiveTempK: 288,
        crustDamage: 0,
      };
    },
    getGlobalMetrics() {
      return {
        totalBiomassGtC: 550.0,
        baselineBiomassGtC: 550.0,
        shannonDiversityH: 1.9,
        overallSurvivalPct: 100.0,
        globalOpticalDepthTau: 0.0,
        globalDeltaTempK: 0.0,
        survivalByBiome: {
          TropicalRainforest: 100,
          Savanna: 100,
          TemperateForest: 100,
          BorealTaiga: 100,
          AlpineTundra: 100,
          AridSteppe: 100,
          OceanReef: 100,
        },
        survivalByGuild: {
          LargeHerbivore: 100,
          ApexPredator: 100,
          AvianFlock: 100,
          BurrowingRodent: 100,
          Flora: 100,
        },
        extinctionTier: 'Nominal' as const,
      };
    },
  };
}

function vec3Sub(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vec3Len(v: [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function findUniformValue(
  uniforms: Record<string, THREE.IUniform<unknown>>,
  pattern: RegExp
): unknown {
  for (const [key, u] of Object.entries(uniforms)) {
    if (pattern.test(key)) {
      return u.value;
    }
  }
  return undefined;
}

describe('OrbitalSurfaceCamera - AC-14', () => {
  it('enforces [0.001, 1.0e5] km altitude bounds and zooms smoothly from 10,000 km down to 1 m (0.001 km)', () => {
    const cam = new OrbitalSurfaceCamera();
    expect(cam.camera).toBeInstanceOf(THREE.PerspectiveCamera);
    expect(cam.altitudeKm).toBeGreaterThanOrEqual(0.001);
    expect(cam.altitudeKm).toBeLessThanOrEqual(1.0e5);

    const anchor: SurfaceAnchorCoord = { face: 0, u: 0.5, v: 0.5 };
    cam.setMode('SurfaceLock', anchor);
    expect(cam.mode).toBe('SurfaceLock');
    expect(cam.surfaceAnchor).toEqual(anchor);

    // Zoom out to upper bound (1.0e5 km) and verify clamping
    for (let i = 0; i < 300; i++) {
      cam.zoomByWheel(240);
      cam.update(0.1, [0, 0, 0], 6371, 0, () => 0);
      expect(cam.altitudeKm).toBeLessThanOrEqual(1.0e5 + 1e-9);
    }
    expect(cam.altitudeKm).toBeGreaterThanOrEqual(10_000);
    expect(cam.altitudeKm).toBeLessThanOrEqual(1.0e5 + 1e-9);

    // Sweep smoothly down from >= 10,000 km to 0.001 km (1 m)
    const trace: number[] = [cam.altitudeKm];
    for (let i = 0; i < 500; i++) {
      cam.zoomByWheel(-120);
      cam.update(0.05, [0, 0, 0], 6371, 0, () => 0);
      trace.push(cam.altitudeKm);
      expect(Number.isFinite(cam.altitudeKm)).toBe(true);
      expect(cam.altitudeKm).toBeGreaterThanOrEqual(0.001 - 1e-9);
      expect(cam.altitudeKm).toBeLessThanOrEqual(trace[trace.length - 2] + 1e-9);
    }
    // Let any smoothing settle
    for (let i = 0; i < 40; i++) {
      cam.update(0.1, [0, 0, 0], 6371, 0, () => 0);
    }

    expect(trace.some((a) => a >= 10_000)).toBe(true);
    expect(trace.some((a) => a > 10 && a < 1000)).toBe(true);
    expect(cam.altitudeKm).toBeCloseTo(0.001, 5);
  });

  it('tracks translated, deformed, and rotated SurfaceLock anchor and preserves < 1e-6 km precision in toCameraRelativePos', () => {
    const cam = new OrbitalSurfaceCamera();
    const comKm: [number, number, number] = [5e5, -2e5, 1e5];
    const radiusKm = 6371;

    cam.setMode('SurfaceLock', { face: 0, u: 0.5, v: 0.5 });
    for (let i = 0; i < 500; i++) {
      cam.zoomByWheel(-120);
      cam.update(0.05, [0, 0, 0], radiusKm, 0, () => 0);
    }
    for (let i = 0; i < 60; i++) {
      cam.update(0.1, [0, 0, 0], radiusKm, 0, () => 0);
    }
    const posOrigin: [number, number, number] = [...cam.worldPositionKm];

    // Translate COM by [5e5, -2e5, 1e5]
    for (let i = 0; i < 80; i++) {
      cam.update(0.1, comKm, radiusKm, 0, () => 0);
    }
    const posTranslated: [number, number, number] = [...cam.worldPositionKm];
    const deltaTranslate = vec3Sub(posTranslated, posOrigin);
    expect(deltaTranslate[0]).toBeCloseTo(comKm[0], 3);
    expect(deltaTranslate[1]).toBeCloseTo(comKm[1], 3);
    expect(deltaTranslate[2]).toBeCloseTo(comKm[2], 3);

    // Deform surface at (face 0, u=0.5, v=0.5) with unit normal (+1, 0, 0) by +12.345 km
    for (let i = 0; i < 80; i++) {
      cam.update(0.1, comKm, radiusKm, 0, (face, u, v) => {
        expect(face).toBe(0);
        expect(u).toBeCloseTo(0.5, 5);
        expect(v).toBeCloseTo(0.5, 5);
        return 12.345;
      });
    }
    const posDeformed: [number, number, number] = [...cam.worldPositionKm];
    const deltaDeform = vec3Sub(posDeformed, posTranslated);
    expect(deltaDeform[0]).toBeCloseTo(12.345, 3);
    expect(deltaDeform[1]).toBeCloseTo(0, 3);
    expect(deltaDeform[2]).toBeCloseTo(0, 3);

    // Sub-millimeter (1 mm = 1e-6 km) camera-relative precision near [5e5 + 6371, -2e5, 1e5]
    const subMeterOffset: [number, number, number] = [0.000123456, -0.000654321, 0.000444555];
    const targetWorldKm: [number, number, number] = [
      cam.worldPositionKm[0] + subMeterOffset[0],
      cam.worldPositionKm[1] + subMeterOffset[1],
      cam.worldPositionKm[2] + subMeterOffset[2],
    ];
    const outVec = new THREE.Vector3();
    const retVec = cam.toCameraRelativePos(targetWorldKm, outVec);
    expect(retVec).toBe(outVec);
    expect(Math.abs(outVec.x - subMeterOffset[0])).toBeLessThan(1e-6); // < 1e-3 m
    expect(Math.abs(outVec.y - subMeterOffset[1])).toBeLessThan(1e-6);
    expect(Math.abs(outVec.z - subMeterOffset[2])).toBeLessThan(1e-6);

    const selfRel = cam.toCameraRelativePos(cam.worldPositionKm, new THREE.Vector3());
    expect(selfRel.length()).toBeLessThan(1e-9);

    // Rotate spinAngleRad by +pi/2 -> radial vector rotates away from +X while preserving radial distance
    for (let i = 0; i < 80; i++) {
      cam.update(0.1, comKm, radiusKm, Math.PI / 2, () => 12.345);
    }
    const posRotated: [number, number, number] = [...cam.worldPositionKm];
    const rBefore = vec3Sub(posDeformed, comKm);
    const rAfter = vec3Sub(posRotated, comKm);
    expect(Math.abs(vec3Len(rAfter) - vec3Len(rBefore))).toBeLessThan(1e-2);
    const cosAngle =
      (rBefore[0] * rAfter[0] + rBefore[1] * rAfter[1] + rBefore[2] * rAfter[2]) /
      (vec3Len(rBefore) * vec3Len(rAfter));
    expect(Math.abs(cosAngle)).toBeLessThan(0.1);

    // OrbitalFree + orbitByDelta stability
    const freeMode: CameraMode = 'OrbitalFree';
    cam.setMode(freeMode);
    expect(cam.mode).toBe('OrbitalFree');
    const beforeOrbit: [number, number, number] = [...cam.worldPositionKm];
    cam.orbitByDelta(45, -20);
    cam.update(0.05, comKm, radiusKm, 0, () => 0);
    expect(vec3Len(vec3Sub(cam.worldPositionKm, beforeOrbit))).toBeGreaterThan(0);
    expect(Number.isFinite(cam.worldPositionKm[0])).toBe(true);
    expect(Number.isFinite(cam.worldPositionKm[1])).toBe(true);
    expect(Number.isFinite(cam.worldPositionKm[2])).toBe(true);
  });
});

describe('SphParticleRenderer & AtmosphereShader - AC-15', () => {
  it('builds valid GLSL shaders, switches all 6 VisMode uniforms, configures slice plane, and packs camera-relative buffers', () => {
    const renderer = new SphParticleRenderer(2048);
    expect(renderer.mesh).toBeInstanceOf(THREE.Points);
    expect(renderer.mesh.material).toBeInstanceOf(THREE.ShaderMaterial);

    const mat = renderer.mesh.material as THREE.ShaderMaterial;
    expect(mat.vertexShader.length).toBeGreaterThan(40);
    expect(mat.fragmentShader.length).toBeGreaterThan(40);

    const combinedGlsl = `${mat.vertexShader}\n${mat.fragmentShader}`;
    expect(combinedGlsl).toContain('uVisMode');
    expect(mat.fragmentShader).toMatch(/\bdiscard\b/);
    expect(mat.uniforms.uVisMode).toBeDefined();

    const sliceKeys = Object.keys(mat.uniforms).filter((k) => /slice/i.test(k));
    expect(sliceKeys.length).toBeGreaterThan(0);
    expect(combinedGlsl).toMatch(/uSlice/i);

    const modes: VisMode[] = [
      'NaturalBlackbody',
      'TemperatureK',
      'PressureGPa',
      'MaterialLayer',
      'DamageFracture',
      'Velocity',
    ];
    const seenModeValues = new Set<unknown>();
    for (const mode of modes) {
      renderer.setVisMode(mode);
      seenModeValues.add(mat.uniforms.uVisMode.value);
    }
    expect(seenModeValues.size).toBe(6);

    // Slice plane toggle
    renderer.setSlicePlane(true, new THREE.Vector3(0, 1, 0), 250, [10, 0, 0]);
    const sliceEnabledOn = findUniformValue(mat.uniforms, /slice.*enabled|uSliceEnabled/i);
    expect(Boolean(sliceEnabledOn)).toBe(true);

    const sliceNormalVal = findUniformValue(mat.uniforms, /slice.*normal/i);
    if (sliceNormalVal instanceof THREE.Vector3) {
      expect(sliceNormalVal.y).toBeCloseTo(1, 4);
    }

    renderer.setSlicePlane(false, new THREE.Vector3(0, 1, 0), 250, [10, 0, 0]);
    const sliceEnabledOff = findUniformValue(mat.uniforms, /slice.*enabled|uSliceEnabled/i);
    expect(Boolean(sliceEnabledOff)).toBe(false);

    // Pack 256 particles with large COM offset into camera-relative positions
    const count = 256;
    const pool = makeParticlePool(2048);
    pool.count = count;
    const camPosKm: [number, number, number] = [500_000, -200_000, 100_000];
    for (let i = 0; i < count; i++) {
      pool.positionsKm[3 * i + 0] = camPosKm[0] + (i === 0 ? 10 : i * 2);
      pool.positionsKm[3 * i + 1] = camPosKm[1] + (i === 0 ? 20 : -i);
      pool.positionsKm[3 * i + 2] = camPosKm[2] + (i === 0 ? -30 : i * 0.5);
      pool.props[4 * i + 0] = 300 + i * 20;
      pool.props[4 * i + 1] = i * 0.5;
      pool.props[4 * i + 2] = i / count;
      pool.props[4 * i + 3] = 10 + (i % 50);
      pool.meta[4 * i + 0] = i % 5;
      pool.meta[4 * i + 1] = i % 6;
      pool.meta[4 * i + 2] = i & 1;
      pool.meta[4 * i + 3] = 1 + i * 0.1;
    }

    renderer.updateFromPool(pool, camPosKm);
    expect(renderer.mesh.geometry.drawRange.count).toBe(count);

    const posAttr = renderer.mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    expect(posAttr).toBeDefined();
    expect(posAttr.getX(0)).toBeCloseTo(10, 3);
    expect(posAttr.getY(0)).toBeCloseTo(20, 3);
    expect(posAttr.getZ(0)).toBeCloseTo(-30, 3);

    // Empty pool edge case
    pool.count = 0;
    renderer.updateFromPool(pool, camPosKm);
    expect(renderer.mesh.geometry.drawRange.count).toBe(0);

    renderer.dispose();
  });

  it('updates AtmosphereShader uniforms and camera-relative mesh position', () => {
    const atmo = new AtmosphereShader(24);
    expect(atmo.mesh).toBeInstanceOf(THREE.Mesh);
    expect(atmo.mesh.material).toBeInstanceOf(THREE.ShaderMaterial);

    const mat = atmo.mesh.material as THREE.ShaderMaterial;
    expect(mat.vertexShader.length).toBeGreaterThan(20);
    expect(mat.fragmentShader.length).toBeGreaterThan(20);

    const camPosKm: [number, number, number] = [510_000, -200_000, 100_000];
    const comKm: [number, number, number] = [500_000, -200_000, 100_000];
    const sunDir = new THREE.Vector3(1, 0.2, 0).normalize();
    atmo.update(camPosKm, comKm, 6371, sunDir, 3.5, 4500);

    expect(atmo.mesh.position.x).toBeCloseTo(-10_000, 2);
    expect(atmo.mesh.position.y).toBeCloseTo(0, 2);
    expect(atmo.mesh.position.z).toBeCloseTo(0, 2);

    const tauVal = findUniformValue(mat.uniforms, /tau|optical/i);
    expect(tauVal).toBeCloseTo(3.5, 4);

    const glowVal = findUniformValue(mat.uniforms, /plume|glow/i);
    expect(glowVal).toBeCloseTo(4500, 2);

    atmo.update(camPosKm, comKm, 6371, sunDir, 0.0, 288);
    expect(findUniformValue(mat.uniforms, /tau|optical/i)).toBeCloseTo(0.0, 4);
    expect(findUniformValue(mat.uniforms, /plume|glow/i)).toBeCloseTo(288, 2);

    atmo.dispose();
  });
});

describe('CubeSphereLod - AC-16', () => {
  it('subdivides to depth 10 near camera, interpolates tracer fields, drives crack dissolve (0.55..0.85), culls D>=0.85, and supports empty-tracer fallback', () => {
    const biomeMock = createMockBiomeSystem(120);
    const couplerMock = createMockCoupler();
    const lod = new CubeSphereLod(biomeMock as unknown as ConstructorParameters<typeof CubeSphereLod>[0]);

    expect(lod.group).toBeInstanceOf(THREE.Group);
    expect(lod.maxDepth).toBe(10);

    const gridRes = 16;
    const T = 6 * gridRes * gridRes; // 1536
    const tracers = makeTracerPool(T, gridRes);
    tracers.count = T;

    const livingComKm: [number, number, number] = [1000, -500, 250];
    const livingRadiusKm = 6371;

    for (let face = 0; face < 6; face++) {
      for (let gy = 0; gy < gridRes; gy++) {
        for (let gx = 0; gx < gridRes; gx++) {
          const idx = face * gridRes * gridRes + gy * gridRes + gx;
          const u = (gx + 0.5) / gridRes;
          const v = (gy + 0.5) / gridRes;
          const [nx, ny, nz] = cubeFaceUvToUnitNormal(face, u, v);

          let dispKm = 0.0;
          let tempK = 288.0;
          let damage = 0.0;
          let overpressureKpa = 0.0;
          let tsunamiHeightM = 0.0;
          let isOcean = false;
          let isDisrupted = false;

          if (face === 0) {
            // +X intact uplifted terrain
            dispKm = 2.5;
            damage = 0.0;
          } else if (face === 1) {
            // -X cracked zone (0.55 <= D < 0.85)
            dispKm = 1.0;
            tempK = 1400.0;
            damage = 0.7;
          } else if (face === 2) {
            // +Y shattered zone (D >= 0.85)
            dispKm = 20.0;
            tempK = 3500.0;
            damage = 0.92;
            isDisrupted = true;
          } else if (face === 3) {
            // -Y intact ocean with tsunami wave
            isOcean = true;
            dispKm = 0.0;
            overpressureKpa = 45.0;
            tsunamiHeightM = 18.0;
          }

          const r = livingRadiusKm + dispKm;
          tracers.anchorIndex[idx] = idx;
          tracers.flags[idx] = (isOcean ? 1 : 0) | (isDisrupted ? 2 : 0);
          tracers.positionsKm[3 * idx + 0] = livingComKm[0] + r * nx;
          tracers.positionsKm[3 * idx + 1] = livingComKm[1] + r * ny;
          tracers.positionsKm[3 * idx + 2] = livingComKm[2] + r * nz;
          tracers.dispKm[idx] = dispKm;
          tracers.normalDelta[idx] = damage * 0.25;
          tracers.tempK[idx] = tempK;
          tracers.damage[idx] = damage;
          tracers.overpressureKpa[idx] = overpressureKpa;
          tracers.tsunamiHeightM[idx] = tsunamiHeightM;
        }
      }
    }

    // Direct bilinear sampling verification
    const sampleFace0 = lod.sampleTracerField(0, 0.5, 0.5, tracers);
    expect(sampleFace0.dispKm).toBeCloseTo(2.5, 3);
    expect(sampleFace0.damage).toBeCloseTo(0, 4);
    expect(sampleFace0.isOcean).toBe(false);

    const sampleFace1 = lod.sampleTracerField(1, 0.5, 0.5, tracers);
    expect(sampleFace1.damage).toBeCloseTo(0.7, 3);
    expect(sampleFace1.tempK).toBeCloseTo(1400, 1);

    const sampleFace3 = lod.sampleTracerField(3, 0.5, 0.5, tracers);
    expect(sampleFace3.isOcean).toBe(true);
    expect(sampleFace3.tsunamiHeightM).toBeCloseTo(18.0, 2);
    expect(sampleFace3.overpressureKpa).toBeCloseTo(45.0, 2);

    // Position camera at 0.05 km (50 m) directly over face 0 (+X)
    const cam = new OrbitalSurfaceCamera();
    cam.setMode('SurfaceLock', { face: 0, u: 0.5, v: 0.5 });
    for (let i = 0; i < 450; i++) {
      cam.zoomByWheel(-120);
      cam.update(0.05, livingComKm, livingRadiusKm, 0, (f, u, v) =>
        lod.sampleTracerField(f, u, v, tracers).dispKm
      );
      if (cam.altitudeKm <= 0.08) break;
    }
    for (let i = 0; i < 40; i++) {
      cam.update(0.1, livingComKm, livingRadiusKm, 0, (f, u, v) =>
        lod.sampleTracerField(f, u, v, tracers).dispKm
      );
    }
    cam.camera.updateProjectionMatrix();
    cam.camera.updateMatrixWorld();

    // Update LOD across multiple frames to allow full subdivision down to depth 10
    for (let frame = 0; frame < 15; frame++) {
      lod.update(
        cam,
        livingComKm,
        livingRadiusKm,
        0,
        tracers,
        couplerMock as unknown as Parameters<CubeSphereLod['update']>[5]
      );
    }

    expect(lod.getActivePatchCount()).toBeGreaterThan(6);
    expect(lod.getCulledDamagedPatchCount()).toBeGreaterThan(0);

    // Inspect active patches for depth=10 near camera and crack-dissolve uniforms on cracked patches
    const patchMeshes: THREE.Mesh[] = [];
    lod.group.traverse((obj) => {
      if (obj instanceof THREE.Mesh && obj.visible) {
        patchMeshes.push(obj);
      }
    });
    expect(patchMeshes.length).toBeGreaterThan(6);

    const recordedDepths: number[] = [];
    let minPatchSpanKm = Infinity;
    let maxCrackOrDissolveUniform = 0;

    for (const m of patchMeshes) {
      const ud = m.userData as { depth?: unknown; lodDepth?: unknown };
      const depthVal = typeof ud.depth === 'number' ? ud.depth : typeof ud.lodDepth === 'number' ? ud.lodDepth : undefined;
      if (typeof depthVal === 'number') {
        recordedDepths.push(depthVal);
      }

      const geom = m.geometry;
      geom.computeBoundingBox();
      if (geom.boundingBox) {
        const size = new THREE.Vector3();
        geom.boundingBox.getSize(size);
        const span = size.length();
        if (span > 0 && span < minPatchSpanKm) {
          minPatchSpanKm = span;
        }
      }

      if (m.material instanceof THREE.ShaderMaterial) {
        for (const [key, uniform] of Object.entries(m.material.uniforms)) {
          if (/dissolve|crack|fracture|damage/i.test(key) && typeof uniform.value === 'number') {
            maxCrackOrDissolveUniform = Math.max(maxCrackOrDissolveUniform, uniform.value);
          }
        }
      }
    }

    if (recordedDepths.length > 0) {
      expect(Math.max(...recordedDepths)).toBe(10);
    }
    expect(minPatchSpanKm).toBeLessThan(25); // depth-10 ~8-12 km patch vs <=8 >=35 km
    expect(maxCrackOrDissolveUniform).toBeGreaterThanOrEqual(0.1);

    // Also focus camera directly over cracked face 1 (-X) at 50 km altitude
    cam.setMode('SurfaceLock', { face: 1, u: 0.5, v: 0.5 });
    for (let i = 0; i < 40; i++) {
      cam.update(0.1, livingComKm, livingRadiusKm, 0, (f, u, v) =>
        lod.sampleTracerField(f, u, v, tracers).dispKm
      );
    }
    cam.camera.updateProjectionMatrix();
    cam.camera.updateMatrixWorld();

    for (let frame = 0; frame < 10; frame++) {
      lod.update(
        cam,
        livingComKm,
        livingRadiusKm,
        0,
        tracers,
        couplerMock as unknown as Parameters<CubeSphereLod['update']>[5]
      );
    }
    expect(lod.getActivePatchCount()).toBeGreaterThan(0);
    expect(lod.getCulledDamagedPatchCount()).toBeGreaterThan(0);

    // Fallback with tracers.count === 0 (uses spinAngleRad + livingComKm + livingRadiusKm)
    const emptyTracers = makeTracerPool(0, 0);
    lod.update(
      cam,
      livingComKm,
      livingRadiusKm,
      0.42,
      emptyTracers,
      couplerMock as unknown as Parameters<CubeSphereLod['update']>[5]
    );
    expect(lod.getActivePatchCount()).toBeGreaterThan(0);
    expect(lod.getCulledDamagedPatchCount()).toBe(0);

    lod.dispose();
  });
});
