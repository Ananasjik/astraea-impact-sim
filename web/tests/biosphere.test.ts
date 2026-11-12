import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import {
  BiomeSystem,
  type BiomeId,
  type SurfaceSample,
} from '../src/biosphere/BiomeSystem';
import {
  ProceduralFlora,
  buildFloraSpeciesGeometry,
  type FloraSpeciesId,
} from '../src/biosphere/ProceduralFlora';
import {
  ProceduralFauna,
  transitionFaunaState,
  type FaunaGuild,
  type FaunaState,
  type FaunaStimulus,
} from '../src/biosphere/ProceduralFauna';

const ALL_BIOMES: readonly BiomeId[] = [
  'TropicalRainforest',
  'Savanna',
  'TemperateForest',
  'BorealTaiga',
  'AlpineTundra',
  'AridSteppe',
  'OceanReef',
];

const ALL_FLORA_SPECIES: readonly FloraSpeciesId[] = [
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

const ALL_FAUNA_GUILDS: readonly FaunaGuild[] = [
  'HerbivoreHerd',
  'PursuitPredator',
  'AvianFlock',
  'PelagicMarine',
];

const ALL_FAUNA_STATES: readonly FaunaState[] = [
  'Grazing',
  'Alert',
  'StampedeOrMassFlight',
  'ThermalEscapeOrBurrow',
  'Lethal',
];

const PLANET_RADIUS_KM = 6371.0;

function generateFibonacciSphereNormals(count: number): THREE.Vector3[] {
  const normals: THREE.Vector3[] = [];
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (2 * (i + 0.5)) / count;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const phi = i * goldenAngle;
    normals.push(new THREE.Vector3(Math.cos(phi) * r, y, Math.sin(phi) * r).normalize());
  }
  return normals;
}

function findLandNormal(biomeSystem: BiomeSystem, samples: THREE.Vector3[]): THREE.Vector3 {
  // Prefer a lush vegetated land biome where flora density is expected to be high
  for (const n of samples) {
    const s = biomeSystem.sampleSurface(n);
    if (
      s.elevationM >= 20 &&
      s.elevationM < 2500 &&
      (s.biome === 'TropicalRainforest' || s.biome === 'TemperateForest' || s.biome === 'Savanna')
    ) {
      return n.clone();
    }
  }
  for (const n of samples) {
    const elev = biomeSystem.getElevationM(n);
    if (elev >= 10 && elev < 3000) {
      return n.clone();
    }
  }
  throw new Error('Failed to locate any land sample on BiomeSystem');
}

function findShallowOceanNormal(biomeSystem: BiomeSystem, samples: THREE.Vector3[]): THREE.Vector3 {
  for (const n of samples) {
    const elev = biomeSystem.getElevationM(n);
    if (elev > -145 && elev < -2) {
      return n.clone();
    }
  }

  // Bisect along great-circle arc between a land normal and a deep ocean normal
  const land = findLandNormal(biomeSystem, samples);
  let deep: THREE.Vector3 | null = null;
  for (const n of samples) {
    if (biomeSystem.getElevationM(n) < -50) {
      deep = n.clone();
      break;
    }
  }
  if (deep === null) {
    throw new Error('Failed to locate any ocean sample on BiomeSystem');
  }

  let a = land.clone();
  let b = deep.clone();
  for (let iter = 0; iter < 32; iter++) {
    const mid = a.clone().add(b).normalize();
    const elev = biomeSystem.getElevationM(mid);
    if (elev > -145 && elev < -2) {
      return mid;
    }
    if (elev >= -2) {
      a = mid;
    } else {
      b = mid;
    }
  }
  return b;
}

function collectInstancedMeshes(root: THREE.Object3D): THREE.InstancedMesh[] {
  const meshes: THREE.InstancedMesh[] = [];
  root.traverse((obj) => {
    if (obj instanceof THREE.InstancedMesh) {
      meshes.push(obj);
    }
  });
  return meshes;
}

function snapshotMaterialState(root: THREE.Object3D): Array<Record<string, unknown>> {
  const snapshots: Array<Record<string, unknown>> = [];
  root.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of mats) {
        if (mat instanceof THREE.ShaderMaterial) {
          const entry: Record<string, unknown> = {};
          for (const [k, uniform] of Object.entries(mat.uniforms)) {
            const val = uniform.value;
            if (typeof val === 'number' || typeof val === 'boolean') {
              entry[k] = val;
            } else if (val instanceof THREE.Vector3) {
              entry[k] = [val.x, val.y, val.z];
            } else if (val instanceof THREE.Vector2) {
              entry[k] = [val.x, val.y];
            } else if (val instanceof THREE.Color) {
              entry[k] = [val.r, val.g, val.b];
            }
          }
          snapshots.push(entry);
        }
      }
    }
  });
  return snapshots;
}

function snapshotInstancedBuffers(meshes: THREE.InstancedMesh[]): Array<{
  count: number;
  matrixSum: number;
  colorSum: number;
}> {
  return meshes.map((mesh) => {
    let matrixSum = 0;
    const matArr = mesh.instanceMatrix.array;
    const limit = Math.min(matArr.length, mesh.count * 16);
    for (let i = 0; i < limit; i++) {
      expect(Number.isFinite(matArr[i])).toBe(true);
      matrixSum += Math.abs(matArr[i]);
    }
    let colorSum = 0;
    if (mesh.instanceColor !== null) {
      const colArr = mesh.instanceColor.array;
      const colLimit = Math.min(colArr.length, mesh.count * 3);
      for (let i = 0; i < colLimit; i++) {
        expect(Number.isFinite(colArr[i])).toBe(true);
        colorSum += colArr[i];
      }
    }
    return { count: mesh.count, matrixSum, colorSum };
  });
}

function makeStimulus(overrides: Partial<FaunaStimulus> = {}): FaunaStimulus {
  return {
    tImpactElapsedSec: 0,
    impactorDistRadii: 10.0,
    pgaG: 0.0,
    overpressureKpa: 0.0,
    localTempK: 290.0,
    radFluxKwM2: 0.0,
    ejectaFluxKgM2: 0.0,
    tsunamiHeightM: 0.0,
    crustDamage: 0.0,
    impactAzimuthLocal: new THREE.Vector3(1, 0, 0),
    ...overrides,
  };
}

function sumStateCounts(counts: Record<FaunaState, number>): number {
  let total = 0;
  for (const state of ALL_FAUNA_STATES) {
    total += counts[state];
  }
  return total;
}

describe('BiomeSystem (AC-17)', () => {
  it('produces deterministic surface samples and elevation for identical seeds and varies across distinct seeds', () => {
    const b1 = new BiomeSystem(42);
    const b2 = new BiomeSystem(42);
    const bOther = new BiomeSystem(1337);
    const bDefault = new BiomeSystem();

    const probes = generateFibonacciSphereNormals(64);
    let elevationDifferences = 0;

    for (const n of probes) {
      const s1: SurfaceSample = b1.sampleSurface(n);
      const s2: SurfaceSample = b2.sampleSurface(n);
      const sOther: SurfaceSample = bOther.sampleSurface(n);
      const e1 = b1.getElevationM(n);
      const e2 = b2.getElevationM(n);

      expect(e1).toBe(e2);
      expect(s1.elevationM).toBe(e1);
      expect(s1.elevationM).toBe(s2.elevationM);
      expect(s1.latitudeRad).toBe(s2.latitudeRad);
      expect(s1.moisture01).toBe(s2.moisture01);
      expect(s1.climateTempK).toBe(s2.climateTempK);
      expect(s1.biome).toBe(s2.biome);
      expect(s1.albedoRgb).toEqual(s2.albedoRgb);

      if (Math.abs(s1.elevationM - sOther.elevationM) > 1.0) {
        elevationDifferences++;
      }

      const sDef = bDefault.sampleSurface(n);
      expect(Number.isFinite(sDef.elevationM)).toBe(true);
      expect(ALL_BIOMES).toContain(sDef.biome);
    }

    expect(elevationDifferences).toBeGreaterThan(16);
  });

  it('covers all 7 BiomeId categories with valid physical bands across 2,048 Fibonacci samples', () => {
    const biomeSystem = new BiomeSystem(42);
    const samples = generateFibonacciSphereNormals(2048);

    const counts: Record<BiomeId, number> = {
      TropicalRainforest: 0,
      Savanna: 0,
      TemperateForest: 0,
      BorealTaiga: 0,
      AlpineTundra: 0,
      AridSteppe: 0,
      OceanReef: 0,
    };

    for (const n of samples) {
      const s = biomeSystem.sampleSurface(n);
      expect(Number.isFinite(s.elevationM)).toBe(true);
      expect(s.elevationM).toBeGreaterThanOrEqual(-25000);
      expect(s.elevationM).toBeLessThanOrEqual(25000);

      const expectedLat = Math.asin(Math.max(-1, Math.min(1, n.y)));
      expect(Math.abs(s.latitudeRad - expectedLat)).toBeLessThan(1e-3);

      expect(Number.isFinite(s.moisture01)).toBe(true);
      expect(s.moisture01).toBeGreaterThanOrEqual(0);
      expect(s.moisture01).toBeLessThanOrEqual(1);

      expect(Number.isFinite(s.climateTempK)).toBe(true);
      expect(s.climateTempK).toBeGreaterThan(150);
      expect(s.climateTempK).toBeLessThan(360);

      expect(s.albedoRgb).toHaveLength(3);
      for (const c of s.albedoRgb) {
        expect(Number.isFinite(c)).toBe(true);
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(1);
      }

      expect(ALL_BIOMES).toContain(s.biome);
      counts[s.biome]++;
    }

    for (const biome of ALL_BIOMES) {
      expect(counts[biome], `Missing coverage for biome ${biome}`).toBeGreaterThan(0);
    }
  });

  it('classifies all 7 biomes strictly according to elevation, latitude, moisture, and temperature thresholds', () => {
    // Ocean overrides any latitude/temperature
    expect(BiomeSystem.classifyBiome(-15, 0.0, 0.9, 300)).toBe('OceanReef');
    expect(BiomeSystem.classifyBiome(-2500, 1.3, 0.1, 250)).toBe('OceanReef');

    // AlpineTundra: elevation >= 3200 m OR climateTempK < 268 K OR polar dry
    expect(BiomeSystem.classifyBiome(3200, 0.0, 0.8, 298)).toBe('AlpineTundra');
    expect(BiomeSystem.classifyBiome(4500, 0.1, 0.6, 295)).toBe('AlpineTundra');
    expect(BiomeSystem.classifyBiome(200, 0.2, 0.7, 265)).toBe('AlpineTundra');
    expect(BiomeSystem.classifyBiome(150, 1.1, 0.2, 272)).toBe('AlpineTundra');

    // BorealTaiga: |lat| > 55 deg (~0.96 rad) and moisture01 >= 0.30 (with non-alpine temp/elev)
    expect(BiomeSystem.classifyBiome(250, 1.05, 0.55, 275)).toBe('BorealTaiga');
    expect(BiomeSystem.classifyBiome(250, -1.05, 0.55, 275)).toBe('BorealTaiga');

    // Tropical climate (T >= 293 K)
    expect(BiomeSystem.classifyBiome(120, 0.05, 0.78, 299)).toBe('TropicalRainforest');
    expect(BiomeSystem.classifyBiome(300, 0.2, 0.45, 297)).toBe('Savanna');
    expect(BiomeSystem.classifyBiome(400, 0.35, 0.15, 303)).toBe('AridSteppe');

    // Temperate climate (268 K <= T < 293 K)
    expect(BiomeSystem.classifyBiome(250, 0.7, 0.6, 284)).toBe('TemperateForest');
    expect(BiomeSystem.classifyBiome(400, 0.5, 0.22, 284)).toBe('AridSteppe');
  });
});

describe('ProceduralFlora (AC-18)', () => {
  it('builds valid BufferGeometry with required attributes and zero NaNs for all 9 FloraSpeciesId species', () => {
    for (const species of ALL_FLORA_SPECIES) {
      const geom = buildFloraSpeciesGeometry(species, 42);
      try {
        expect(geom).toBeInstanceOf(THREE.BufferGeometry);

        const pos = geom.getAttribute('position');
        const normal = geom.getAttribute('normal');
        const branchWeight = geom.getAttribute('aBranchWeight');
        const partType = geom.getAttribute('aPartType');

        expect(pos, `${species} missing position`).toBeDefined();
        expect(normal, `${species} missing normal`).toBeDefined();
        expect(branchWeight, `${species} missing aBranchWeight`).toBeDefined();
        expect(partType, `${species} missing aPartType`).toBeDefined();

        expect(pos.itemSize).toBe(3);
        expect(normal.itemSize).toBe(3);
        expect(branchWeight.itemSize).toBe(1);
        expect(partType.itemSize).toBe(1);

        expect(pos.count).toBeGreaterThan(0);
        expect(normal.count).toBe(pos.count);
        expect(branchWeight.count).toBe(pos.count);
        expect(partType.count).toBe(pos.count);

        for (let i = 0; i < pos.count; i++) {
          const px = pos.getX(i);
          const py = pos.getY(i);
          const pz = pos.getZ(i);
          expect(Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz)).toBe(true);

          const nx = normal.getX(i);
          const ny = normal.getY(i);
          const nz = normal.getZ(i);
          expect(Number.isFinite(nx) && Number.isFinite(ny) && Number.isFinite(nz)).toBe(true);
          expect(Math.hypot(nx, ny, nz)).toBeGreaterThan(0.1);

          const bw = branchWeight.getX(i);
          expect(Number.isFinite(bw)).toBe(true);
          expect(bw).toBeGreaterThanOrEqual(0);
          expect(bw).toBeLessThanOrEqual(1);

          const pt = partType.getX(i);
          expect(Number.isFinite(pt)).toBe(true);
        }

        const index = geom.getIndex();
        if (index !== null) {
          expect(index.count).toBeGreaterThan(0);
          for (let i = 0; i < index.count; i++) {
            const idx = index.getX(i);
            expect(idx).toBeGreaterThanOrEqual(0);
            expect(idx).toBeLessThan(pos.count);
          }
        }
      } finally {
        geom.dispose();
      }
    }
  });

  it('populates land and ocean chunks within 4096 instance cap and updates blast/thermal shader uniforms', () => {
    const biomeSystem = new BiomeSystem(42);
    const samples = generateFibonacciSphereNormals(2048);
    const landNormal = findLandNormal(biomeSystem, samples);
    const shallowOceanNormal = findShallowOceanNormal(biomeSystem, samples);

    const flora = new ProceduralFlora(biomeSystem, 4096);
    const oceanFlora = new ProceduralFlora(biomeSystem, 4096);

    try {
      expect(flora.group).toBeInstanceOf(THREE.Group);
      expect(flora.getSpeciesCount()).toBeGreaterThanOrEqual(8);
      expect(flora.getSpeciesCount()).toBe(9);

      const landCamKm: [number, number, number] = [
        landNormal.x * (PLANET_RADIUS_KM + 0.5),
        landNormal.y * (PLANET_RADIUS_KM + 0.5),
        landNormal.z * (PLANET_RADIUS_KM + 0.5),
      ];
      flora.populateChunk(landNormal, PLANET_RADIUS_KM, landCamKm, [0, 0, 0], 0.0);

      const landMeshes = collectInstancedMeshes(flora.group);
      expect(landMeshes.length).toBeGreaterThan(0);
      const landInstanceCount = landMeshes.reduce((acc, m) => acc + m.count, 0);
      expect(landInstanceCount).toBeGreaterThanOrEqual(1);
      expect(landInstanceCount).toBeLessThanOrEqual(4096);

      // Baseline benign response
      flora.updateImpactResponse(1.0, 0.0, 0.0, new THREE.Vector3(1, 0, 0), 292.0, 0.0);
      const benignUniforms = snapshotMaterialState(flora.group);
      const benignBuffers = snapshotInstancedBuffers(landMeshes);

      // High blast overpressure & seismic PGA response
      flora.updateImpactResponse(2.5, 1.5, 60.0, new THREE.Vector3(0, 0, 1), 305.0, 4.0);
      const blastUniforms = snapshotMaterialState(flora.group);
      const blastBuffers = snapshotInstancedBuffers(landMeshes);

      // High thermal radiation & ignition response
      flora.updateImpactResponse(4.0, 1.5, 60.0, new THREE.Vector3(0, 0, 1), 900.0, 50.0);
      const thermalUniforms = snapshotMaterialState(flora.group);
      const thermalBuffers = snapshotInstancedBuffers(landMeshes);

      const uniformChangedBlast = JSON.stringify(blastUniforms) !== JSON.stringify(benignUniforms);
      const bufferChangedBlast = JSON.stringify(blastBuffers) !== JSON.stringify(benignBuffers);
      expect(uniformChangedBlast || bufferChangedBlast).toBe(true);

      const uniformChangedThermal = JSON.stringify(thermalUniforms) !== JSON.stringify(blastUniforms);
      const bufferChangedThermal = JSON.stringify(thermalBuffers) !== JSON.stringify(blastBuffers);
      expect(uniformChangedThermal || bufferChangedThermal).toBe(true);

      // Populating a second ocean chunk onto the same flora never exceeds maxInstances (4096)
      const oceanCamKm: [number, number, number] = [
        shallowOceanNormal.x * (PLANET_RADIUS_KM + 0.2),
        shallowOceanNormal.y * (PLANET_RADIUS_KM + 0.2),
        shallowOceanNormal.z * (PLANET_RADIUS_KM + 0.2),
      ];
      flora.populateChunk(shallowOceanNormal, PLANET_RADIUS_KM, oceanCamKm, [0, 0, 0], 0.0);
      const combinedCount = collectInstancedMeshes(flora.group).reduce((acc, m) => acc + m.count, 0);
      expect(combinedCount).toBeLessThanOrEqual(4096);

      // Ocean reef chunk on dedicated ProceduralFlora populates marine flora cleanly
      oceanFlora.populateChunk(shallowOceanNormal, PLANET_RADIUS_KM, oceanCamKm, [0, 0, 0], 0.0);
      const oceanMeshes = collectInstancedMeshes(oceanFlora.group);
      const oceanCount = oceanMeshes.reduce((acc, m) => acc + m.count, 0);
      expect(oceanCount).toBeGreaterThanOrEqual(1);
      expect(oceanCount).toBeLessThanOrEqual(4096);
    } finally {
      flora.dispose();
      oceanFlora.dispose();
    }
  });
});

describe('ProceduralFauna (AC-19)', () => {
  it('executes exact Spec 5.5 transitionFaunaState thresholds across all 4 guilds and marine tsunami exemption', () => {
    for (const guild of ALL_FAUNA_GUILDS) {
      // Benign stays Grazing
      expect(transitionFaunaState('Grazing', guild, makeStimulus(), true)).toBe('Grazing');
      expect(transitionFaunaState('Grazing', guild, makeStimulus(), false)).toBe('Grazing');

      // Grazing -> Alert
      expect(
        transitionFaunaState('Grazing', guild, makeStimulus({ impactorDistRadii: 3.5 }), true)
      ).toBe('Alert');
      expect(transitionFaunaState('Grazing', guild, makeStimulus({ pgaG: 0.02 }), true)).toBe(
        'Alert'
      );

      // Alert -> StampedeOrMassFlight
      expect(transitionFaunaState('Alert', guild, makeStimulus({ pgaG: 0.1 }), true)).toBe(
        'StampedeOrMassFlight'
      );
      expect(
        transitionFaunaState('Alert', guild, makeStimulus({ overpressureKpa: 3.0 }), true)
      ).toBe('StampedeOrMassFlight');
      expect(
        transitionFaunaState('Alert', guild, makeStimulus({ radFluxKwM2: 6.0 }), true)
      ).toBe('StampedeOrMassFlight');

      // Grazing | Alert | StampedeOrMassFlight -> ThermalEscapeOrBurrow
      for (const srcState of ['Grazing', 'Alert', 'StampedeOrMassFlight'] as const) {
        expect(
          transitionFaunaState(srcState, guild, makeStimulus({ radFluxKwM2: 15.0 }), true)
        ).toBe('ThermalEscapeOrBurrow');
        expect(
          transitionFaunaState(srcState, guild, makeStimulus({ localTempK: 360.0 }), true)
        ).toBe('ThermalEscapeOrBurrow');
      }

      // Any non-lethal state -> Lethal across all lethal conditions
      for (const srcState of [
        'Grazing',
        'Alert',
        'StampedeOrMassFlight',
        'ThermalEscapeOrBurrow',
      ] as const) {
        expect(
          transitionFaunaState(srcState, guild, makeStimulus({ localTempK: 450.0 }), true)
        ).toBe('Lethal');
        expect(
          transitionFaunaState(srcState, guild, makeStimulus({ overpressureKpa: 90.0 }), true)
        ).toBe('Lethal');
        expect(transitionFaunaState(srcState, guild, makeStimulus({ pgaG: 3.0 }), true)).toBe(
          'Lethal'
        );
        expect(
          transitionFaunaState(srcState, guild, makeStimulus({ ejectaFluxKgM2: 60.0 }), true)
        ).toBe('Lethal');
        expect(
          transitionFaunaState(srcState, guild, makeStimulus({ crustDamage: 0.75 }), true)
        ).toBe('Lethal');
        expect(
          transitionFaunaState(srcState, guild, makeStimulus({ tsunamiHeightM: 25.0 }), true)
        ).toBe('Lethal');

        // Marine exemption: tsunamiHeightM > 20 when isLand === false is NOT lethal
        expect(
          transitionFaunaState(srcState, guild, makeStimulus({ tsunamiHeightM: 40.0 }), false)
        ).not.toBe('Lethal');
      }

      // Lethal is an absorbing state even when stimulus returns to benign
      expect(transitionFaunaState('Lethal', guild, makeStimulus(), true)).toBe('Lethal');
      expect(transitionFaunaState('Lethal', guild, makeStimulus(), false)).toBe('Lethal');
    }
  });

  it('spawns chunk agents within 512 cap and steps state counts cleanly through escalating stimuli', () => {
    const biomeSystem = new BiomeSystem(42);
    const samples = generateFibonacciSphereNormals(1024);
    const landNormal = findLandNormal(biomeSystem, samples);
    const oceanNormal = findShallowOceanNormal(biomeSystem, samples);

    const fauna = new ProceduralFauna(biomeSystem, 512);
    try {
      expect(fauna.group).toBeInstanceOf(THREE.Group);

      fauna.spawnChunkAgents(landNormal, PLANET_RADIUS_KM, 42);
      fauna.spawnChunkAgents(oceanNormal, PLANET_RADIUS_KM, 43);

      const camKm: [number, number, number] = [
        landNormal.x * (PLANET_RADIUS_KM + 0.5),
        landNormal.y * (PLANET_RADIUS_KM + 0.5),
        landNormal.z * (PLANET_RADIUS_KM + 0.5),
      ];

      fauna.step(0.1, 0.1, makeStimulus(), camKm, [0, 0, 0], PLANET_RADIUS_KM, 0.0);
      const initialCounts = fauna.getStateCounts();
      const totalAgents = sumStateCounts(initialCounts);
      expect(totalAgents).toBeGreaterThan(0);
      expect(totalAgents).toBeLessThanOrEqual(512);
      expect(initialCounts.Grazing).toBeGreaterThan(0);

      // Alert wave
      fauna.step(
        0.5,
        1.0,
        makeStimulus({ impactorDistRadii: 3.0, pgaG: 0.02 }),
        camKm,
        [0, 0, 0],
        PLANET_RADIUS_KM,
        0.0
      );
      const alertCounts = fauna.getStateCounts();
      expect(sumStateCounts(alertCounts)).toBeLessThanOrEqual(512);
      expect(alertCounts.Alert + alertCounts.StampedeOrMassFlight).toBeGreaterThan(0);

      // Stampede / Mass Flight wave
      fauna.step(
        0.5,
        2.0,
        makeStimulus({ impactorDistRadii: 1.5, pgaG: 0.25, overpressureKpa: 5.0 }),
        camKm,
        [0, 0, 0],
        PLANET_RADIUS_KM,
        0.0
      );
      const stampedeCounts = fauna.getStateCounts();
      expect(sumStateCounts(stampedeCounts)).toBeLessThanOrEqual(512);
      expect(stampedeCounts.StampedeOrMassFlight).toBeGreaterThan(0);

      // Thermal Escape / Burrow wave
      fauna.step(
        0.5,
        3.0,
        makeStimulus({
          impactorDistRadii: 1.1,
          pgaG: 0.5,
          overpressureKpa: 10.0,
          radFluxKwM2: 20.0,
          localTempK: 365.0,
        }),
        camKm,
        [0, 0, 0],
        PLANET_RADIUS_KM,
        0.0
      );
      const thermalCounts = fauna.getStateCounts();
      expect(sumStateCounts(thermalCounts)).toBeLessThanOrEqual(512);
      expect(thermalCounts.ThermalEscapeOrBurrow).toBeGreaterThan(0);

      // Lethal shockwave
      fauna.step(
        0.5,
        4.0,
        makeStimulus({
          impactorDistRadii: 1.0,
          pgaG: 4.0,
          overpressureKpa: 120.0,
          localTempK: 600.0,
          crustDamage: 0.85,
        }),
        camKm,
        [0, 0, 0],
        PLANET_RADIUS_KM,
        0.0
      );
      const lethalCounts = fauna.getStateCounts();
      expect(sumStateCounts(lethalCounts)).toBeLessThanOrEqual(512);
      expect(lethalCounts.Lethal).toBeGreaterThan(0);

      // Verify InstancedMesh caps and matrix finiteness
      const meshes = collectInstancedMeshes(fauna.group);
      const instTotal = meshes.reduce((acc, m) => acc + m.count, 0);
      expect(instTotal).toBeLessThanOrEqual(512);
      snapshotInstancedBuffers(meshes);
    } finally {
      fauna.dispose();
    }
  });
});
