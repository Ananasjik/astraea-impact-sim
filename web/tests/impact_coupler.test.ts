import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import {
  createParticleBufferPool,
  createTracerBufferPool,
  TRACER_FLAG_OCEAN,
  type ParticleBufferPool,
  type TracerBufferPool,
} from '../src/net/BinaryProtocol';
import type { TelemetrySnapshotDto } from '../src/net/SimClient';
import { BiomeSystem, type BiomeId } from '../src/biosphere/BiomeSystem';
import type { FaunaGuild } from '../src/biosphere/ProceduralFauna';
import {
  ImpactBiosphereCoupler,
  type BiosphereMetrics,
  type ExtinctionTier,
  type SitePhysicalExposure,
} from '../src/biosphere/ImpactBiosphereCoupler';

const ALL_BIOMES: readonly BiomeId[] = [
  'TropicalRainforest',
  'Savanna',
  'TemperateForest',
  'BorealTaiga',
  'AlpineTundra',
  'AridSteppe',
  'OceanReef',
];

const ALL_GUILDS: readonly (FaunaGuild | 'Flora')[] = [
  'HerbivoreHerd',
  'PursuitPredator',
  'AvianFlock',
  'PelagicMarine',
  'Flora',
];

const EARTH_RADIUS_KM = 6371.0;

function cubeFaceUvToUnitNormal(face: number, u: number, v: number): THREE.Vector3 {
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
  return new THREE.Vector3(x, y, z).normalize();
}

function buildOrthonormalBasis(axis: THREE.Vector3): {
  forward: THREE.Vector3;
  u: THREE.Vector3;
  v: THREE.Vector3;
} {
  const forward = axis.clone().normalize();
  const helper = Math.abs(forward.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const u = new THREE.Vector3().crossVectors(helper, forward).normalize();
  const v = new THREE.Vector3().crossVectors(forward, u).normalize();
  return { forward, u, v };
}

function findLandSiteNormal(biomeSystem: BiomeSystem): THREE.Vector3 {
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < 1024; i++) {
    const y = 1 - (2 * (i + 0.5)) / 1024;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const phi = i * goldenAngle;
    const n = new THREE.Vector3(Math.cos(phi) * r, y, Math.sin(phi) * r).normalize();
    const s = biomeSystem.sampleSurface(n);
    if (s.elevationM >= 30 && s.elevationM < 2500) {
      return n;
    }
  }
  throw new Error('Could not locate a land normal on BiomeSystem');
}

function findOceanSiteAtAngle(
  biomeSystem: BiomeSystem,
  axis: THREE.Vector3,
  thetaRad: number
): THREE.Vector3 {
  const { forward, u, v } = buildOrthonormalBasis(axis);
  const cosT = Math.cos(thetaRad);
  const sinT = Math.sin(thetaRad);

  let bestNormal: THREE.Vector3 | null = null;
  let lowestElev = Infinity;

  const steps = 128;
  for (let i = 0; i < steps; i++) {
    const phi = (2 * Math.PI * i) / steps;
    const candidate = new THREE.Vector3()
      .addScaledVector(forward, cosT)
      .addScaledVector(u, sinT * Math.cos(phi))
      .addScaledVector(v, sinT * Math.sin(phi))
      .normalize();
    const elev = biomeSystem.getElevationM(candidate);
    if (elev < -50) {
      return candidate;
    }
    if (elev < lowestElev) {
      lowestElev = elev;
      bestNormal = candidate;
    }
  }
  if (bestNormal === null) {
    throw new Error('Failed to generate site normal on ring');
  }
  return bestNormal;
}

function makeTelemetry(overrides: Partial<TelemetrySnapshotDto> = {}): TelemetrySnapshotDto {
  return {
    type: 'telemetry',
    frameSeq: 1,
    simTimeSec: 0,
    dtSec: 1.0,
    state: 'playing',
    preset: 'LivingGaiaAsteroidCataclysm',
    outcome: 'Approaching',
    kineticEnergyJ: 1.0e28,
    potentialEnergyJ: -2.24e32,
    internalEnergyJ: 1.5e30,
    radiatedEnergyJ: 0.0,
    totalEnergyJ: -2.225e32,
    energyDriftRel: 1.0e-6,
    momentumKgMS: [0, 0, 0],
    angularMomentumKgM2S: [0, 0, 4.4e33],
    virialRatio: 0.99,
    maxTempK: 295,
    maxPressureGPa: 360,
    boundMassFrac: 1.0,
    diskMassFrac: 0.0,
    ejectaMassFrac: 0.0,
    qRatioQrd: 0.01,
    gpuStepMs: 8.5,
    livingBodyComKm: [0, 0, 0],
    livingBodyRadiusKm: EARTH_RADIUS_KM,
    livingBodySpinRadS: 7.292115e-5,
    impactorComKm: [25000, 0, 0],
    tracerGridRes: 16,
    livingBodyPresent: true,
    ...overrides,
  };
}

function populateSphereTracers(
  tracers: TracerBufferPool,
  gridRes: number,
  biomeSystem: BiomeSystem,
  impactNormal: THREE.Vector3,
  craterCosThreshold: number,
  craterState: {
    tempK: number;
    damage: number;
    overpressureKpa: number;
    tsunamiHeightM: number;
    dispKm: number;
  },
  ambientState: {
    tempK: number;
    damage: number;
    overpressureKpa: number;
    tsunamiHeightM: number;
    dispKm: number;
  }
): void {
  const total = 6 * gridRes * gridRes;
  tracers.count = total;
  tracers.gridRes = gridRes;

  let idx = 0;
  for (let face = 0; face < 6; face++) {
    for (let gy = 0; gy < gridRes; gy++) {
      const v = (gy + 0.5) / gridRes;
      for (let gx = 0; gx < gridRes; gx++) {
        const u = (gx + 0.5) / gridRes;
        const n = cubeFaceUvToUnitNormal(face, u, v);
        const elev = biomeSystem.getElevationM(n);
        const isOcean = elev < 0;
        const inCrater = n.dot(impactNormal) >= craterCosThreshold;
        const st = inCrater ? craterState : ambientState;

        tracers.anchorIndex[idx] = idx;
        tracers.flags[idx] = isOcean ? TRACER_FLAG_OCEAN : 0;
        tracers.positionsKm[idx * 3] = n.x * EARTH_RADIUS_KM;
        tracers.positionsKm[idx * 3 + 1] = n.y * EARTH_RADIUS_KM;
        tracers.positionsKm[idx * 3 + 2] = n.z * EARTH_RADIUS_KM;
        tracers.dispKm[idx] = st.dispKm;
        tracers.normalDelta[idx] = inCrater ? 0.2 : 0.0;
        tracers.tempK[idx] = st.tempK;
        tracers.damage[idx] = st.damage;
        tracers.overpressureKpa[idx] = st.overpressureKpa;
        tracers.tsunamiHeightM[idx] = isOcean ? st.tsunamiHeightM : 0.0;
        idx++;
      }
    }
  }
}

function populateParticlesWithPlume(
  particles: ParticleBufferPool,
  impactNormal: THREE.Vector3,
  plumeActive: boolean,
  plumeTempK: number,
  plumeAltitudeKm: number,
  plumeRadiusKm: number
): void {
  const { forward, u, v } = buildOrthonormalBasis(impactNormal);
  const planetSamples = 64;
  const plumeSamples = plumeActive ? 24 : 8;
  particles.count = planetSamples + plumeSamples;

  // LivingTerra intact solid crust/mantle particles (bodyId = 0, phaseId = 0 Solid)
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < planetSamples; i++) {
    const y = 1 - (2 * (i + 0.5)) / planetSamples;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const phi = i * goldenAngle;
    const rad = EARTH_RADIUS_KM * 0.95;
    particles.positionsKm[i * 3] = Math.cos(phi) * r * rad;
    particles.positionsKm[i * 3 + 1] = y * rad;
    particles.positionsKm[i * 3 + 2] = Math.sin(phi) * r * rad;
    particles.props[i * 4] = 300;
    particles.props[i * 4 + 1] = 1.0;
    particles.props[i * 4 + 2] = 0.0;
    particles.props[i * 4 + 3] = 250.0;
    particles.meta[i * 4] = 1; // Granite
    particles.meta[i * 4 + 1] = 0; // Solid
    particles.meta[i * 4 + 2] = 0; // bodyId = 0
    particles.meta[i * 4 + 3] = 0.5;
  }

  // Impactor / Vapor Plume particles (bodyId = 1)
  const centerDistKm = plumeActive ? EARTH_RADIUS_KM + plumeAltitudeKm : 25000;
  for (let k = 0; k < plumeSamples; k++) {
    const idx = planetSamples + k;
    const angle = (2 * Math.PI * k) / plumeSamples;
    const jitter = plumeActive ? 180.0 : 400.0;
    const p = new THREE.Vector3()
      .addScaledVector(forward, centerDistKm + (k % 3) * 80)
      .addScaledVector(u, Math.cos(angle) * jitter)
      .addScaledVector(v, Math.sin(angle) * jitter);

    particles.positionsKm[idx * 3] = p.x;
    particles.positionsKm[idx * 3 + 1] = p.y;
    particles.positionsKm[idx * 3 + 2] = p.z;
    particles.props[idx * 4] = plumeActive ? plumeTempK : 240.0;
    particles.props[idx * 4 + 1] = plumeActive ? 60.0 : 0.01;
    particles.props[idx * 4 + 2] = plumeActive ? 1.0 : 0.0;
    particles.props[idx * 4 + 3] = plumeRadiusKm;
    particles.meta[idx * 4] = 1; // Granite
    particles.meta[idx * 4 + 1] = plumeActive ? 2 : 0; // Vapor if active, Solid if approaching
    particles.meta[idx * 4 + 2] = 1; // bodyId = 1
    particles.meta[idx * 4 + 3] = plumeActive ? 16.0 : 12.0;
  }
}

describe('ImpactBiosphereCoupler (AC-20, AC-21)', () => {
  it('initializes with BASELINE_BIOMASS_GTC === 550.0 and valid nominal baseline metrics', () => {
    expect(ImpactBiosphereCoupler.BASELINE_BIOMASS_GTC).toBe(550.0);

    const biomeSystem = new BiomeSystem(42);
    const coupler = new ImpactBiosphereCoupler(biomeSystem, 256);
    const metrics: BiosphereMetrics = coupler.getGlobalMetrics();

    expect(metrics.baselineBiomassGtC).toBe(550.0);
    expect(metrics.totalBiomassGtC).toBeCloseTo(550.0, 1);
    expect(metrics.overallSurvivalPct).toBeGreaterThanOrEqual(98.0);
    expect(metrics.overallSurvivalPct).toBeLessThanOrEqual(100.01);
    expect(metrics.shannonDiversityH).toBeGreaterThan(0);
    expect(Number.isFinite(metrics.shannonDiversityH)).toBe(true);
    expect(metrics.extinctionTier).toBe('Nominal');
    expect(metrics.globalOpticalDepthTau).toBeGreaterThanOrEqual(0);
    expect(Math.abs(metrics.globalDeltaTempK)).toBeLessThan(1.0);

    for (const biome of ALL_BIOMES) {
      expect(metrics.survivalByBiome[biome]).toBeGreaterThanOrEqual(98.0);
      expect(metrics.survivalByBiome[biome]).toBeLessThanOrEqual(100.01);
    }
    for (const guild of ALL_GUILDS) {
      expect(metrics.survivalByGuild[guild]).toBeGreaterThanOrEqual(98.0);
      expect(metrics.survivalByGuild[guild]).toBeLessThanOrEqual(100.01);
    }
  });

  it('enforces exact P/S/blast delay ordering, ~200 m/s tsunami propagation, wildfire ignition, and antipodal horizon occultation (AC-20)', () => {
    const biomeSystem = new BiomeSystem(42);
    const coupler = new ImpactBiosphereCoupler(biomeSystem, 256);
    const gridRes = 16;
    const tracers = createTracerBufferPool(6 * gridRes * gridRes);
    const particles = createParticleBufferPool(128);

    const impactNormal = findLandSiteNormal(biomeSystem);
    const { u: tangentU } = buildOrthonormalBasis(impactNormal);
    const antipodeNormal = impactNormal.clone().negate();

    // Sites at exact spherical angles from impactNormal
    const midArcOcean = findOceanSiteAtAngle(biomeSystem, impactNormal, Math.PI / 3);
    const nearOcean = findOceanSiteAtAngle(biomeSystem, impactNormal, Math.PI / 6);
    const farOcean = findOceanSiteAtAngle(biomeSystem, impactNormal, Math.PI / 2);

    // 1) Pre-impact approaching frame at t = 0 s
    const preImpactorCom: [number, number, number] = [
      impactNormal.x * 25000,
      impactNormal.y * 25000,
      impactNormal.z * 25000,
    ];
    populateSphereTracers(
      tracers,
      gridRes,
      biomeSystem,
      impactNormal,
      2.0, // none in crater
      { tempK: 288, damage: 0, overpressureKpa: 0, tsunamiHeightM: 0, dispKm: 0 },
      { tempK: 288, damage: 0, overpressureKpa: 0, tsunamiHeightM: 0, dispKm: 0 }
    );
    populateParticlesWithPlume(particles, impactNormal, false, 240, 20000, 350);
    coupler.ingestSimulationState(
      makeTelemetry({
        frameSeq: 1,
        simTimeSec: 0,
        outcome: 'Approaching',
        impactorComKm: preImpactorCom,
        maxTempK: 295,
      }),
      tracers,
      particles
    );

    // 2) Impact onset frame at t_imp = 100 s with luminous plume centered 1,400 km above impactNormal
    const impactComKm: [number, number, number] = [
      impactNormal.x * 6800,
      impactNormal.y * 6800,
      impactNormal.z * 6800,
    ];
    populateSphereTracers(
      tracers,
      gridRes,
      biomeSystem,
      impactNormal,
      Math.cos((16 * Math.PI) / 180), // ~16 deg crater cap only; midArc (60 deg) & antipode untouched
      { tempK: 5500, damage: 0.9, overpressureKpa: 320, tsunamiHeightM: 120, dispKm: -30 },
      { tempK: 288, damage: 0, overpressureKpa: 0, tsunamiHeightM: 0, dispKm: 0 }
    );
    populateParticlesWithPlume(particles, impactNormal, true, 14000, 1400, 350);
    coupler.ingestSimulationState(
      makeTelemetry({
        frameSeq: 2,
        simTimeSec: 100,
        outcome: 'Merging',
        impactorComKm: impactComKm,
        kineticEnergyJ: 2.5e31,
        internalEnergyJ: 5.0e30,
        maxTempK: 14000,
        maxPressureGPa: 220,
        boundMassFrac: 0.98,
        diskMassFrac: 0.01,
        ejectaMassFrac: 0.01,
      }),
      tracers,
      particles
    );

    // At theta = pi/3 (60 deg), R = 6371 km:
    // d_surf = 6,671,696 m, d_chord = 6,371,000 m
    // t_P ≈ 100 + 980.2 = 1080.2 s
    // t_S ≈ 100 + 1803.2 = 1903.2 s
    // t_blast ≈ 100 + 19622.6 = 19722.6 s
    const tBeforeP: SitePhysicalExposure = coupler.evaluateSiteExposure(midArcOcean, 200);
    expect(tBeforeP.arcDistanceM).toBeCloseTo(EARTH_RADIUS_KM * 1000 * (Math.PI / 3), -4);
    expect(tBeforeP.chordDistanceM).toBeCloseTo(EARTH_RADIUS_KM * 1000, -4);
    expect(tBeforeP.pWaveArrived).toBe(false);
    expect(tBeforeP.sWaveArrived).toBe(false);
    expect(tBeforeP.blastArrived).toBe(false);

    const tBetweenPAndS = coupler.evaluateSiteExposure(midArcOcean, 1150);
    expect(tBetweenPAndS.pWaveArrived).toBe(true);
    expect(tBetweenPAndS.sWaveArrived).toBe(false);
    expect(tBetweenPAndS.blastArrived).toBe(false);

    const tBetweenSAndBlast = coupler.evaluateSiteExposure(midArcOcean, 1980);
    expect(tBetweenSAndBlast.pWaveArrived).toBe(true);
    expect(tBetweenSAndBlast.sWaveArrived).toBe(true);
    expect(tBetweenSAndBlast.blastArrived).toBe(false);
    expect(tBetweenSAndBlast.pgaG).toBeGreaterThan(0);

    const tAfterBlast = coupler.evaluateSiteExposure(midArcOcean, 19850);
    expect(tAfterBlast.pWaveArrived).toBe(true);
    expect(tAfterBlast.sWaveArrived).toBe(true);
    expect(tAfterBlast.blastArrived).toBe(true);
    expect(tAfterBlast.overpressureKpa).toBeGreaterThan(0);

    // Tsunami propagation at ~200 m/s:
    // At theta = pi/3: t_tsu ≈ 100 + 6,671,696 / 200 = 33,458 s
    const midBeforeTsu = coupler.evaluateSiteExposure(midArcOcean, 5000);
    const midAfterTsu = coupler.evaluateSiteExposure(midArcOcean, 33600);
    expect(midAfterTsu.tsunamiHeightM).toBeGreaterThan(1.0);
    expect(midAfterTsu.tsunamiHeightM).toBeGreaterThan(midBeforeTsu.tsunamiHeightM);

    // At theta = pi/6: t_tsu ≈ 100 + 16,679 = 16,779 s; at theta = pi/2: t_tsu ≈ 100 + 50,038 = 50,138 s
    const nearArrived = coupler.evaluateSiteExposure(nearOcean, 16900);
    const farBeforeArrived = coupler.evaluateSiteExposure(farOcean, 16900);
    const farAfterArrived = coupler.evaluateSiteExposure(farOcean, 50260);
    expect(nearArrived.tsunamiHeightM).toBeGreaterThan(1.0);
    expect(nearArrived.tsunamiHeightM).toBeGreaterThan(farBeforeArrived.tsunamiHeightM);
    expect(farAfterArrived.tsunamiHeightM).toBeGreaterThan(farBeforeArrived.tsunamiHeightM);

    // Horizon occultation & wildfire ignition:
    const zenithFacing = coupler.evaluateSiteExposure(impactNormal, 200);
    expect(zenithFacing.radFluxKwM2).toBeGreaterThan(35);
    expect(zenithFacing.isWildfireIgnited).toBe(true);

    // Orthogonal limb (theta = 90 deg) has plume below local tangent plane -> occulted
    const limb = coupler.evaluateSiteExposure(tangentU, 200);
    expect(limb.radFluxKwM2).toBe(0);

    // Antipode (-impactNormal) strictly occulted -> radFluxKwM2 === 0
    const antipode = coupler.evaluateSiteExposure(antipodeNormal, 200);
    expect(antipode.radFluxKwM2).toBe(0);
  });

  it('tracks global biomass loss, optical depth, impact winter cooling, extinction tier progression, and resetBaseline (AC-21)', () => {
    const biomeSystem = new BiomeSystem(42);
    const coupler = new ImpactBiosphereCoupler(biomeSystem, 256);
    const gridRes = 16;
    const tracers = createTracerBufferPool(6 * gridRes * gridRes);
    const particles = createParticleBufferPool(128);
    const impactNormal = findLandSiteNormal(biomeSystem);

    // Pre-impact frame
    populateSphereTracers(
      tracers,
      gridRes,
      biomeSystem,
      impactNormal,
      2.0,
      { tempK: 288, damage: 0, overpressureKpa: 0, tsunamiHeightM: 0, dispKm: 0 },
      { tempK: 288, damage: 0, overpressureKpa: 0, tsunamiHeightM: 0, dispKm: 0 }
    );
    populateParticlesWithPlume(particles, impactNormal, false, 240, 20000, 350);
    coupler.ingestSimulationState(
      makeTelemetry({
        frameSeq: 1,
        simTimeSec: 0,
        outcome: 'Approaching',
        impactorComKm: [impactNormal.x * 25000, impactNormal.y * 25000, impactNormal.z * 25000],
      }),
      tracers,
      particles
    );

    const preMetrics = coupler.getGlobalMetrics();
    expect(preMetrics.extinctionTier).toBe('Nominal');

    // Cataclysm wave onset: widespread blast, seismic shock, tsunami, thermal flash, and ejecta
    populateSphereTracers(
      tracers,
      gridRes,
      biomeSystem,
      impactNormal,
      -0.3, // large hemisphere + global shock
      { tempK: 2400, damage: 0.85, overpressureKpa: 220, tsunamiHeightM: 140, dispKm: -25 },
      { tempK: 580, damage: 0.35, overpressureKpa: 45, tsunamiHeightM: 50, dispKm: -2 }
    );
    populateParticlesWithPlume(particles, impactNormal, true, 16000, 2500, 500);
    coupler.ingestSimulationState(
      makeTelemetry({
        frameSeq: 2,
        simTimeSec: 100,
        outcome: 'Merging',
        impactorComKm: [impactNormal.x * 6500, impactNormal.y * 6500, impactNormal.z * 6500],
        kineticEnergyJ: 3.0e31,
        internalEnergyJ: 8.0e30,
        maxTempK: 16000,
        maxPressureGPa: 260,
        boundMassFrac: 0.88,
        diskMassFrac: 0.05,
        ejectaMassFrac: 0.07,
      }),
      tracers,
      particles
    );

    // Advance into global blast / ejecta / tsunami arrival & impact winter onset
    populateSphereTracers(
      tracers,
      gridRes,
      biomeSystem,
      impactNormal,
      0.7, // crater rim remains damaged while global surface enters impact winter cooling
      { tempK: 315, damage: 0.9, overpressureKpa: 110, tsunamiHeightM: 120, dispKm: -25 },
      { tempK: 245, damage: 0.45, overpressureKpa: 85, tsunamiHeightM: 60, dispKm: -3 }
    );
    populateParticlesWithPlume(particles, impactNormal, true, 9000, 4500, 600);
    coupler.ingestSimulationState(
      makeTelemetry({
        frameSeq: 3,
        simTimeSec: 120000,
        outcome: 'Merging',
        impactorComKm: [0, 0, 0],
        kineticEnergyJ: 8.0e30,
        internalEnergyJ: 1.2e31,
        maxTempK: 9000,
        maxPressureGPa: 180,
        boundMassFrac: 0.82,
        diskMassFrac: 0.06,
        ejectaMassFrac: 0.12,
      }),
      tracers,
      particles
    );

    const postMetrics = coupler.getGlobalMetrics();
    expect(postMetrics.baselineBiomassGtC).toBe(550.0);
    expect(postMetrics.totalBiomassGtC).toBeLessThan(ImpactBiosphereCoupler.BASELINE_BIOMASS_GTC);
    expect(postMetrics.overallSurvivalPct).toBeLessThan(85.0);
    expect(postMetrics.overallSurvivalPct).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(postMetrics.shannonDiversityH)).toBe(true);
    expect(postMetrics.shannonDiversityH).toBeGreaterThanOrEqual(0);
    expect(postMetrics.globalOpticalDepthTau).toBeGreaterThan(0);
    expect(postMetrics.globalDeltaTempK).toBeLessThan(0);

    const nonNominalTiers: readonly ExtinctionTier[] = [
      'RegionalCrisis',
      'MassExtinction',
      'Sterilization',
    ];
    expect(nonNominalTiers).toContain(postMetrics.extinctionTier);

    for (const biome of ALL_BIOMES) {
      expect(Number.isFinite(postMetrics.survivalByBiome[biome])).toBe(true);
      expect(postMetrics.survivalByBiome[biome]).toBeGreaterThanOrEqual(0);
      expect(postMetrics.survivalByBiome[biome]).toBeLessThanOrEqual(100);
    }
    for (const guild of ALL_GUILDS) {
      expect(Number.isFinite(postMetrics.survivalByGuild[guild])).toBe(true);
      expect(postMetrics.survivalByGuild[guild]).toBeGreaterThanOrEqual(0);
      expect(postMetrics.survivalByGuild[guild]).toBeLessThanOrEqual(100);
    }

    // Resetting baseline restores clean 550.0 GtC Nominal state
    coupler.resetBaseline();
    const resetMetrics = coupler.getGlobalMetrics();
    expect(resetMetrics.extinctionTier).toBe('Nominal');
    expect(resetMetrics.totalBiomassGtC).toBeCloseTo(550.0, 1);
    expect(resetMetrics.overallSurvivalPct).toBeGreaterThanOrEqual(98.0);
    expect(resetMetrics.globalOpticalDepthTau).toBeCloseTo(0, 4);
  });
});
