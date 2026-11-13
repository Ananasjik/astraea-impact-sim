import * as THREE from 'three';
import {
  TRACER_FLAG_DISRUPTED,
  TRACER_FLAG_OCEAN,
  type ParticleBufferPool,
  type TracerBufferPool,
} from '../net/BinaryProtocol';
import type { TelemetrySnapshotDto } from '../net/SimClient';
import { unitNormalToCubeFaceUv } from '../render/OrbitalSurfaceCamera';
import { BiomeSystem, type BiomeId } from './BiomeSystem';
import type { FaunaGuild } from './ProceduralFauna';

export type ExtinctionTier =
  | 'Nominal'
  | 'RegionalCrisis'
  | 'MassExtinction'
  | 'Sterilization';

export interface SitePhysicalExposure {
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

export interface BiosphereMetrics {
  totalBiomassGtC: number;
  baselineBiomassGtC: number;
  shannonDiversityH: number;
  overallSurvivalPct: number;
  globalOpticalDepthTau: number;
  globalDeltaTempK: number;
  survivalByBiome: Record<BiomeId, number>;
  survivalByGuild: Record<FaunaGuild | 'Flora', number>;
  extinctionTier: ExtinctionTier;
}

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
  'Flora',
  'HerbivoreHerd',
  'PursuitPredator',
  'AvianFlock',
  'PelagicMarine',
];

const SIGMA_SB = 5.670374419e-8; // W m^-2 K^-4
const PLUME_EMISSIVITY = 0.85;
const SPATIAL_GRID_RES = 16;
const SPATIAL_GRID_CELLS = 6 * SPATIAL_GRID_RES * SPATIAL_GRID_RES;

function clamp(v: number, minVal: number, maxVal: number): number {
  return v < minVal ? minVal : v > maxVal ? maxVal : v;
}

function normalizeVec3(
  x: number,
  y: number,
  z: number,
  out: THREE.Vector3
): THREE.Vector3 {
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return out.set(0.0, 1.0, 0.0);
  }
  const lenSq = x * x + y * y + z * z;
  if (lenSq <= 1.0e-18) {
    return out.set(0.0, 1.0, 0.0);
  }
  const invLen = 1.0 / Math.sqrt(lenSq);
  return out.set(x * invLen, y * invLen, z * invLen);
}

interface BiosphereTile {
  normal: THREE.Vector3;
  biome: BiomeId;
  elevationM: number;
  climateTempK: number;
  isOcean: boolean;
  baselineBiomass: number;
  guildWeights: [number, number, number, number, number]; // Flora, Herbivore, Predator, Avian, Marine
  guildSurvival: [number, number, number, number, number];
}

function getBiomeBiomassDensity(biome: BiomeId): number {
  switch (biome) {
    case 'TropicalRainforest':
      return 2.6;
    case 'TemperateForest':
      return 1.65;
    case 'BorealTaiga':
      return 1.25;
    case 'Savanna':
      return 0.85;
    case 'OceanReef':
      return 0.22;
    case 'AridSteppe':
      return 0.18;
    case 'AlpineTundra':
    default:
      return 0.1;
  }
}

function getBiomeGuildShares(
  biome: BiomeId,
  elevationM: number
): [number, number, number, number, number] {
  if (biome === 'OceanReef' || elevationM < 0.0) {
    const shallowBoost = elevationM > -180.0 ? 0.35 : 0.12;
    return [shallowBoost, 0.0, 0.0, 0.12, 1.0 - shallowBoost - 0.12];
  }
  // Land: Flora dominates biomass (~84%), fauna cohorts carry ecological guild share (~16%)
  return [0.84, 0.08, 0.03, 0.05, 0.0];
}

export class ImpactBiosphereCoupler {
  static readonly BASELINE_BIOMASS_GTC = 550.0;

  private readonly biomeSystem: BiomeSystem;
  private readonly tiles: BiosphereTile[] = [];
  private readonly tileCount: number;

  // Impact & plume state
  private impactActive = false;
  private impactTimeSec = 0.0;
  private lastSimTimeSec = 0.0;
  private planetRadiusM = 6.371e6;
  private livingComKm: [number, number, number] = [0.0, 0.0, 0.0];
  private impactorComKm: [number, number, number] = [0.0, 0.0, 0.0];
  private impactorMassRatio = 0.0;
  private impactorSepM = 0.0;
  private hasImpactor = false;

  private readonly epicenterNormal = new THREE.Vector3(1.0, 0.0, 0.0);
  private readonly plumeCenterKm = new THREE.Vector3(6371.0 * 1.2, 0.0, 0.0);
  private plumeRadiusM = 1.0e6;
  private plumeTempK = 288.0;
  private impactSeverity = 0.0; // normalized dimensionless impact scale
  private ejectaMassFrac = 0.0;
  private globalOpticalDepthTau = 0.0;
  private globalDeltaTempK = 0.0;
  private globalDisruptedFrac = 0.0;

  // Live cube-sphere tracer lookup (indexed in system COM frame)
  private hasTracers = false;
  private readonly gridCount = new Uint16Array(SPATIAL_GRID_CELLS);
  private readonly gridTempK = new Float32Array(SPATIAL_GRID_CELLS);
  private readonly gridDamage = new Float32Array(SPATIAL_GRID_CELLS);
  private readonly gridOverpKpa = new Float32Array(SPATIAL_GRID_CELLS);
  private readonly gridTsunamiM = new Float32Array(SPATIAL_GRID_CELLS);
  private readonly gridOceanFrac = new Float32Array(SPATIAL_GRID_CELLS);

  // Scratch vectors
  private readonly tmpNorm = new THREE.Vector3();
  private readonly tmpVec = new THREE.Vector3();

  constructor(biomeSystem: BiomeSystem, sampleTileCount = 256) {
    this.biomeSystem = biomeSystem;
    this.tileCount = Math.max(
      16,
      Math.trunc(Number.isFinite(sampleTileCount) ? sampleTileCount : 256)
    );
    this.initFibonacciTiles();
    this.resetBaseline();
  }

  private initFibonacciTiles(): void {
    this.tiles.length = 0;
    const goldenAngle = Math.PI * (3.0 - Math.sqrt(5.0));
    let rawSum = 0.0;

    for (let i = 0; i < this.tileCount; i++) {
      const y = 1.0 - (2.0 * (i + 0.5)) / this.tileCount;
      const r = Math.sqrt(Math.max(0.0, 1.0 - y * y));
      const phi = i * goldenAngle;
      const normal = new THREE.Vector3(
        r * Math.cos(phi),
        y,
        r * Math.sin(phi)
      ).normalize();

      const sample = this.biomeSystem.sampleSurface(normal);
      const rawDensity = getBiomeBiomassDensity(sample.biome);
      const shares = getBiomeGuildShares(sample.biome, sample.elevationM);
      rawSum += rawDensity;

      this.tiles.push({
        normal,
        biome: sample.biome,
        elevationM: sample.elevationM,
        climateTempK: sample.climateTempK,
        isOcean: sample.elevationM < 0.0 || sample.biome === 'OceanReef',
        baselineBiomass: rawDensity,
        guildWeights: shares,
        guildSurvival: [1.0, 1.0, 1.0, 1.0, 1.0],
      });
    }

    const scale =
      rawSum > 0.0 ? ImpactBiosphereCoupler.BASELINE_BIOMASS_GTC / rawSum : 1.0;
    for (let i = 0; i < this.tiles.length; i++) {
      this.tiles[i].baselineBiomass *= scale;
    }
  }

  resetBaseline(): void {
    this.impactActive = false;
    this.impactTimeSec = 0.0;
    this.lastSimTimeSec = 0.0;
    this.impactorMassRatio = 0.0;
    this.impactorSepM = 0.0;
    this.hasImpactor = false;
    this.epicenterNormal.set(1.0, 0.0, 0.0);
    this.plumeCenterKm.set(this.planetRadiusM * 0.00125, 0.0, 0.0);
    this.plumeRadiusM = this.planetRadiusM * 0.15;
    this.plumeTempK = 288.0;
    this.impactSeverity = 0.0;
    this.ejectaMassFrac = 0.0;
    this.globalOpticalDepthTau = 0.0;
    this.globalDeltaTempK = 0.0;
    this.globalDisruptedFrac = 0.0;
    this.hasTracers = false;
    this.gridCount.fill(0);

    for (let i = 0; i < this.tiles.length; i++) {
      const s = this.tiles[i].guildSurvival;
      s[0] = 1.0;
      s[1] = 1.0;
      s[2] = 1.0;
      s[3] = 1.0;
      s[4] = 1.0;
    }
  }

  ingestSimulationState(
    telemetry: TelemetrySnapshotDto,
    tracers: TracerBufferPool,
    particles: ParticleBufferPool
  ): void {
    const simTimeSec = Number.isFinite(telemetry.simTimeSec)
      ? telemetry.simTimeSec
      : 0.0;
    this.lastSimTimeSec = simTimeSec;

    const rKm =
      Number.isFinite(telemetry.livingBodyRadiusKm) &&
      telemetry.livingBodyRadiusKm > 10.0
        ? telemetry.livingBodyRadiusKm
        : 6371.0;
    this.planetRadiusM = rKm * 1000.0;

    const lCom = telemetry.livingBodyComKm;
    this.livingComKm = [
      Number.isFinite(lCom[0]) ? lCom[0] : 0.0,
      Number.isFinite(lCom[1]) ? lCom[1] : 0.0,
      Number.isFinite(lCom[2]) ? lCom[2] : 0.0,
    ];

    const iCom = telemetry.impactorComKm;
    const iComX = Number.isFinite(iCom[0]) ? iCom[0] : 0.0;
    const iComY = Number.isFinite(iCom[1]) ? iCom[1] : 0.0;
    const iComZ = Number.isFinite(iCom[2]) ? iCom[2] : 0.0;
    this.impactorComKm = [iComX, iComY, iComZ];

    const sepDxKm = iComX - this.livingComKm[0];
    const sepDyKm = iComY - this.livingComKm[1];
    const sepDzKm = iComZ - this.livingComKm[2];
    const sepKm = Math.sqrt(
      sepDxKm * sepDxKm + sepDyKm * sepDyKm + sepDzKm * sepDzKm
    );
    this.hasImpactor = sepKm > 1.0;
    this.impactorSepM = sepKm * 1000.0;

    // Bin live tracers into cube-sphere lookup and compute shock weighted direction
    this.gridCount.fill(0);
    this.gridTempK.fill(0.0);
    this.gridDamage.fill(0.0);
    this.gridOverpKpa.fill(0.0);
    this.gridTsunamiM.fill(0.0);
    this.gridOceanFrac.fill(0.0);

    let tracerShockWeightSum = 0.0;
    let tracerShockX = 0.0;
    let tracerShockY = 0.0;
    let tracerShockZ = 0.0;
    let tracerPeakTemp = 0.0;
    let tracerPeakDam = 0.0;
    let tracerPeakOverp = 0.0;
    let tracerPeakTsunami = 0.0;
    let disruptedCount = 0;

    const tCount = Math.min(tracers.count, tracers.capacity);
    this.hasTracers = tCount > 0;

    for (let i = 0; i < tCount; i++) {
      const i3 = i * 3;
      const dx = tracers.positionsKm[i3] - this.livingComKm[0];
      const dy = tracers.positionsKm[i3 + 1] - this.livingComKm[1];
      const dz = tracers.positionsKm[i3 + 2] - this.livingComKm[2];
      const lenSq = dx * dx + dy * dy + dz * dz;
      if (lenSq <= 1.0e-9) {
        continue;
      }
      const invLen = 1.0 / Math.sqrt(lenSq);
      const nx = dx * invLen;
      const ny = dy * invLen;
      const nz = dz * invLen;

      const uv = unitNormalToCubeFaceUv(nx, ny, nz);
      const gx = Math.min(
        SPATIAL_GRID_RES - 1,
        Math.max(0, Math.floor(uv.u * SPATIAL_GRID_RES))
      );
      const gy = Math.min(
        SPATIAL_GRID_RES - 1,
        Math.max(0, Math.floor(uv.v * SPATIAL_GRID_RES))
      );
      const cellIdx =
        uv.face * SPATIAL_GRID_RES * SPATIAL_GRID_RES +
        gy * SPATIAL_GRID_RES +
        gx;

      const tempK = Number.isFinite(tracers.tempK[i]) ? tracers.tempK[i] : 288.0;
      const dam = Number.isFinite(tracers.damage[i]) ? tracers.damage[i] : 0.0;
      const overp = Number.isFinite(tracers.overpressureKpa[i])
        ? tracers.overpressureKpa[i]
        : 0.0;
      const tsu = Number.isFinite(tracers.tsunamiHeightM[i])
        ? tracers.tsunamiHeightM[i]
        : 0.0;
      const flag = tracers.flags[i];
      const isOcean = (flag & TRACER_FLAG_OCEAN) !== 0 ? 1.0 : 0.0;
      if ((flag & TRACER_FLAG_DISRUPTED) !== 0 || dam >= 0.85) {
        disruptedCount += 1;
      }

      this.gridCount[cellIdx] += 1;
      this.gridTempK[cellIdx] += tempK;
      this.gridDamage[cellIdx] += dam;
      this.gridOverpKpa[cellIdx] += overp;
      this.gridTsunamiM[cellIdx] += tsu;
      this.gridOceanFrac[cellIdx] += isOcean;

      if (tempK > tracerPeakTemp) tracerPeakTemp = tempK;
      if (dam > tracerPeakDam) tracerPeakDam = dam;
      if (overp > tracerPeakOverp) tracerPeakOverp = overp;
      if (tsu > tracerPeakTsunami) tracerPeakTsunami = tsu;

      const wShock =
        Math.max(0.0, tempK - 360.0) * 0.5 + dam * 1500.0 + overp * 10.0;
      if (wShock > 1.0) {
        tracerShockWeightSum += wShock;
        tracerShockX += nx * wShock;
        tracerShockY += ny * wShock;
        tracerShockZ += nz * wShock;
      }
    }

    for (let c = 0; c < SPATIAL_GRID_CELLS; c++) {
      const cnt = this.gridCount[c];
      if (cnt > 1) {
        const invC = 1.0 / cnt;
        this.gridTempK[c] *= invC;
        this.gridDamage[c] *= invC;
        this.gridOverpKpa[c] *= invC;
        this.gridTsunamiM[c] *= invC;
        this.gridOceanFrac[c] *= invC;
      }
    }

    this.globalDisruptedFrac = tCount > 0 ? disruptedCount / tCount : 0.0;

    // Scan particles for body mass ratio, impactor/plume thermal centroid & luminous extent
    const pCount = Math.min(particles.count, particles.capacity);
    let targetParticles = 0;
    let impactorParticles = 0;
    let hotWeightSum = 0.0;
    let hotX = 0.0;
    let hotY = 0.0;
    let hotZ = 0.0;
    let hotTempWeightedSum = 0.0;
    let hotCount = 0;
    let maxExteriorTempK = 288.0;
    let vaporMoltenCount = 0;
    const isApproachingFar =
      telemetry.outcome === 'Approaching' && (!this.hasImpactor || sepKm > 1.35 * rKm);

    for (let i = 0; i < pCount; i++) {
      const i3 = i * 3;
      const i4 = i * 4;
      const bodyId = Math.round(particles.meta[i4 + 2]);
      const phaseId = Math.round(particles.meta[i4 + 1]);
      if (bodyId === 0) {
        targetParticles += 1;
      } else {
        impactorParticles += 1;
      }

      const px = particles.positionsKm[i3];
      const py = particles.positionsKm[i3 + 1];
      const pz = particles.positionsKm[i3 + 2];
      const dxLiving = px - this.livingComKm[0];
      const dyLiving = py - this.livingComKm[1];
      const dzLiving = pz - this.livingComKm[2];
      const rFromLivingKm = Math.sqrt(
        dxLiving * dxLiving + dyLiving * dyLiving + dzLiving * dzLiving
      );
      const damage01 = particles.props[i4 + 2];

      // Exclude intact planetary interiors (e.g., 5500 K LivingTerra core or approaching impactor core)
      const isExteriorOrPlume =
        phaseId === 3 ||
        (bodyId === 0
          ? rFromLivingKm >= 0.88 * rKm
          : !isApproachingFar || damage01 >= 0.15);

      if (isExteriorOrPlume && (phaseId === 1 || phaseId === 2 || phaseId === 3)) {
        vaporMoltenCount += 1;
      }

      const tempK = particles.props[i4];
      if (isExteriorOrPlume && tempK > maxExteriorTempK) {
        maxExteriorTempK = tempK;
      }
      if (isExteriorOrPlume && tempK > 700.0) {
        hotCount += 1;
        const w = Math.pow(clamp((tempK - 600.0) / 1000.0, 0.1, 20.0), 1.5);
        hotWeightSum += w;
        hotX += px * w;
        hotY += py * w;
        hotZ += pz * w;
        hotTempWeightedSum += tempK * w;
      }
    }

    if (targetParticles > 0 && impactorParticles > 0) {
      this.impactorMassRatio = clamp(
        impactorParticles / targetParticles,
        0.001,
        1.5
      );
    } else if (this.hasImpactor) {
      this.impactorMassRatio = 0.025;
    } else {
      this.impactorMassRatio = 0.0;
    }

    const telemetryMaxTempK = Number.isFinite(telemetry.maxTempK)
      ? telemetry.maxTempK
      : 288.0;
    const effectivePeakTempK =
      pCount > 0 || this.hasTracers
        ? Math.max(maxExteriorTempK, tracerPeakTemp)
        : Math.max(telemetryMaxTempK, tracerPeakTemp);
    const ejectaFrac = clamp(
      Number.isFinite(telemetry.ejectaMassFrac) ? telemetry.ejectaMassFrac : 0.0,
      0.0,
      1.0
    );
    const diskFrac = clamp(
      Number.isFinite(telemetry.diskMassFrac) ? telemetry.diskMassFrac : 0.0,
      0.0,
      1.0
    );
    const qRatio = Math.max(
      0.0,
      Number.isFinite(telemetry.qRatioQrd) ? telemetry.qRatioQrd : 0.0
    );
    this.ejectaMassFrac = ejectaFrac;

    // Detect impact onset cleanly without false-triggering on intact planet interiors or settling jitter
    const impactContact =
      this.hasImpactor && sepKm <= 1.28 * rKm;
    const preContactNoiseFloor = isApproachingFar;
    const debrisTriggered =
      !preContactNoiseFloor && (ejectaFrac > 0.002 || diskFrac > 0.002);
    const thermalOrShockTrigger =
      tracerPeakTemp > (preContactNoiseFloor ? 900.0 : 650.0) ||
      hotCount > 0 ||
      (!this.hasTracers && pCount === 0 && !this.hasImpactor && telemetryMaxTempK > 650.0) ||
      tracerPeakDam > 0.05 ||
      tracerPeakOverp > (preContactNoiseFloor ? 120.0 : 1.0) ||
      tracerPeakTsunami > 8.0 ||
      debrisTriggered ||
      telemetry.outcome !== 'Approaching';

    if (!this.impactActive && (impactContact || thermalOrShockTrigger)) {
      this.impactActive = true;
      this.impactTimeSec = simTimeSec;
    }

    if (this.impactActive) {
      // Determine authoritative epicenter normal
      if (tracerShockWeightSum > 10.0) {
        normalizeVec3(
          tracerShockX,
          tracerShockY,
          tracerShockZ,
          this.epicenterNormal
        );
      } else if (hotWeightSum > 0.0) {
        normalizeVec3(
          hotX / hotWeightSum - this.livingComKm[0],
          hotY / hotWeightSum - this.livingComKm[1],
          hotZ / hotWeightSum - this.livingComKm[2],
          this.epicenterNormal
        );
      } else if (this.hasImpactor) {
        normalizeVec3(sepDxKm, sepDyKm, sepDzKm, this.epicenterNormal);
      }

      // Determine luminous plume centroid & radius
      if (hotWeightSum > 0.0) {
        const cx = hotX / hotWeightSum;
        const cy = hotY / hotWeightSum;
        const cz = hotZ / hotWeightSum;
        // Compute spread around hot centroid using the same exterior/plume filter
        let rSpreadSq = 0.0;
        for (let i = 0; i < pCount; i++) {
          const i4 = i * 4;
          const bodyId = Math.round(particles.meta[i4 + 2]);
          const phaseId = Math.round(particles.meta[i4 + 1]);
          const i3 = i * 3;
          const px = particles.positionsKm[i3];
          const py = particles.positionsKm[i3 + 1];
          const pz = particles.positionsKm[i3 + 2];
          const dxLiving = px - this.livingComKm[0];
          const dyLiving = py - this.livingComKm[1];
          const dzLiving = pz - this.livingComKm[2];
          const rFromLivingKm = Math.sqrt(
            dxLiving * dxLiving + dyLiving * dyLiving + dzLiving * dzLiving
          );
          const isExteriorOrPlume =
            phaseId === 2 ||
            phaseId === 3 ||
            (bodyId === 0
              ? rFromLivingKm >= 0.88 * rKm
              : !isApproachingFar || particles.props[i4 + 2] >= 0.15);
          if (isExteriorOrPlume && particles.props[i4] > 700.0) {
            const dx = px - cx;
            const dy = py - cy;
            const dz = pz - cz;
            rSpreadSq += dx * dx + dy * dy + dz * dz;
          }
        }
        const rmsKm = hotCount > 0 ? Math.sqrt(rSpreadSq / hotCount) : rKm * 0.15;
        this.plumeRadiusM = clamp(rmsKm * 1000.0, this.planetRadiusM * 0.08, this.planetRadiusM * 0.65);

        // Ensure plume altitude lifts at least 0.22 * R_p above livingCom along impact hemisphere
        const fromComX = cx - this.livingComKm[0];
        const fromComY = cy - this.livingComKm[1];
        const fromComZ = cz - this.livingComKm[2];
        const plumeDistKm = Math.sqrt(
          fromComX * fromComX + fromComY * fromComY + fromComZ * fromComZ
        );
        const minPlumeR = rKm * 1.24;
        if (plumeDistKm < minPlumeR) {
          this.plumeCenterKm.set(
            this.livingComKm[0] + this.epicenterNormal.x * minPlumeR,
            this.livingComKm[1] + this.epicenterNormal.y * minPlumeR,
            this.livingComKm[2] + this.epicenterNormal.z * minPlumeR
          );
        } else {
          this.plumeCenterKm.set(cx, cy, cz);
        }

        const meanHotT = hotTempWeightedSum / hotWeightSum;
        this.plumeTempK = Math.max(meanHotT, 0.72 * effectivePeakTempK, 1800.0);
      } else {
        const plumeRKm = rKm * 1.25;
        this.plumeCenterKm.set(
          this.livingComKm[0] + this.epicenterNormal.x * plumeRKm,
          this.livingComKm[1] + this.epicenterNormal.y * plumeRKm,
          this.livingComKm[2] + this.epicenterNormal.z * plumeRKm
        );
        this.plumeRadiusM = this.planetRadiusM * 0.18;
        this.plumeTempK = Math.max(effectivePeakTempK, 2400.0);
      }

      // Dimensionless severity combining temperature, ejecta, Q/Q_RD, disruption, and outcome
      const tempFactor = clamp((effectivePeakTempK - 500.0) / 4500.0, 0.05, 2.5);
      const energyFactor = clamp(
        ejectaFrac * 18.0 + diskFrac * 8.0 + qRatio * 1.5,
        0.15,
        3.5
      );
      const outcomeBoost =
        telemetry.outcome === 'CatastrophicDisruption'
          ? 3.0
          : telemetry.outcome === 'HitAndRun'
            ? 1.1
            : telemetry.outcome === 'Merging' || telemetry.outcome === 'GrazeAndMerge'
              ? 1.6
              : 0.8;
      this.impactSeverity = Math.max(
        this.impactSeverity,
        clamp(0.45 * tempFactor + 0.45 * energyFactor + 0.25 * outcomeBoost, 0.25, 4.0)
      );

      // Stratospheric optical depth tau & impact winter Delta T
      const tElapsed = Math.max(0.0, simTimeSec - this.impactTimeSec);
      const globalSpread = 1.0 - Math.exp(-(tElapsed + 60.0) / 450.0);
      const vaporFrac = pCount > 0 ? vaporMoltenCount / pCount : 0.0;
      const tauRaw =
        (ejectaFrac * 45.0 +
          diskFrac * 18.0 +
          vaporFrac * 12.0 +
          this.impactSeverity * 2.2) *
        globalSpread;
      this.globalOpticalDepthTau = Math.max(
        this.globalOpticalDepthTau * 0.99,
        clamp(tauRaw, 0.0, 25.0)
      );
      this.globalDeltaTempK =
        -38.0 * (1.0 - Math.exp(-0.42 * this.globalOpticalDepthTau));
    }

    this.updateGlobalTiles(simTimeSec);
  }

  evaluateSiteExposure(
    unitNormal: THREE.Vector3,
    simTimeSec: number
  ): SitePhysicalExposure {
    const siteN = normalizeVec3(
      unitNormal.x,
      unitNormal.y,
      unitNormal.z,
      this.tmpNorm.clone()
    );
    const tSec = Number.isFinite(simTimeSec) ? simTimeSec : this.lastSimTimeSec;
    const rP = this.planetRadiusM;

    const dotImp = clamp(siteN.dot(this.epicenterNormal), -1.0, 1.0);
    const theta = Math.acos(dotImp);
    const arcDistanceM = rP * theta;
    const chordDistanceM = 2.0 * rP * Math.sin(0.5 * theta);

    const surfaceSample = this.biomeSystem.sampleSurface(siteN);
    const tracerSample = this.sampleSpatialTracer(siteN);
    const isOceanSite =
      tracerSample.valid && tracerSample.isOcean
        ? true
        : surfaceSample.elevationM < 0.0;

    // Pre-impact Roche tidal bulge on ocean sites
    let rocheTideM = 0.0;
    if (
      this.hasImpactor &&
      this.impactorMassRatio > 0.0 &&
      isOceanSite &&
      this.impactorSepM > 0.0
    ) {
      const impDir = normalizeVec3(
        this.impactorComKm[0] - this.livingComKm[0],
        this.impactorComKm[1] - this.livingComKm[1],
        this.impactorComKm[2] - this.livingComKm[2],
        this.tmpVec
      );
      const cosPsi = clamp(siteN.dot(impDir), -1.0, 1.0);
      const rSepEff = Math.max(this.impactorSepM, rP * 1.15);
      const tide =
        0.6 *
        this.impactorMassRatio *
        Math.pow(rP / rSepEff, 3.0) *
        rP *
        0.5 *
        (3.0 * cosPsi * cosPsi - 1.0);
      rocheTideM = Math.max(0.0, tide);
    }

    if (!this.impactActive || tSec < this.impactTimeSec) {
      return {
        siteNormal: siteN,
        arcDistanceM,
        chordDistanceM,
        tsunamiHeightM: Math.max(rocheTideM, tracerSample.tsunamiHeightM),
        pWaveArrived: false,
        sWaveArrived: false,
        pgaG: 0.0,
        radFluxKwM2: 0.0,
        isWildfireIgnited: false,
        ejectaArrived: false,
        ejectaFluxKgM2: 0.0,
        blastArrived: false,
        overpressureKpa: 0.0,
        effectiveTempK: surfaceSample.climateTempK,
        crustDamage: tracerSample.damage,
      };
    }

    // Strict causal arrival times
    const tP = this.impactTimeSec + chordDistanceM / 6500.0;
    const tS = this.impactTimeSec + arcDistanceM / 3700.0;
    const tEj = this.impactTimeSec + 2400.0 * Math.sin(0.5 * theta);
    const tBlast = this.impactTimeSec + arcDistanceM / 340.0;
    const tTsu = this.impactTimeSec + arcDistanceM / 200.0;

    const pWaveArrived = tSec >= tP;
    const sWaveArrived = tSec >= tS;
    const ejectaArrived = tSec >= tEj;
    const blastArrived = tSec >= tBlast;
    const tsuArrived = tSec >= tTsu;

    // 1. Seismic PGA (g)
    const sinThetaEff = Math.max(0.08, Math.sin(theta));
    const antipodalFocus = 1.0 + 0.85 * Math.exp(-Math.pow((Math.PI - theta) / 0.25, 2.0));
    const geomAtten = (1.0 / Math.pow(sinThetaEff, 0.55)) * antipodalFocus;

    let wavePgaG = 0.0;
    if (pWaveArrived) {
      const dtP = tSec - tP;
      const pEnv = Math.exp(-dtP / 140.0) * (1.0 - Math.exp(-dtP / 8.0 + 0.05));
      wavePgaG += 0.28 * this.impactSeverity * geomAtten * Math.max(0.15, pEnv);
    }
    if (sWaveArrived) {
      const dtS = tSec - tS;
      const sEnv = Math.exp(-dtS / 260.0) * (1.0 - Math.exp(-dtS / 12.0 + 0.05));
      wavePgaG += 1.45 * this.impactSeverity * geomAtten * Math.max(0.25, sEnv);
    }
    const tracerPgaG =
      tracerSample.valid && tracerSample.damage > 0.02
        ? tracerSample.damage * 1.5 + tracerSample.overpressureKpa * 0.01
        : 0.0;
    const pgaG = pWaveArrived ? Math.max(wavePgaG, tracerPgaG) : tracerPgaG;

    // 2. Thermal Radiation Flash & Horizon Occultation
    const rSiteKmX = this.livingComKm[0] + siteN.x * (rP * 0.001);
    const rSiteKmY = this.livingComKm[1] + siteN.y * (rP * 0.001);
    const rSiteKmZ = this.livingComKm[2] + siteN.z * (rP * 0.001);

    const losX = (this.plumeCenterKm.x - rSiteKmX) * 1000.0;
    const losY = (this.plumeCenterKm.y - rSiteKmY) * 1000.0;
    const losZ = (this.plumeCenterKm.z - rSiteKmZ) * 1000.0;
    const horizonDot = siteN.x * losX + siteN.y * losY + siteN.z * losZ;

    let radFluxKwM2 = 0.0;
    if (horizonDot > 0.0 && this.plumeTempK > 450.0) {
      const distSqM = Math.max(
        losX * losX + losY * losY + losZ * losZ,
        this.plumeRadiusM * this.plumeRadiusM
      );
      const tElapsed = Math.max(0.0, tSec - this.impactTimeSec);
      const flashDecay = Math.exp(-tElapsed / 1800.0);
      const viewFactor = (this.plumeRadiusM * this.plumeRadiusM) / distSqM;
      const optTrans = Math.exp(-0.18 * this.globalOpticalDepthTau);
      const qWm2 =
        PLUME_EMISSIVITY *
        SIGMA_SB *
        Math.pow(this.plumeTempK, 4.0) *
        viewFactor *
        optTrans *
        Math.max(0.15, flashDecay);
      radFluxKwM2 = qWm2 * 0.001;
    }

    // 3. Ballistic Ejecta Re-Entry Loading & Thermal Pulse
    let ejectaFluxKgM2 = 0.0;
    if (ejectaArrived) {
      const dtEj = tSec - tEj;
      const ejEnv = Math.exp(-dtEj / 900.0) * (1.0 - Math.exp(-dtEj / 25.0 + 0.1));
      const ejectaBlanket =
        (85.0 * Math.exp(-theta / 0.55) + 18.0 + 180.0 * this.ejectaMassFrac) *
        this.impactSeverity;
      ejectaFluxKgM2 = Math.max(0.0, ejectaBlanket * Math.max(0.1, ejEnv));
    }

    // 4. Atmospheric Blast Overpressure (Sach wave + tracer overpressure)
    let overpressureKpa = tracerSample.valid ? tracerSample.overpressureKpa : 0.0;
    if (blastArrived) {
      const dtBlast = tSec - tBlast;
      const blastDecay = Math.exp(-dtBlast / 500.0);
      const angularSpread = Math.max(0.12, Math.sin(theta * 0.5));
      const sachOverpKpa =
        (95.0 * this.impactSeverity * Math.max(0.2, blastDecay)) /
        Math.pow(angularSpread, 1.35);
      overpressureKpa = Math.max(overpressureKpa, sachOverpKpa);
    }

    // 5. Tsunami Propagation & Run-up Attenuation
    let tsunamiHeightM = Math.max(
      rocheTideM,
      tracerSample.valid ? tracerSample.tsunamiHeightM : 0.0
    );
    if (tsuArrived) {
      const dtTsu = tSec - tTsu;
      const tsuEnv = Math.exp(-dtTsu / 2400.0);
      const h0 = 260.0 * this.impactSeverity;
      const rawWaveM =
        (h0 / Math.sqrt(Math.max(0.05, Math.sin(theta)))) *
        Math.max(0.2, tsuEnv);
      const elevAtten =
        surfaceSample.elevationM >= 0.0
          ? Math.exp(-surfaceSample.elevationM / 120.0)
          : 1.0;
      const propagatedTsuM =
        surfaceSample.elevationM > 450.0 ? 0.0 : rawWaveM * elevAtten;
      tsunamiHeightM = Math.max(tsunamiHeightM, propagatedTsuM);
    }

    // 6. Effective Local Temperature & Wildfire Ignition
    const radHeatK = Math.min(900.0, radFluxKwM2 * 4.5);
    const ejectaHeatK = ejectaArrived ? Math.min(350.0, ejectaFluxKgM2 * 2.8) : 0.0;
    const blastHeatK = blastArrived ? Math.min(260.0, overpressureKpa * 1.4) : 0.0;
    const analyticalTempK = Math.max(
      180.0,
      surfaceSample.climateTempK +
        this.globalDeltaTempK +
        radHeatK +
        ejectaHeatK +
        blastHeatK
    );
    const effectiveTempK =
      tracerSample.valid && tracerSample.tempK > 340.0
        ? Math.max(analyticalTempK, tracerSample.tempK)
        : analyticalTempK;

    const isWildfireIgnited =
      !isOceanSite && (radFluxKwM2 > 35.0 || effectiveTempK > 573.0);

    // 7. Crustal Damage
    const craterMask = Math.exp(-Math.pow(theta / (0.18 * Math.max(0.6, this.impactSeverity)), 2.0));
    const analyticalDamage = clamp(
      craterMask * 0.98 + (pgaG > 2.2 ? clamp((pgaG - 2.2) * 0.15, 0.0, 0.4) : 0.0),
      0.0,
      1.0
    );
    const crustDamage = clamp(
      Math.max(tracerSample.valid ? tracerSample.damage : 0.0, analyticalDamage),
      0.0,
      1.0
    );

    return {
      siteNormal: siteN,
      arcDistanceM,
      chordDistanceM,
      tsunamiHeightM,
      pWaveArrived,
      sWaveArrived,
      pgaG,
      radFluxKwM2,
      isWildfireIgnited,
      ejectaArrived,
      ejectaFluxKgM2,
      blastArrived,
      overpressureKpa,
      effectiveTempK,
      crustDamage,
    };
  }

  getGlobalMetrics(): BiosphereMetrics {
    let totalBiomassGtC = 0.0;

    const biomeBase = new Map<BiomeId, number>();
    const biomeSurv = new Map<BiomeId, number>();
    for (const b of ALL_BIOMES) {
      biomeBase.set(b, 0.0);
      biomeSurv.set(b, 0.0);
    }

    const guildBase = new Float64Array(ALL_GUILDS.length);
    const guildSurv = new Float64Array(ALL_GUILDS.length);

    for (let i = 0; i < this.tiles.length; i++) {
      const tile = this.tiles[i];
      let tileBio = 0.0;
      for (let g = 0; g < ALL_GUILDS.length; g++) {
        const baseG = tile.baselineBiomass * tile.guildWeights[g];
        const survG = baseG * tile.guildSurvival[g];
        guildBase[g] += baseG;
        guildSurv[g] += survG;
        tileBio += survG;
      }
      totalBiomassGtC += tileBio;
      biomeBase.set(tile.biome, (biomeBase.get(tile.biome) ?? 0.0) + tile.baselineBiomass);
      biomeSurv.set(tile.biome, (biomeSurv.get(tile.biome) ?? 0.0) + tileBio);
    }

    const overallSurvivalPct = clamp(
      (totalBiomassGtC / ImpactBiosphereCoupler.BASELINE_BIOMASS_GTC) * 100.0,
      0.0,
      100.0
    );

    let shannonDiversityH = 0.0;
    if (totalBiomassGtC > 1.0e-9) {
      // Aggregate surviving biomass per (biome, guild) functional group for canonical Shannon index
      const groupTotals = new Float64Array(ALL_BIOMES.length * ALL_GUILDS.length);
      for (let i = 0; i < this.tiles.length; i++) {
        const tile = this.tiles[i];
        const bIdx = ALL_BIOMES.indexOf(tile.biome);
        for (let g = 0; g < ALL_GUILDS.length; g++) {
          groupTotals[bIdx * ALL_GUILDS.length + g] +=
            tile.baselineBiomass * tile.guildWeights[g] * tile.guildSurvival[g];
        }
      }
      for (let k = 0; k < groupTotals.length; k++) {
        if (groupTotals[k] > 1.0e-9) {
          const p = groupTotals[k] / totalBiomassGtC;
          shannonDiversityH -= p * Math.log(p);
        }
      }
    }

    const survivalByBiome = {} as Record<BiomeId, number>;
    for (const b of ALL_BIOMES) {
      const base = biomeBase.get(b) ?? 0.0;
      const surv = biomeSurv.get(b) ?? 0.0;
      survivalByBiome[b] =
        base > 1.0e-9 ? clamp((surv / base) * 100.0, 0.0, 100.0) : overallSurvivalPct;
    }

    const survivalByGuild = {} as Record<FaunaGuild | 'Flora', number>;
    for (let g = 0; g < ALL_GUILDS.length; g++) {
      const base = guildBase[g];
      const surv = guildSurv[g];
      survivalByGuild[ALL_GUILDS[g]] =
        base > 1.0e-9 ? clamp((surv / base) * 100.0, 0.0, 100.0) : overallSurvivalPct;
    }

    let extinctionTier: ExtinctionTier;
    if (overallSurvivalPct < 0.5) {
      extinctionTier = 'Sterilization';
    } else if (overallSurvivalPct < 25.0) {
      extinctionTier = 'MassExtinction';
    } else if (overallSurvivalPct < 85.0) {
      extinctionTier = 'RegionalCrisis';
    } else {
      extinctionTier = 'Nominal';
    }

    return {
      totalBiomassGtC,
      baselineBiomassGtC: ImpactBiosphereCoupler.BASELINE_BIOMASS_GTC,
      shannonDiversityH: Math.max(0.0, shannonDiversityH),
      overallSurvivalPct,
      globalOpticalDepthTau: this.globalOpticalDepthTau,
      globalDeltaTempK: this.globalDeltaTempK,
      survivalByBiome,
      survivalByGuild,
      extinctionTier,
    };
  }

  private sampleSpatialTracer(siteNormal: THREE.Vector3): {
    valid: boolean;
    tempK: number;
    damage: number;
    overpressureKpa: number;
    tsunamiHeightM: number;
    isOcean: boolean;
  } {
    if (!this.hasTracers) {
      return {
        valid: false,
        tempK: 288.0,
        damage: 0.0,
        overpressureKpa: 0.0,
        tsunamiHeightM: 0.0,
        isOcean: false,
      };
    }

    const uv = unitNormalToCubeFaceUv(siteNormal.x, siteNormal.y, siteNormal.z);
    const gx = Math.min(
      SPATIAL_GRID_RES - 1,
      Math.max(0, Math.floor(uv.u * SPATIAL_GRID_RES))
    );
    const gy = Math.min(
      SPATIAL_GRID_RES - 1,
      Math.max(0, Math.floor(uv.v * SPATIAL_GRID_RES))
    );
    const cellIdx =
      uv.face * SPATIAL_GRID_RES * SPATIAL_GRID_RES +
      gy * SPATIAL_GRID_RES +
      gx;

    if (this.gridCount[cellIdx] > 0) {
      return {
        valid: true,
        tempK: this.gridTempK[cellIdx],
        damage: this.gridDamage[cellIdx],
        overpressureKpa: this.gridOverpKpa[cellIdx],
        tsunamiHeightM: this.gridTsunamiM[cellIdx],
        isOcean: this.gridOceanFrac[cellIdx] >= 0.5,
      };
    }

    // Small 1-ring fallback around missing/disrupted tracer bin
    let sumTemp = 0.0;
    let sumDam = 0.0;
    let sumOverp = 0.0;
    let sumTsu = 0.0;
    let sumOcean = 0.0;
    let hits = 0;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = clamp(gx + dx, 0, SPATIAL_GRID_RES - 1);
        const ny = clamp(gy + dy, 0, SPATIAL_GRID_RES - 1);
        const idx =
          uv.face * SPATIAL_GRID_RES * SPATIAL_GRID_RES +
          ny * SPATIAL_GRID_RES +
          nx;
        if (this.gridCount[idx] > 0) {
          sumTemp += this.gridTempK[idx];
          sumDam += this.gridDamage[idx];
          sumOverp += this.gridOverpKpa[idx];
          sumTsu += this.gridTsunamiM[idx];
          sumOcean += this.gridOceanFrac[idx];
          hits += 1;
        }
      }
    }

    if (hits > 0) {
      const inv = 1.0 / hits;
      return {
        valid: true,
        tempK: sumTemp * inv,
        damage: sumDam * inv,
        overpressureKpa: sumOverp * inv,
        tsunamiHeightM: sumTsu * inv,
        isOcean: sumOcean * inv >= 0.5,
      };
    }

    return {
      valid: false,
      tempK: 288.0,
      damage: 0.0,
      overpressureKpa: 0.0,
      tsunamiHeightM: 0.0,
      isOcean: false,
    };
  }

  private updateGlobalTiles(simTimeSec: number): void {
    if (!this.impactActive && this.globalDisruptedFrac < 0.01) {
      return;
    }

    // Global climate / aerosol photosynthesis stress from optical depth & cooling
    const winterFreezeStress =
      this.globalDeltaTempK < -6.0
        ? clamp((-this.globalDeltaTempK - 6.0) / 28.0, 0.0, 0.92)
        : 0.0;
    const opticalDarknessStress =
      this.globalOpticalDepthTau > 1.2
        ? clamp((this.globalOpticalDepthTau - 1.2) / 9.0, 0.0, 0.92)
        : 0.0;
    const disruptionStress =
      this.globalDisruptedFrac > 0.35
        ? clamp((this.globalDisruptedFrac - 0.35) / 0.55, 0.0, 1.0)
        : 0.0;

    for (let i = 0; i < this.tiles.length; i++) {
      const tile = this.tiles[i];
      const exp = this.evaluateSiteExposure(tile.normal, simTimeSec);

      const thermalMortality = clamp((exp.effectiveTempK - 335.0) / 110.0, 0.0, 1.0);
      const blastMortality = clamp((exp.overpressureKpa - 12.0) / 70.0, 0.0, 1.0);
      const seismicMortality = clamp((exp.pgaG - 0.6) / 2.2, 0.0, 1.0);
      const ejectaMortality = clamp((exp.ejectaFluxKgM2 - 15.0) / 45.0, 0.0, 1.0);
      const damageMortality = clamp((exp.crustDamage - 0.25) / 0.5, 0.0, 1.0);
      const tsunamiLandMortality = !tile.isOcean
        ? clamp((exp.tsunamiHeightM - 10.0) / 20.0, 0.0, 1.0)
        : 0.0;
      const wildfireMortality = exp.isWildfireIgnited ? 0.85 : 0.0;

      // Flora (guild 0)
      const floraKill = Math.max(
        thermalMortality,
        blastMortality,
        ejectaMortality,
        damageMortality,
        tsunamiLandMortality,
        wildfireMortality,
        opticalDarknessStress * 0.72,
        winterFreezeStress * 0.65,
        disruptionStress
      );
      // Herbivores (guild 1)
      const herbKill = Math.max(
        thermalMortality,
        blastMortality,
        seismicMortality * 0.85,
        ejectaMortality,
        damageMortality,
        tsunamiLandMortality,
        wildfireMortality * 0.75,
        winterFreezeStress * 0.7,
        opticalDarknessStress * 0.55,
        disruptionStress
      );
      // Predators (guild 2) - slightly higher starvation amplification
      const predKill = Math.max(
        herbKill,
        winterFreezeStress * 0.8,
        opticalDarknessStress * 0.65,
        disruptionStress
      );
      // Avian (guild 3) - flies above small tsunamis, vulnerable to blast/thermal/ejecta/winter
      const avianKill = Math.max(
        thermalMortality,
        blastMortality,
        ejectaMortality * 1.1,
        damageMortality * 0.85,
        wildfireMortality * 0.55,
        winterFreezeStress * 0.72,
        opticalDarknessStress * 0.6,
        disruptionStress
      );
      // PelagicMarine (guild 4) - shielded from wildfire & tsunami height; hit by heat, crust loss, ocean darkening
      const marineThermal = clamp((exp.effectiveTempK - 365.0) / 140.0, 0.0, 1.0);
      const marineKill = Math.max(
        marineThermal,
        damageMortality,
        blastMortality * 0.55,
        ejectaMortality * 0.45,
        opticalDarknessStress * 0.58,
        winterFreezeStress * 0.42,
        disruptionStress
      );

      const kills = [floraKill, herbKill, predKill, avianKill, marineKill];
      for (let g = 0; g < ALL_GUILDS.length; g++) {
        const nextSurv = clamp(1.0 - kills[g], 0.0, 1.0);
        if (nextSurv < tile.guildSurvival[g]) {
          tile.guildSurvival[g] = nextSurv;
        }
      }
    }
  }
}
