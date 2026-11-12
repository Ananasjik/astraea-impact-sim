import * as THREE from 'three';

export type BiomeId =
  | 'TropicalRainforest'
  | 'Savanna'
  | 'TemperateForest'
  | 'BorealTaiga'
  | 'AlpineTundra'
  | 'AridSteppe'
  | 'OceanReef';

export interface SurfaceSample {
  elevationM: number;
  latitudeRad: number;
  moisture01: number;
  climateTempK: number;
  biome: BiomeId;
  albedoRgb: [number, number, number];
}

const ABS_LAT_BOREAL_RAD = (55.0 * Math.PI) / 180.0;

// 12 3D Perlin gradient directions (unit-scaled integer triplets)
const GRAD3: ReadonlyArray<readonly [number, number, number]> = [
  [1, 1, 0],
  [-1, 1, 0],
  [1, -1, 0],
  [-1, -1, 0],
  [1, 0, 1],
  [-1, 0, 1],
  [1, 0, -1],
  [-1, 0, -1],
  [0, 1, 1],
  [0, -1, 1],
  [0, 1, -1],
  [0, -1, -1],
];

function clamp(v: number, minVal: number, maxVal: number): number {
  return v < minVal ? minVal : v > maxVal ? maxVal : v;
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
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

function normalizeComponents(
  nx: number,
  ny: number,
  nz: number
): [number, number, number] {
  if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz)) {
    return [0.0, 1.0, 0.0];
  }
  const lenSq = nx * nx + ny * ny + nz * nz;
  if (lenSq <= 1.0e-18) {
    return [0.0, 1.0, 0.0];
  }
  const invLen = 1.0 / Math.sqrt(lenSq);
  return [nx * invLen, ny * invLen, nz * invLen];
}

export class BiomeSystem {
  readonly seed: number;
  private readonly perm: Uint8Array;

  constructor(seed = 42) {
    this.seed = Number.isFinite(seed) ? Math.trunc(seed) : 42;
    const rng = createMulberry32(this.seed);
    const base = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
      base[i] = i;
    }
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = base[i];
      base[i] = base[j];
      base[j] = tmp;
    }
    this.perm = new Uint8Array(512);
    for (let i = 0; i < 512; i++) {
      this.perm[i] = base[i & 255];
    }
  }

  static classifyBiome(
    elevationM: number,
    latitudeRad: number,
    moisture01: number,
    climateTempK: number
  ): BiomeId {
    if (elevationM < 0.0) {
      return 'OceanReef';
    }
    if (elevationM >= 3200.0 || climateTempK < 268.0) {
      return 'AlpineTundra';
    }
    if (Math.abs(latitudeRad) > ABS_LAT_BOREAL_RAD) {
      return moisture01 >= 0.3 ? 'BorealTaiga' : 'AlpineTundra';
    }
    if (climateTempK >= 293.0) {
      if (moisture01 >= 0.62) {
        return 'TropicalRainforest';
      }
      if (moisture01 >= 0.28) {
        return 'Savanna';
      }
      return 'AridSteppe';
    }
    return moisture01 >= 0.4 ? 'TemperateForest' : 'AridSteppe';
  }

  getElevationM(unitNormal: THREE.Vector3): number {
    const [nx, ny, nz] = normalizeComponents(
      unitNormal.x,
      unitNormal.y,
      unitNormal.z
    );
    return this.computeElevationM(nx, ny, nz);
  }

  sampleSurface(unitNormal: THREE.Vector3): SurfaceSample {
    const [nx, ny, nz] = normalizeComponents(
      unitNormal.x,
      unitNormal.y,
      unitNormal.z
    );

    const elevationM = this.computeElevationM(nx, ny, nz);
    const latitudeRad = Math.asin(clamp(ny, -1.0, 1.0));
    const absLat = Math.abs(latitudeRad);
    const cosLat = Math.cos(latitudeRad);

    // Sea-level planetary insolation + regional climate perturbation + tropospheric lapse rate
    const tempNoise = this.fbm3(
      nx * 2.1 + 11.3,
      ny * 2.1 - 7.1,
      nz * 2.1 + 3.7,
      3
    );
    const seaLevelTempK =
      248.0 + 57.0 * Math.pow(Math.max(0.0, cosLat), 0.72) + 4.5 * tempNoise;
    const lapseRateK = Math.max(0.0, elevationM) * 0.0065;
    const climateTempK = clamp(seaLevelTempK - lapseRateK, 240.0, 312.0);

    // Atmospheric moisture: ITCZ equatorial belt, subtropical dry belt, mid-lat storm tracks, orographic rainshadow
    let moisture01: number;
    if (elevationM < 0.0) {
      moisture01 = 1.0;
    } else {
      const itcz = 0.36 * Math.exp(-Math.pow(absLat / 0.28, 2.0));
      const subtropicalDry =
        -0.3 * Math.exp(-Math.pow((absLat - 0.45) / 0.18, 2.0));
      const stormTrack =
        0.22 * Math.exp(-Math.pow((absLat - 0.91) / 0.24, 2.0));
      const coastalBoost = 0.1 * Math.exp(-elevationM / 1400.0);
      const altitudeShadow = -0.18 * clamp(elevationM / 4500.0, 0.0, 1.0);
      const moistNoise = this.fbm3(
        nx * 2.4 - 5.2,
        ny * 2.4 + 9.1,
        nz * 2.4 - 4.4,
        4
      );
      moisture01 = clamp(
        0.48 +
          itcz +
          subtropicalDry +
          stormTrack +
          coastalBoost +
          altitudeShadow +
          0.42 * moistNoise,
        0.0,
        1.0
      );
    }

    const biome = BiomeSystem.classifyBiome(
      elevationM,
      latitudeRad,
      moisture01,
      climateTempK
    );

    const detailNoise = this.fbm3(
      nx * 7.5 + 2.1,
      ny * 7.5 - 3.4,
      nz * 7.5 + 5.7,
      2
    );
    const albedoRgb = this.computeBiomeAlbedo(
      biome,
      elevationM,
      moisture01,
      climateTempK,
      detailNoise
    );

    return {
      elevationM,
      latitudeRad,
      moisture01,
      climateTempK,
      biome,
      albedoRgb,
    };
  }

  private computeElevationM(nx: number, ny: number, nz: number): number {
    // Domain-warped tectonic continental plates
    const warpX = this.fbm3(nx * 1.4 + 4.2, ny * 1.4 - 1.9, nz * 1.4 + 7.5, 2);
    const warpY = this.fbm3(nx * 1.4 - 6.1, ny * 1.4 + 3.7, nz * 1.4 - 2.3, 2);
    const warpZ = this.fbm3(nx * 1.4 + 2.8, ny * 1.4 + 8.4, nz * 1.4 - 5.9, 2);

    const wx = nx + 0.26 * warpX;
    const wy = ny + 0.26 * warpY;
    const wz = nz + 0.26 * warpZ;

    const continent = this.fbm3(wx * 1.65, wy * 1.65, wz * 1.65, 4);
    const rawCont = continent * 1.18 - 0.03;

    const ridgeSample = this.ridgedFbm3(
      wx * 3.4 + 13.7,
      wy * 3.4 - 8.2,
      wz * 3.4 + 4.9,
      4
    );
    const microDetail = this.fbm3(
      nx * 11.0 - 3.1,
      ny * 11.0 + 6.4,
      nz * 11.0 - 9.2,
      3
    );

    if (rawCont < 0.0) {
      // Continental shelf transition [-150..0 m] into abyssal basins down to ~ -6000 m
      const shelfT = clamp(-rawCont / 0.06, 0.0, 1.0);
      const deepT = clamp((-rawCont - 0.05) / 0.42, 0.0, 1.0);
      const shelfDepth = -140.0 * shelfT;
      const basinDepth = -5850.0 * Math.pow(deepT, 1.15);
      return clamp(
        shelfDepth + basinDepth + microDetail * (40.0 + 240.0 * deepT),
        -6200.0,
        -0.5
      );
    }

    const landT = clamp(rawCont / 0.48, 0.0, 1.0);
    const baseLand = Math.pow(landT, 1.05) * 2500.0;
    const orogenyMask = clamp(rawCont * 3.0, 0.0, 1.0);
    const ridgeHeight =
      Math.pow(Math.max(0.0, ridgeSample - 0.22) / 0.78, 1.35) *
      4300.0 *
      orogenyMask;

    return clamp(baseLand + ridgeHeight + microDetail * 210.0, 0.0, 6500.0);
  }

  private computeBiomeAlbedo(
    biome: BiomeId,
    elevationM: number,
    moisture01: number,
    climateTempK: number,
    detail: number
  ): [number, number, number] {
    const d = detail * 0.04;
    let r = 0.2;
    let g = 0.4;
    let b = 0.2;

    switch (biome) {
      case 'OceanReef': {
        if (elevationM > -160.0) {
          const shallow = 1.0 - clamp(-elevationM / 160.0, 0.0, 1.0);
          r = lerp(0.04, 0.11, shallow) + d * 0.4;
          g = lerp(0.24, 0.52, shallow) + d * 0.5;
          b = lerp(0.46, 0.62, shallow) + d * 0.5;
        } else {
          const abyss = clamp((-elevationM - 160.0) / 5500.0, 0.0, 1.0);
          r = lerp(0.04, 0.015, abyss) + d * 0.2;
          g = lerp(0.21, 0.08, abyss) + d * 0.3;
          b = lerp(0.44, 0.24, abyss) + d * 0.4;
        }
        break;
      }
      case 'TropicalRainforest': {
        r = 0.07 + 0.04 * (1.0 - moisture01) + d;
        g = 0.28 + 0.09 * moisture01 + d;
        b = 0.08 + 0.03 * moisture01 + d * 0.5;
        break;
      }
      case 'Savanna': {
        const lush = clamp((moisture01 - 0.28) / 0.34, 0.0, 1.0);
        r = lerp(0.56, 0.38, lush) + d;
        g = lerp(0.46, 0.44, lush) + d;
        b = lerp(0.2, 0.16, lush) + d * 0.5;
        break;
      }
      case 'TemperateForest': {
        r = 0.14 + 0.05 * (1.0 - moisture01) + d;
        g = 0.34 + 0.09 * moisture01 + d;
        b = 0.13 + 0.04 * moisture01 + d * 0.5;
        break;
      }
      case 'BorealTaiga': {
        r = 0.1 + 0.04 * (1.0 - moisture01) + d;
        g = 0.23 + 0.06 * moisture01 + d;
        b = 0.15 + 0.04 * moisture01 + d * 0.5;
        break;
      }
      case 'AlpineTundra': {
        const snowLine = clamp(
          Math.max(
            (270.0 - climateTempK) / 10.0,
            (elevationM - 3400.0) / 1200.0
          ),
          0.0,
          1.0
        );
        r = lerp(0.36, 0.84, snowLine) + d;
        g = lerp(0.34, 0.88, snowLine) + d;
        b = lerp(0.31, 0.92, snowLine) + d;
        break;
      }
      case 'AridSteppe':
      default: {
        const scrub = clamp(moisture01 / 0.4, 0.0, 1.0);
        r = lerp(0.72, 0.58, scrub) + d;
        g = lerp(0.56, 0.48, scrub) + d;
        b = lerp(0.35, 0.29, scrub) + d * 0.7;
        break;
      }
    }

    return [clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0)];
  }

  private noise3(x: number, y: number, z: number): number {
    const X = Math.floor(x) & 255;
    const Y = Math.floor(y) & 255;
    const Z = Math.floor(z) & 255;

    const fx = x - Math.floor(x);
    const fy = y - Math.floor(y);
    const fz = z - Math.floor(z);

    const u = fade(fx);
    const v = fade(fy);
    const w = fade(fz);

    const p = this.perm;
    const A = p[X] + Y;
    const AA = p[A] + Z;
    const AB = p[A + 1] + Z;
    const B = p[X + 1] + Y;
    const BA = p[B] + Z;
    const BB = p[B + 1] + Z;

    const g000 = GRAD3[p[AA] % 12];
    const g100 = GRAD3[p[BA] % 12];
    const g010 = GRAD3[p[AB] % 12];
    const g110 = GRAD3[p[BB] % 12];
    const g001 = GRAD3[p[AA + 1] % 12];
    const g101 = GRAD3[p[BA + 1] % 12];
    const g011 = GRAD3[p[AB + 1] % 12];
    const g111 = GRAD3[p[BB + 1] % 12];

    const n000 = g000[0] * fx + g000[1] * fy + g000[2] * fz;
    const n100 = g100[0] * (fx - 1.0) + g100[1] * fy + g100[2] * fz;
    const n010 = g010[0] * fx + g010[1] * (fy - 1.0) + g010[2] * fz;
    const n110 = g110[0] * (fx - 1.0) + g110[1] * (fy - 1.0) + g110[2] * fz;
    const n001 = g001[0] * fx + g001[1] * fy + g001[2] * (fz - 1.0);
    const n101 = g101[0] * (fx - 1.0) + g101[1] * fy + g101[2] * (fz - 1.0);
    const n011 = g011[0] * fx + g011[1] * (fy - 1.0) + g011[2] * (fz - 1.0);
    const n111 =
      g111[0] * (fx - 1.0) + g111[1] * (fy - 1.0) + g111[2] * (fz - 1.0);

    return lerp(
      lerp(lerp(n000, n100, u), lerp(n010, n110, u), v),
      lerp(lerp(n001, n101, u), lerp(n011, n111, u), v),
      w
    );
  }

  private fbm3(x: number, y: number, z: number, octaves: number): number {
    let sum = 0.0;
    let amp = 0.5;
    let freq = 1.0;
    let norm = 0.0;
    for (let i = 0; i < octaves; i++) {
      sum += amp * this.noise3(x * freq, y * freq, z * freq);
      norm += amp;
      freq *= 2.07;
      amp *= 0.5;
    }
    return norm > 0.0 ? sum / norm : 0.0;
  }

  private ridgedFbm3(x: number, y: number, z: number, octaves: number): number {
    let sum = 0.0;
    let amp = 0.5;
    let freq = 1.0;
    let norm = 0.0;
    for (let i = 0; i < octaves; i++) {
      const n = 1.0 - Math.abs(this.noise3(x * freq, y * freq, z * freq));
      sum += amp * n * n;
      norm += amp;
      freq *= 2.11;
      amp *= 0.5;
    }
    return norm > 0.0 ? sum / norm : 0.0;
  }
}
