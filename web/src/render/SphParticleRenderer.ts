import * as THREE from 'three';
import type { ParticleBufferPool } from '../net/BinaryProtocol';

export type VisMode =
  | 'NaturalBlackbody'
  | 'TemperatureK'
  | 'PressureGPa'
  | 'MaterialLayer'
  | 'DamageFracture'
  | 'Velocity';

export const VIS_MODE_TO_INDEX: Record<VisMode, number> = {
  NaturalBlackbody: 0,
  TemperatureK: 1,
  PressureGPa: 2,
  MaterialLayer: 3,
  DamageFracture: 4,
  Velocity: 5,
};

const VERTEX_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>

attribute vec4 aProp; // [tempK, pressureGPa, damage01, radiusKm]
attribute vec4 aMeta; // [materialId, phaseId, bodyId, speedKmS]

uniform float uViewportHeight;

varying vec3 vWorldPosKm;
varying vec4 vProp;
varying vec4 vMeta;

void main() {
  vWorldPosKm = position;
  vProp = aProp;
  vMeta = aMeta;

  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;

  float radiusKm = max(aProp.w, 1.0);
  float projDiameterPx =
    (uViewportHeight * projectionMatrix[1][1] * radiusKm * 1.4) /
    max(gl_Position.w, 1.0e-4);
  gl_PointSize = clamp(projDiameterPx, 1.5, 48.0);

  #include <logdepthbuf_vertex>
}
`;

const FRAGMENT_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>

// VisMode indices:
// 0 = NaturalBlackbody
// 1 = TemperatureK
// 2 = PressureGPa
// 3 = MaterialLayer
// 4 = DamageFracture
// 5 = Velocity

uniform int uVisMode;
uniform float uSliceEnabled;
uniform vec3 uSliceNormal;
uniform float uSliceOffsetKm;
uniform vec3 uSliceComKm;
uniform float uLivingLodActive;

varying vec3 vWorldPosKm;
varying vec4 vProp;
varying vec4 vMeta;

vec3 materialAlbedo(float matId) {
  if (matId < 0.5) {
    // Iron
    return vec3(0.56, 0.59, 0.64);
  } else if (matId < 1.5) {
    // Granite
    return vec3(0.43, 0.36, 0.31);
  } else if (matId < 2.5) {
    // WaterIce
    return vec3(0.14, 0.45, 0.78);
  } else if (matId < 3.5) {
    // HydrogenHelium
    return vec3(0.76, 0.68, 0.54);
  }
  // Regolith
  return vec3(0.52, 0.49, 0.44);
}

vec3 materialLayerPalette(float matId) {
  if (matId < 0.5) {
    // Iron core: metallic nickel-gold
    return vec3(0.88, 0.75, 0.40);
  } else if (matId < 1.5) {
    // Granite mantle: silicate rust-crimson
    return vec3(0.80, 0.30, 0.18);
  } else if (matId < 2.5) {
    // WaterIce: cerulean blue
    return vec3(0.15, 0.62, 0.96);
  } else if (matId < 3.5) {
    // HydrogenHelium: violet-cyan
    return vec3(0.64, 0.42, 0.95);
  }
  // Regolith: warm tan
  return vec3(0.78, 0.68, 0.48);
}

vec3 blackbodyColor(float tempK) {
  float t = clamp(tempK, 800.0, 25000.0) / 100.0;
  float r;
  float g;
  float b;

  if (t <= 66.0) {
    r = 1.0;
    g = clamp(0.39008158 * log(max(t, 1.0)) - 0.63184144, 0.0, 1.0);
  } else {
    r = clamp(1.29293619 * pow(t - 60.0, -0.13320476), 0.0, 1.0);
    g = clamp(1.12989086 * pow(t - 60.0, -0.07551485), 0.0, 1.0);
  }

  if (t >= 66.0) {
    b = 1.0;
  } else if (t <= 19.0) {
    b = 0.0;
  } else {
    b = clamp(0.54320679 * log(max(t - 10.0, 1.0)) - 1.19625409, 0.0, 1.0);
  }

  // Deep cherry red transition below 1600 K
  float lowBlend = smoothstep(850.0, 1600.0, tempK);
  vec3 cherry = vec3(0.85, 0.08, 0.02);
  return mix(cherry, vec3(r, g, b), lowBlend);
}

vec3 infernoColormap(float x) {
  float t = clamp(x, 0.0, 1.0);
  vec3 c0 = vec3(0.02, 0.02, 0.08);
  vec3 c1 = vec3(0.32, 0.07, 0.48);
  vec3 c2 = vec3(0.72, 0.21, 0.47);
  vec3 c3 = vec3(0.99, 0.54, 0.28);
  vec3 c4 = vec3(0.99, 0.99, 0.75);
  if (t < 0.25) return mix(c0, c1, t * 4.0);
  if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
  if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
  return mix(c3, c4, (t - 0.75) * 4.0);
}

vec3 viridisColormap(float x) {
  float t = clamp(x, 0.0, 1.0);
  vec3 c0 = vec3(0.27, 0.01, 0.33);
  vec3 c1 = vec3(0.23, 0.32, 0.55);
  vec3 c2 = vec3(0.13, 0.57, 0.55);
  vec3 c3 = vec3(0.37, 0.79, 0.38);
  vec3 c4 = vec3(0.99, 0.91, 0.15);
  if (t < 0.25) return mix(c0, c1, t * 4.0);
  if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
  if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
  return mix(c3, c4, (t - 0.75) * 4.0);
}

void main() {
  #include <logdepthbuf_fragment>

  if (uSliceEnabled > 0.5 && dot(vWorldPosKm - uSliceComKm, uSliceNormal) > uSliceOffsetKm) {
    discard;
  }

  float tempK = vProp.x;
  float pressureGPa = vProp.y;
  float damage01 = vProp.z;
  float matId = vMeta.x;
  float phaseId = vMeta.y;
  float bodyId = vMeta.z;
  float speedKmS = vMeta.w;

  // Suppress intact cool outer LivingTerra spheres behind active CubeSphereLod unless sliced
  if (
    uLivingLodActive > 0.5 &&
    uSliceEnabled < 0.5 &&
    bodyId < 0.5 &&
    damage01 < 0.55 &&
    tempK < 700.0 &&
    (phaseId < 0.5 || (phaseId > 3.5 && phaseId < 4.5))
  ) {
    discard;
  }

  vec2 coord = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(coord, coord);
  if (r2 > 1.0) {
    discard;
  }

  float sphereShade = 0.68 + 0.32 * sqrt(max(0.0, 1.0 - r2));
  float splatFalloff = exp(-2.1 * r2) * (1.0 - r2 * r2);

  vec3 color = vec3(1.0);

  if (uVisMode == 0) {
    // NaturalBlackbody
    vec3 albedo = materialAlbedo(matId) * sphereShade;
    float glowFrac = smoothstep(850.0, 1800.0, tempK);
    vec3 bb = blackbodyColor(tempK);
    float boost = 1.0 + 0.65 * smoothstep(3500.0, 18000.0, tempK);
    color = mix(albedo, bb * boost, glowFrac);
  } else if (uVisMode == 1) {
    // TemperatureK
    float logT = (log(max(tempK, 10.0)) - log(10.0)) / log(20000.0);
    color = infernoColormap(logT) * (0.85 + 0.15 * sphereShade);
  } else if (uVisMode == 2) {
    // PressureGPa
    float logP = pressureGPa <= 1.0e-3
      ? 0.0
      : (log(max(pressureGPa, 1.0e-3)) - log(1.0e-3)) / log(1.0e7);
    color = viridisColormap(logP) * (0.85 + 0.15 * sphereShade);
  } else if (uVisMode == 3) {
    // MaterialLayer
    color = materialLayerPalette(matId) * sphereShade;
  } else if (uVisMode == 4) {
    // DamageFracture
    float d = clamp(damage01, 0.0, 1.0);
    vec3 intactCol = vec3(0.18, 0.24, 0.30);
    vec3 stressCol = vec3(0.92, 0.58, 0.14);
    vec3 failCol = vec3(1.00, 0.14, 0.08);
    color = d < 0.5
      ? mix(intactCol, stressCol, d * 2.0)
      : mix(stressCol, failCol, (d - 0.5) * 2.0);
    color *= (0.8 + 0.2 * sphereShade);
  } else {
    // Velocity (uVisMode == 5)
    float logV = (log(max(speedKmS, 0.05)) - log(0.05)) / log(2000.0);
    vec3 slowCol = vec3(0.10, 0.32, 0.78);
    vec3 midCol = vec3(0.14, 0.78, 0.64);
    vec3 fastCol = vec3(1.00, 0.95, 0.55);
    float v = clamp(logV, 0.0, 1.0);
    color = v < 0.5
      ? mix(slowCol, midCol, v * 2.0)
      : mix(midCol, fastCol, (v - 0.5) * 2.0);
    color *= (0.82 + 0.18 * sphereShade);
  }

  float alpha = clamp(splatFalloff * 1.15, 0.0, 0.96);
  gl_FragColor = vec4(color, alpha);
}
`;

export class SphParticleRenderer {
  readonly mesh: THREE.Points;
  readonly maxParticles: number;

  private readonly geometry: THREE.BufferGeometry;
  private readonly material: THREE.ShaderMaterial;
  private readonly positionAttr: THREE.BufferAttribute;
  private readonly propAttr: THREE.BufferAttribute;
  private readonly metaAttr: THREE.BufferAttribute;

  private readonly posArray: Float32Array;
  private readonly propArray: Float32Array;
  private readonly metaArray: Float32Array;

  private readonly sliceComWorldKm: [number, number, number] = [0.0, 0.0, 0.0];
  private readonly lastCameraPosKm: [number, number, number] = [0.0, 0.0, 0.0];

  constructor(maxParticles: number) {
    if (!Number.isInteger(maxParticles) || maxParticles < 0) {
      throw new Error(`Invalid maxParticles: ${maxParticles}`);
    }
    this.maxParticles = maxParticles;

    this.posArray = new Float32Array(maxParticles * 3);
    this.propArray = new Float32Array(maxParticles * 4);
    this.metaArray = new Float32Array(maxParticles * 4);

    this.positionAttr = new THREE.BufferAttribute(this.posArray, 3);
    this.positionAttr.setUsage(THREE.DynamicDrawUsage);
    this.propAttr = new THREE.BufferAttribute(this.propArray, 4);
    this.propAttr.setUsage(THREE.DynamicDrawUsage);
    this.metaAttr = new THREE.BufferAttribute(this.metaArray, 4);
    this.metaAttr.setUsage(THREE.DynamicDrawUsage);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', this.positionAttr);
    this.geometry.setAttribute('aProp', this.propAttr);
    this.geometry.setAttribute('aMeta', this.metaAttr);
    this.geometry.setDrawRange(0, 0);

    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      uniforms: {
        uVisMode: { value: 0 },
        uSliceEnabled: { value: 0.0 },
        uSliceNormal: { value: new THREE.Vector3(0.0, 1.0, 0.0) },
        uSliceOffsetKm: { value: 0.0 },
        uSliceComKm: { value: new THREE.Vector3(0.0, 0.0, 0.0) },
        uViewportHeight: { value: 900.0 },
        uLivingLodActive: { value: 0.0 },
      },
    });

    this.mesh = new THREE.Points(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;
  }

  getUniforms(): Record<string, THREE.IUniform<unknown>> {
    return this.material.uniforms;
  }

  getShaderSources(): { vertexShader: string; fragmentShader: string } {
    return {
      vertexShader: this.material.vertexShader,
      fragmentShader: this.material.fragmentShader,
    };
  }

  setViewportHeight(heightPx: number): void {
    if (Number.isFinite(heightPx) && heightPx > 0.0) {
      this.material.uniforms.uViewportHeight.value = heightPx;
    }
  }

  setLivingLodActive(active: boolean): void {
    this.material.uniforms.uLivingLodActive.value = active ? 1.0 : 0.0;
  }

  setVisMode(mode: VisMode): void {
    this.material.uniforms.uVisMode.value = VIS_MODE_TO_INDEX[mode];
  }

  setSlicePlane(
    enabled: boolean,
    normal: THREE.Vector3,
    offsetKm: number,
    comKm: [number, number, number]
  ): void {
    this.material.uniforms.uSliceEnabled.value = enabled ? 1.0 : 0.0;

    const n = this.material.uniforms.uSliceNormal.value as THREE.Vector3;
    n.copy(normal);
    if (n.lengthSq() > 1.0e-12) {
      n.normalize();
    } else {
      n.set(0.0, 1.0, 0.0);
    }

    this.material.uniforms.uSliceOffsetKm.value = offsetKm;
    this.sliceComWorldKm[0] = comKm[0];
    this.sliceComWorldKm[1] = comKm[1];
    this.sliceComWorldKm[2] = comKm[2];

    const uCom = this.material.uniforms.uSliceComKm.value as THREE.Vector3;
    uCom.set(
      comKm[0] - this.lastCameraPosKm[0],
      comKm[1] - this.lastCameraPosKm[1],
      comKm[2] - this.lastCameraPosKm[2]
    );
  }

  updateFromPool(
    pool: ParticleBufferPool,
    cameraPosKm: [number, number, number]
  ): void {
    const camX = cameraPosKm[0];
    const camY = cameraPosKm[1];
    const camZ = cameraPosKm[2];

    this.lastCameraPosKm[0] = camX;
    this.lastCameraPosKm[1] = camY;
    this.lastCameraPosKm[2] = camZ;

    const uCom = this.material.uniforms.uSliceComKm.value as THREE.Vector3;
    uCom.set(
      this.sliceComWorldKm[0] - camX,
      this.sliceComWorldKm[1] - camY,
      this.sliceComWorldKm[2] - camZ
    );

    const n = Math.min(pool.count, this.maxParticles);
    const srcPos = pool.positionsKm;
    const srcProps = pool.props;
    const srcMeta = pool.meta;

    const dstPos = this.posArray;
    const dstProps = this.propArray;
    const dstMeta = this.metaArray;

    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      dstPos[i3] = srcPos[i3] - camX;
      dstPos[i3 + 1] = srcPos[i3 + 1] - camY;
      dstPos[i3 + 2] = srcPos[i3 + 2] - camZ;

      const i4 = i * 4;
      dstProps[i4] = srcProps[i4];
      dstProps[i4 + 1] = srcProps[i4 + 1];
      dstProps[i4 + 2] = srcProps[i4 + 2];
      dstProps[i4 + 3] = srcProps[i4 + 3];

      dstMeta[i4] = srcMeta[i4];
      dstMeta[i4 + 1] = srcMeta[i4 + 1];
      dstMeta[i4 + 2] = srcMeta[i4 + 2];
      dstMeta[i4 + 3] = srcMeta[i4 + 3];
    }

    this.positionAttr.clearUpdateRanges();
    this.propAttr.clearUpdateRanges();
    this.metaAttr.clearUpdateRanges();
    if (n > 0) {
      this.positionAttr.addUpdateRange(0, n * 3);
      this.propAttr.addUpdateRange(0, n * 4);
      this.metaAttr.addUpdateRange(0, n * 4);
    }

    this.positionAttr.needsUpdate = true;
    this.propAttr.needsUpdate = true;
    this.metaAttr.needsUpdate = true;
    this.geometry.setDrawRange(0, n);
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
