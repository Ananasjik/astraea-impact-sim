import * as THREE from 'three';

const ATMOSPHERE_VERTEX_SHADER = /* glsl */ `
varying vec3 vWorldRelKm;

void main() {
  vec4 worldRel = modelMatrix * vec4(position, 1.0);
  vWorldRelKm = worldRel.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldRel;
}
`;

const ATMOSPHERE_FRAGMENT_SHADER = /* glsl */ `
uniform vec3 uPlanetCenterRelKm;
uniform float uPlanetRadiusKm;
uniform float uAtmosphereRadiusKm;
uniform vec3 uSunDir;
uniform float uOpticalDepthTau;
uniform float uPlumeGlowTempK;

varying vec3 vWorldRelKm;

vec2 raySphereIntersect(vec3 ro, vec3 rd, vec3 center, float radius) {
  vec3 oc = ro - center;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - radius * radius;
  float disc = b * b - c;
  if (disc < 0.0) {
    return vec2(-1.0);
  }
  float s = sqrt(disc);
  return vec2(-b - s, -b + s);
}

vec3 plumeBlackbody(float tempK) {
  float t = clamp((tempK - 800.0) / 5000.0, 0.0, 1.0);
  vec3 redOrange = vec3(1.0, 0.34, 0.06);
  vec3 warmWhite = vec3(1.0, 0.90, 0.72);
  return mix(redOrange, warmWhite, t);
}

void main() {
  vec3 rd = normalize(vWorldRelKm);
  vec3 ro = vec3(0.0);

  vec2 atmHit = raySphereIntersect(ro, rd, uPlanetCenterRelKm, uAtmosphereRadiusKm);
  if (atmHit.y <= 0.0) {
    discard;
  }

  float tStart = max(atmHit.x, 0.0);
  float tEnd = atmHit.y;

  vec2 planetHit = raySphereIntersect(ro, rd, uPlanetCenterRelKm, uPlanetRadiusKm);
  bool hitsPlanet = planetHit.x > 0.0;
  if (hitsPlanet) {
    tEnd = min(tEnd, planetHit.x);
  }

  float pathLen = max(0.0, tEnd - tStart);
  if (pathLen <= 0.0) {
    discard;
  }

  float shellThickness = max(uAtmosphereRadiusKm - uPlanetRadiusKm, 1.0);
  const int SAMPLES = 6;
  float stepLen = pathLen / float(SAMPLES);

  float rayleighAccum = 0.0;
  float mieAccum = 0.0;
  float sunWeightAccum = 0.0;

  for (int i = 0; i < SAMPLES; i++) {
    float t = tStart + (float(i) + 0.5) * stepLen;
    vec3 samplePos = ro + rd * t;
    vec3 fromCenter = samplePos - uPlanetCenterRelKm;
    float alt = max(0.0, length(fromCenter) - uPlanetRadiusKm);
    vec3 upNorm = normalize(fromCenter);

    float rayleighDensity = exp(-alt / (shellThickness * 0.28));
    float mieDensity = exp(-alt / (shellThickness * 0.14));

    float sunDot = clamp(dot(upNorm, uSunDir) * 0.5 + 0.5, 0.0, 1.0);
    float daySide = smoothstep(0.15, 0.85, sunDot);

    rayleighAccum += rayleighDensity * stepLen;
    mieAccum += mieDensity * stepLen;
    sunWeightAccum += rayleighDensity * daySide * stepLen;
  }

  float opticalNorm = rayleighAccum / max(shellThickness * 3.5, 1.0);
  float mieNorm = mieAccum / max(shellThickness * 2.5, 1.0);
  float dayFrac = sunWeightAccum / max(rayleighAccum, 1.0e-5);

  float cosTheta = dot(rd, uSunDir);
  float rayleighPhase = 0.75 * (1.0 + cosTheta * cosTheta);
  float g = 0.76;
  float g2 = g * g;
  float miePhase = (1.0 - g2) / (4.0 * 3.14159265 * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 0.05), 1.5));

  vec3 rayleighColor = vec3(0.24, 0.56, 1.00) * rayleighPhase * opticalNorm * (0.12 + 1.25 * dayFrac);
  vec3 mieColor = vec3(1.00, 0.82, 0.58) * min(miePhase, 3.0) * mieNorm * (0.05 + 0.75 * dayFrac);

  // Impact winter aerosol darkening & dusty soot haze
  float tau = max(0.0, uOpticalDepthTau);
  float hazeTransmittance = exp(-0.45 * tau);
  float dustScatter = 1.0 - exp(-0.35 * tau);
  vec3 sootHaze = vec3(0.14, 0.10, 0.08) * dustScatter * clamp(opticalNorm * 1.6, 0.0, 1.0);

  // Incandescent impact plume scatter
  float plumeIntensity = smoothstep(650.0, 4200.0, uPlumeGlowTempK);
  vec3 plumeGlow = plumeBlackbody(uPlumeGlowTempK) * plumeIntensity * (0.55 * mieNorm + 0.35 * opticalNorm);

  vec3 color = (rayleighColor + mieColor) * hazeTransmittance + sootHaze + plumeGlow;
  float alpha = clamp(
    opticalNorm * 0.78 + mieNorm * 0.32 + dustScatter * 0.52 + plumeIntensity * 0.45,
    0.0,
    0.88
  );

  gl_FragColor = vec4(color, alpha);
}
`;

export class AtmosphereShader {
  readonly mesh: THREE.Mesh;

  private readonly geometry: THREE.SphereGeometry;
  private readonly material: THREE.ShaderMaterial;

  constructor(segments = 64) {
    this.geometry = new THREE.SphereGeometry(1.0, segments, segments);
    this.material = new THREE.ShaderMaterial({
      vertexShader: ATMOSPHERE_VERTEX_SHADER,
      fragmentShader: ATMOSPHERE_FRAGMENT_SHADER,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.NormalBlending,
      uniforms: {
        uPlanetCenterRelKm: { value: new THREE.Vector3(0.0, 0.0, 0.0) },
        uPlanetRadiusKm: { value: 6371.0 },
        uAtmosphereRadiusKm: { value: 6371.0 * 1.045 },
        uSunDir: { value: new THREE.Vector3(0.6, 0.5, 0.624).normalize() },
        uOpticalDepthTau: { value: 0.0 },
        uPlumeGlowTempK: { value: 0.0 },
      },
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 2;
  }

  getUniforms(): Record<string, THREE.IUniform<unknown>> {
    return this.material.uniforms;
  }

  update(
    cameraPosKm: [number, number, number],
    planetComKm: [number, number, number],
    planetRadiusKm: number,
    sunDir: THREE.Vector3,
    opticalDepthTau: number,
    plumeGlowTempK: number
  ): void {
    if (!Number.isFinite(planetRadiusKm) || planetRadiusKm <= 0.0) {
      this.mesh.visible = false;
      return;
    }

    this.mesh.visible = true;
    const relX = planetComKm[0] - cameraPosKm[0];
    const relY = planetComKm[1] - cameraPosKm[1];
    const relZ = planetComKm[2] - cameraPosKm[2];
    const atmRadiusKm = planetRadiusKm * 1.045;

    this.mesh.position.set(relX, relY, relZ);
    this.mesh.scale.setScalar(atmRadiusKm);
    this.mesh.updateMatrixWorld();

    const uCenter = this.material.uniforms.uPlanetCenterRelKm.value as THREE.Vector3;
    uCenter.set(relX, relY, relZ);

    this.material.uniforms.uPlanetRadiusKm.value = planetRadiusKm;
    this.material.uniforms.uAtmosphereRadiusKm.value = atmRadiusKm;

    const uSun = this.material.uniforms.uSunDir.value as THREE.Vector3;
    uSun.copy(sunDir);
    if (uSun.lengthSq() > 1.0e-12) {
      uSun.normalize();
    } else {
      uSun.set(0.0, 1.0, 0.0);
    }

    this.material.uniforms.uOpticalDepthTau.value = Math.max(0.0, opticalDepthTau);
    this.material.uniforms.uPlumeGlowTempK.value = Math.max(0.0, plumeGlowTempK);
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
