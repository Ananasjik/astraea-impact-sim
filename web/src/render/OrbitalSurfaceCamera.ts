import * as THREE from 'three';

export type CameraMode = 'OrbitalFree' | 'SurfaceLock';

export interface SurfaceAnchorCoord {
  face: number; // 0..5
  u: number;    // 0..1
  v: number;    // 0..1
}

export const MIN_ALTITUDE_KM = 0.001; // 1 meter
export const MAX_ALTITUDE_KM = 1.0e5; // 10^8 meters

const MAX_ORBITAL_PITCH_RAD = Math.PI * 0.495;
const MAX_SURFACE_TILT_RAD = Math.PI * 0.48;
const WHEEL_LOG_SENSITIVITY = 0.0024;
const ORBIT_SENSITIVITY_RAD = 0.005;

function clamp(v: number, minVal: number, maxVal: number): number {
  return Math.max(minVal, Math.min(maxVal, v));
}

export function clampAltitudeKm(altKm: number): number {
  if (!Number.isFinite(altKm)) {
    return MIN_ALTITUDE_KM;
  }
  return clamp(altKm, MIN_ALTITUDE_KM, MAX_ALTITUDE_KM);
}

/**
 * Maps cube-sphere (face, u, v) to unit sphere normal matching engine/planet_builder.cpp:
 * face 0 (+X): ( 1,  sy, -sx), face 1 (-X): (-1,  sy,  sx)
 * face 2 (+Y): ( sx,  1, -sy), face 3 (-Y): ( sx, -1,  sy)
 * face 4 (+Z): ( sx, sy,   1), face 5 (-Z): (-sx, sy,  -1)
 */
export function cubeFaceUvToUnitNormal(
  face: number,
  u: number,
  v: number,
  out: THREE.Vector3 = new THREE.Vector3()
): THREE.Vector3 {
  const sx = 2.0 * u - 1.0;
  const sy = 2.0 * v - 1.0;
  const f = ((Math.trunc(face) % 6) + 6) % 6;

  let vx = 0.0;
  let vy = 0.0;
  let vz = 0.0;

  switch (f) {
    case 0:
      vx = 1.0;
      vy = sy;
      vz = -sx;
      break;
    case 1:
      vx = -1.0;
      vy = sy;
      vz = sx;
      break;
    case 2:
      vx = sx;
      vy = 1.0;
      vz = -sy;
      break;
    case 3:
      vx = sx;
      vy = -1.0;
      vz = sy;
      break;
    case 4:
      vx = sx;
      vy = sy;
      vz = 1.0;
      break;
    case 5:
    default:
      vx = -sx;
      vy = sy;
      vz = -1.0;
      break;
  }

  const invLen = 1.0 / Math.sqrt(vx * vx + vy * vy + vz * vz);
  return out.set(vx * invLen, vy * invLen, vz * invLen);
}

/**
 * Inverse mapping from unit sphere normal to cube-sphere (face, u, v).
 */
export function unitNormalToCubeFaceUv(
  nx: number,
  ny: number,
  nz: number
): SurfaceAnchorCoord {
  const ax = Math.abs(nx);
  const ay = Math.abs(ny);
  const az = Math.abs(nz);

  let face = 0;
  let sx = 0.0;
  let sy = 0.0;

  if (ax >= ay && ax >= az) {
    const denom = Math.max(ax, 1.0e-12);
    if (nx >= 0.0) {
      face = 0;
      sx = -nz / denom;
      sy = ny / denom;
    } else {
      face = 1;
      sx = nz / denom;
      sy = ny / denom;
    }
  } else if (ay >= ax && ay >= az) {
    const denom = Math.max(ay, 1.0e-12);
    if (ny >= 0.0) {
      face = 2;
      sx = nx / denom;
      sy = -nz / denom;
    } else {
      face = 3;
      sx = nx / denom;
      sy = nz / denom;
    }
  } else {
    const denom = Math.max(az, 1.0e-12);
    if (nz >= 0.0) {
      face = 4;
      sx = nx / denom;
      sy = ny / denom;
    } else {
      face = 5;
      sx = -nx / denom;
      sy = ny / denom;
    }
  }

  return {
    face,
    u: clamp(0.5 * (sx + 1.0), 0.0, 1.0),
    v: clamp(0.5 * (sy + 1.0), 0.0, 1.0),
  };
}

export function rotateYUnitNormal(
  nx: number,
  ny: number,
  nz: number,
  angleRad: number,
  out: THREE.Vector3 = new THREE.Vector3()
): THREE.Vector3 {
  if (angleRad === 0.0) {
    return out.set(nx, ny, nz);
  }
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  return out.set(nx * c + nz * s, ny, -nx * s + nz * c);
}

export class OrbitalSurfaceCamera {
  readonly camera: THREE.PerspectiveCamera;
  mode: CameraMode = 'OrbitalFree';
  altitudeKm: number;
  worldPositionKm: [number, number, number] = [0.0, 0.0, 24371.0];
  surfaceAnchor: SurfaceAnchorCoord = { face: 4, u: 0.5, v: 0.5 };

  private orbitalYawRad = 0.45;
  private orbitalPitchRad = 0.25;
  private surfaceYawRad = 0.0;
  private surfaceTiltRad = 0.0;
  private smoothTargetAltitudeKm: number | null = null;

  private readonly tmpNormal = new THREE.Vector3();
  private readonly tmpLookTarget = new THREE.Vector3();
  private readonly tmpUp = new THREE.Vector3();

  constructor(aspect = 16 / 9, initialAltitudeKm = 18000.0) {
    this.camera = new THREE.PerspectiveCamera(50, aspect, 1.0e-4, 1.0e8);
    this.altitudeKm = clampAltitudeKm(initialAltitudeKm);
  }

  setAspect(aspect: number): void {
    if (aspect > 0.0 && Number.isFinite(aspect)) {
      this.camera.aspect = aspect;
      this.camera.updateProjectionMatrix();
    }
  }

  setMode(mode: CameraMode, anchor?: SurfaceAnchorCoord): void {
    this.mode = mode;
    if (anchor !== undefined) {
      this.surfaceAnchor = {
        face: ((Math.trunc(anchor.face) % 6) + 6) % 6,
        u: clamp(anchor.u, 0.0, 1.0),
        v: clamp(anchor.v, 0.0, 1.0),
      };
    }
  }

  setTargetAltitudeKm(targetAltitudeKm: number): void {
    this.smoothTargetAltitudeKm = clampAltitudeKm(targetAltitudeKm);
  }

  zoomByWheel(deltaY: number): void {
    if (!Number.isFinite(deltaY) || deltaY === 0.0) {
      return;
    }
    this.smoothTargetAltitudeKm = null;
    const factor = Math.exp(deltaY * WHEEL_LOG_SENSITIVITY);
    this.altitudeKm = clampAltitudeKm(this.altitudeKm * factor);
  }

  orbitByDelta(dx: number, dy: number): void {
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
      return;
    }
    if (this.mode === 'OrbitalFree') {
      this.orbitalYawRad -= dx * ORBIT_SENSITIVITY_RAD;
      this.orbitalPitchRad = clamp(
        this.orbitalPitchRad + dy * ORBIT_SENSITIVITY_RAD,
        -MAX_ORBITAL_PITCH_RAD,
        MAX_ORBITAL_PITCH_RAD
      );
    } else {
      this.surfaceYawRad -= dx * ORBIT_SENSITIVITY_RAD;
      this.surfaceTiltRad = clamp(
        this.surfaceTiltRad + dy * ORBIT_SENSITIVITY_RAD,
        0.0,
        MAX_SURFACE_TILT_RAD
      );
    }
  }

  panSurfaceAnchor(du: number, dv: number): void {
    const nextU = this.surfaceAnchor.u + du;
    const nextV = this.surfaceAnchor.v + dv;
    if (nextU >= 0.0 && nextU <= 1.0 && nextV >= 0.0 && nextV <= 1.0) {
      this.surfaceAnchor.u = nextU;
      this.surfaceAnchor.v = nextV;
      return;
    }
    const n = cubeFaceUvToUnitNormal(
      this.surfaceAnchor.face,
      nextU,
      nextV,
      this.tmpNormal
    );
    this.surfaceAnchor = unitNormalToCubeFaceUv(n.x, n.y, n.z);
  }

  update(
    dtSec: number,
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    sampleSurfaceDisplacementKm: (face: number, u: number, v: number) => number,
    sampleTracerSurfacePosKm?: (
      face: number,
      u: number,
      v: number
    ) => [number, number, number] | null
  ): void {
    this.altitudeKm = clampAltitudeKm(this.altitudeKm);

    if (this.smoothTargetAltitudeKm !== null) {
      if (dtSec > 0.0 && Number.isFinite(dtSec)) {
        const alpha = 1.0 - Math.exp(-10.0 * dtSec);
        const logCur = Math.log(this.altitudeKm);
        const logTgt = Math.log(this.smoothTargetAltitudeKm);
        const nextLog = logCur + (logTgt - logCur) * alpha;
        this.altitudeKm = clampAltitudeKm(Math.exp(nextLog));
        if (Math.abs(nextLog - logTgt) < 1.0e-4) {
          this.altitudeKm = this.smoothTargetAltitudeKm;
          this.smoothTargetAltitudeKm = null;
        }
      }
    }

    const safeRadiusKm = Math.max(livingRadiusKm, 0.0);

    if (this.mode === 'OrbitalFree') {
      const cp = Math.cos(this.orbitalPitchRad);
      const sp = Math.sin(this.orbitalPitchRad);
      const cy = Math.cos(this.orbitalYawRad);
      const sy = Math.sin(this.orbitalYawRad);

      const nx = cp * sy;
      const ny = sp;
      const nz = cp * cy;
      const rKm = safeRadiusKm + this.altitudeKm;

      this.worldPositionKm[0] = livingComKm[0] + nx * rKm;
      this.worldPositionKm[1] = livingComKm[1] + ny * rKm;
      this.worldPositionKm[2] = livingComKm[2] + nz * rKm;

      this.tmpLookTarget.set(-nx * rKm, -ny * rKm, -nz * rKm);
      this.tmpUp.set(-sp * sy, cp, -sp * cy).normalize();
    } else {
      let nx = 0.0;
      let ny = 0.0;
      let nz = 1.0;
      let resolvedFromTracer = false;

      if (sampleTracerSurfacePosKm !== undefined) {
        const tracerPos = sampleTracerSurfacePosKm(
          this.surfaceAnchor.face,
          this.surfaceAnchor.u,
          this.surfaceAnchor.v
        );
        if (tracerPos !== null) {
          const ox = tracerPos[0] - livingComKm[0];
          const oy = tracerPos[1] - livingComKm[1];
          const oz = tracerPos[2] - livingComKm[2];
          const len = Math.sqrt(ox * ox + oy * oy + oz * oz);
          if (len > 1.0e-6) {
            const invLen = 1.0 / len;
            nx = ox * invLen;
            ny = oy * invLen;
            nz = oz * invLen;
            this.worldPositionKm[0] = tracerPos[0] + nx * this.altitudeKm;
            this.worldPositionKm[1] = tracerPos[1] + ny * this.altitudeKm;
            this.worldPositionKm[2] = tracerPos[2] + nz * this.altitudeKm;
            resolvedFromTracer = true;
          }
        }
      }

      if (!resolvedFromTracer) {
        cubeFaceUvToUnitNormal(
          this.surfaceAnchor.face,
          this.surfaceAnchor.u,
          this.surfaceAnchor.v,
          this.tmpNormal
        );
        rotateYUnitNormal(
          this.tmpNormal.x,
          this.tmpNormal.y,
          this.tmpNormal.z,
          spinAngleRad,
          this.tmpNormal
        );
        nx = this.tmpNormal.x;
        ny = this.tmpNormal.y;
        nz = this.tmpNormal.z;

        const dispKm = sampleSurfaceDisplacementKm(
          this.surfaceAnchor.face,
          this.surfaceAnchor.u,
          this.surfaceAnchor.v
        );
        const rKm = safeRadiusKm + dispKm + this.altitudeKm;
        this.worldPositionKm[0] = livingComKm[0] + nx * rKm;
        this.worldPositionKm[1] = livingComKm[1] + ny * rKm;
        this.worldPositionKm[2] = livingComKm[2] + nz * rKm;
      }

      // Build smooth tangent frame perpendicular to outward surface normal
      let refX = 0.0;
      let refY = 1.0;
      let refZ = 0.0;
      if (Math.abs(ny) > 0.99) {
        refY = 0.0;
        refZ = 1.0;
      }
      // east = normalize(cross(ref, n))
      let ex = refY * nz - refZ * ny;
      let ey = refZ * nx - refX * nz;
      let ez = refX * ny - refY * nx;
      const eInv = 1.0 / Math.max(Math.sqrt(ex * ex + ey * ey + ez * ez), 1.0e-12);
      ex *= eInv;
      ey *= eInv;
      ez *= eInv;

      // north = cross(n, east)
      const noX = ny * ez - nz * ey;
      const noY = nz * ex - nx * ez;
      const noZ = nx * ey - ny * ex;

      const cy = Math.cos(this.surfaceYawRad);
      const sy = Math.sin(this.surfaceYawRad);
      const hx = cy * noX + sy * ex;
      const hy = cy * noY + sy * ey;
      const hz = cy * noZ + sy * ez;

      const ct = Math.cos(this.surfaceTiltRad);
      const st = Math.sin(this.surfaceTiltRad);

      this.tmpLookTarget.set(
        -ct * nx + st * hx,
        -ct * ny + st * hy,
        -ct * nz + st * hz
      );
      this.tmpUp.set(
        st * nx + ct * hx,
        st * ny + ct * hy,
        st * nz + ct * hz
      ).normalize();
    }

    // Camera-centered rendering: keep camera at origin (0, 0, 0) to eliminate float32 jitter
    this.camera.position.set(0.0, 0.0, 0.0);
    this.camera.up.copy(this.tmpUp);
    this.camera.lookAt(this.tmpLookTarget);
    this.camera.updateMatrixWorld();
  }

  toCameraRelativePos(
    worldKm: [number, number, number],
    outVec3: THREE.Vector3
  ): THREE.Vector3 {
    const dx = worldKm[0] - this.worldPositionKm[0];
    const dy = worldKm[1] - this.worldPositionKm[1];
    const dz = worldKm[2] - this.worldPositionKm[2];
    return outVec3.set(dx, dy, dz);
  }
}
