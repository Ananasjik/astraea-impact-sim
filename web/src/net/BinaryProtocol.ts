import { unzlibSync } from 'fflate';

export const PCLS_MAGIC = 0x50434c53;
export const PCLS_VERSION = 1;
export const HEADER_BYTES = 40;
export const PARTICLE_RECORD_BYTES = 20;
export const TRACER_RECORD_BYTES = 40;

export const FLAG_ZLIB = 0x01;
export const FLAG_HAS_TRACERS = 0x02;
export const FLAG_HAS_LIVING_COM = 0x04;
export const FLAG_HAS_SPEED_ARRAY = 0x08;

export const TRACER_FLAG_OCEAN = 0x01;
export const TRACER_FLAG_DISRUPTED = 0x02;

export interface DecodedFrameHeader {
  version: number;
  flags: number;
  simTimeSec: number;
  frameSeq: number;
  dtSec: number;
  particleCount: number;
  tracerCount: number;
  tracerGridRes: number;
  payloadBytes: number;
}

export interface ParticleBufferPool {
  capacity: number;
  count: number;
  positionsKm: Float32Array;
  props: Float32Array;
  meta: Float32Array;
}

export interface TracerBufferPool {
  capacity: number;
  count: number;
  gridRes: number;
  anchorIndex: Uint16Array;
  flags: Uint16Array;
  positionsKm: Float32Array;
  dispKm: Float32Array;
  normalDelta: Float32Array;
  tempK: Float32Array;
  damage: Float32Array;
  overpressureKpa: Float32Array;
  tsunamiHeightM: Float32Array;
}

const INV_255 = 1.0 / 255.0;

const TEMP_LUT = new Float32Array(65536);
const PRESSURE_LUT = new Float32Array(65536);
const RADIUS_LUT = new Float32Array(256);
const SPEED_LUT = new Float32Array(256);

for (let i = 0; i < 65536; i++) {
  TEMP_LUT[i] = 10.0 * Math.pow(20000.0, i / 65535.0);
  PRESSURE_LUT[i] = i === 0 ? 0.0 : 1.0e-3 * Math.pow(1.0e7, (i - 1) / 65534.0);
}

for (let i = 0; i < 256; i++) {
  const frac = i * INV_255;
  RADIUS_LUT[i] = Math.pow(2000.0, frac);
  SPEED_LUT[i] = 0.05 * Math.pow(2000.0, frac);
}

export function createParticleBufferPool(capacity: number): ParticleBufferPool {
  if (!Number.isInteger(capacity) || capacity < 0) {
    throw new Error(`Invalid particle pool capacity: ${capacity}`);
  }
  return {
    capacity,
    count: 0,
    positionsKm: new Float32Array(capacity * 3),
    props: new Float32Array(capacity * 4),
    meta: new Float32Array(capacity * 4),
  };
}

export function createTracerBufferPool(capacity: number): TracerBufferPool {
  if (!Number.isInteger(capacity) || capacity < 0) {
    throw new Error(`Invalid tracer pool capacity: ${capacity}`);
  }
  return {
    capacity,
    count: 0,
    gridRes: 0,
    anchorIndex: new Uint16Array(capacity),
    flags: new Uint16Array(capacity),
    positionsKm: new Float32Array(capacity * 3),
    dispKm: new Float32Array(capacity),
    normalDelta: new Float32Array(capacity),
    tempK: new Float32Array(capacity),
    damage: new Float32Array(capacity),
    overpressureKpa: new Float32Array(capacity),
    tsunamiHeightM: new Float32Array(capacity),
  };
}

function validatePoolCapacity(
  particleCount: number,
  tracerCount: number,
  particles: ParticleBufferPool,
  tracers: TracerBufferPool
): void {
  if (particleCount > particles.capacity) {
    throw new Error(
      `Frame particleCount (${particleCount}) exceeds pool capacity (${particles.capacity})`
    );
  }
  if (tracerCount > tracers.capacity) {
    throw new Error(
      `Frame tracerCount (${tracerCount}) exceeds pool capacity (${tracers.capacity})`
    );
  }
  if (
    particles.positionsKm.length < particleCount * 3 ||
    particles.props.length < particleCount * 4 ||
    particles.meta.length < particleCount * 4
  ) {
    throw new Error('ParticleBufferPool backing TypedArrays are smaller than particleCount');
  }
  if (
    tracerCount > 0 &&
    (tracers.anchorIndex.length < tracerCount ||
      tracers.flags.length < tracerCount ||
      tracers.positionsKm.length < tracerCount * 3 ||
      tracers.dispKm.length < tracerCount ||
      tracers.normalDelta.length < tracerCount ||
      tracers.tempK.length < tracerCount ||
      tracers.damage.length < tracerCount ||
      tracers.overpressureKpa.length < tracerCount ||
      tracers.tsunamiHeightM.length < tracerCount)
  ) {
    throw new Error('TracerBufferPool backing TypedArrays are smaller than tracerCount');
  }
}

export function decodeBinaryFrame(
  buffer: ArrayBuffer,
  particles: ParticleBufferPool,
  tracers: TracerBufferPool
): DecodedFrameHeader {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error(
      `Binary frame too short: expected >= ${HEADER_BYTES} bytes, got ${buffer.byteLength}`
    );
  }

  const headerView = new DataView(buffer, 0, HEADER_BYTES);
  const magic = headerView.getUint32(0, true);
  if (magic !== PCLS_MAGIC) {
    throw new Error(
      `Invalid PCLS magic: expected 0x${PCLS_MAGIC.toString(16)}, got 0x${magic.toString(16)}`
    );
  }

  const version = headerView.getUint16(4, true);
  if (version !== PCLS_VERSION) {
    throw new Error(`Unsupported PCLS frame version: ${version}`);
  }

  const flags = headerView.getUint16(6, true);
  const simTimeSec = headerView.getFloat64(8, true);
  const frameSeq = headerView.getUint32(16, true);
  const dtSec = headerView.getFloat32(20, true);
  const particleCount = headerView.getUint32(24, true);
  const tracerCount = headerView.getUint32(28, true);
  const tracerGridRes = headerView.getUint32(32, true);
  const payloadBytes = headerView.getUint32(36, true);

  const hasZlib = (flags & FLAG_ZLIB) !== 0;
  const hasSpeed = (flags & FLAG_HAS_SPEED_ARRAY) !== 0;
  const expectedUncompressed =
    particleCount * PARTICLE_RECORD_BYTES +
    tracerCount * TRACER_RECORD_BYTES +
    (hasSpeed ? particleCount : 0);

  if (payloadBytes !== expectedUncompressed) {
    throw new Error(
      `Payload byte mismatch: header payloadBytes=${payloadBytes}, expected=${expectedUncompressed}`
    );
  }

  validatePoolCapacity(particleCount, tracerCount, particles, tracers);

  let payloadView: DataView;
  if (hasZlib) {
    if (buffer.byteLength <= HEADER_BYTES) {
      throw new Error('Zlib-flagged frame has empty compressed payload');
    }
    const compressedSlice = new Uint8Array(buffer, HEADER_BYTES);
    const decompressed = unzlibSync(compressedSlice);
    if (decompressed.byteLength !== payloadBytes) {
      throw new Error(
        `Decompressed payload length mismatch: expected ${payloadBytes}, got ${decompressed.byteLength}`
      );
    }
    payloadView = new DataView(
      decompressed.buffer,
      decompressed.byteOffset,
      decompressed.byteLength
    );
  } else {
    if (buffer.byteLength < HEADER_BYTES + payloadBytes) {
      throw new Error(
        `Truncated binary frame: expected ${HEADER_BYTES + payloadBytes} bytes, got ${buffer.byteLength}`
      );
    }
    payloadView = new DataView(buffer, HEADER_BYTES, payloadBytes);
  }

  const pPos = particles.positionsKm;
  const pProps = particles.props;
  const pMeta = particles.meta;

  for (let i = 0; i < particleCount; i++) {
    const off = i * PARTICLE_RECORD_BYTES;
    const i3 = i * 3;
    const i4 = i * 4;

    pPos[i3] = payloadView.getFloat32(off, true);
    pPos[i3 + 1] = payloadView.getFloat32(off + 4, true);
    pPos[i3 + 2] = payloadView.getFloat32(off + 8, true);

    pProps[i4] = TEMP_LUT[payloadView.getUint16(off + 12, true)];
    pProps[i4 + 1] = PRESSURE_LUT[payloadView.getUint16(off + 14, true)];
    pProps[i4 + 2] = payloadView.getUint8(off + 18) * INV_255;
    pProps[i4 + 3] = RADIUS_LUT[payloadView.getUint8(off + 19)];

    const materialId = payloadView.getUint8(off + 16);
    const phaseAndBody = payloadView.getUint8(off + 17);
    pMeta[i4] = materialId;
    pMeta[i4 + 1] = phaseAndBody & 0x0f;
    pMeta[i4 + 2] = (phaseAndBody >> 4) & 0x0f;
    pMeta[i4 + 3] = 0.0;
  }

  const tracerBase = particleCount * PARTICLE_RECORD_BYTES;
  const tAnchor = tracers.anchorIndex;
  const tFlags = tracers.flags;
  const tPos = tracers.positionsKm;
  const tDisp = tracers.dispKm;
  const tNormalDelta = tracers.normalDelta;
  const tTemp = tracers.tempK;
  const tDamage = tracers.damage;
  const tOverpressure = tracers.overpressureKpa;
  const tTsunami = tracers.tsunamiHeightM;

  for (let i = 0; i < tracerCount; i++) {
    const off = tracerBase + i * TRACER_RECORD_BYTES;
    const i3 = i * 3;

    tAnchor[i] = payloadView.getUint16(off, true);
    tFlags[i] = payloadView.getUint16(off + 2, true);

    tPos[i3] = payloadView.getFloat32(off + 4, true);
    tPos[i3 + 1] = payloadView.getFloat32(off + 8, true);
    tPos[i3 + 2] = payloadView.getFloat32(off + 12, true);

    tDisp[i] = payloadView.getFloat32(off + 16, true);
    tNormalDelta[i] = payloadView.getFloat32(off + 20, true);
    tTemp[i] = payloadView.getFloat32(off + 24, true);
    tDamage[i] = payloadView.getFloat32(off + 28, true);
    tOverpressure[i] = payloadView.getFloat32(off + 32, true);
    tTsunami[i] = payloadView.getFloat32(off + 36, true);
  }

  if (hasSpeed) {
    const speedBase = tracerBase + tracerCount * TRACER_RECORD_BYTES;
    for (let i = 0; i < particleCount; i++) {
      pMeta[i * 4 + 3] = SPEED_LUT[payloadView.getUint8(speedBase + i)];
    }
  }

  particles.count = particleCount;
  tracers.count = tracerCount;
  tracers.gridRes = tracerGridRes;

  return {
    version,
    flags,
    simTimeSec,
    frameSeq,
    dtSec,
    particleCount,
    tracerCount,
    tracerGridRes,
    payloadBytes,
  };
}
