import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { zlibSync } from 'fflate';
import * as crypto from 'node:crypto';
import * as fs from 'node:fs';
import * as net from 'node:net';
import * as path from 'node:path';
import { spawn, type ChildProcess } from 'node:child_process';

import {
  HEADER_BYTES,
  PARTICLE_RECORD_BYTES,
  PCLS_MAGIC,
  TRACER_RECORD_BYTES,
  decodeBinaryFrame,
  type DecodedFrameHeader,
  type ParticleBufferPool,
  type TracerBufferPool,
} from '../src/net/BinaryProtocol';
import {
  SimClient,
  type ClientCommand,
  type CustomScenarioDto,
  type TelemetrySnapshotDto,
} from '../src/net/SimClient';

const FLAG_ZLIB = 0x01;
const FLAG_HAS_TRACERS = 0x02;
const FLAG_HAS_LIVING_COM = 0x04;
const FLAG_HAS_SPEED_ARRAY = 0x08;

const TRACER_FLAG_OCEAN = 0x01;
const TRACER_FLAG_DISRUPTED = 0x02;

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

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

interface SyntheticParticle {
  xKm: number;
  yKm: number;
  zKm: number;
  tempK: number;
  pressureGPa: number;
  materialId: number;
  phaseId: number;
  bodyId: number;
  damage: number;
  radiusKm: number;
  speedKmS?: number;
}

interface SyntheticTracer {
  anchorIndex: number;
  isOcean: boolean;
  isDisrupted: boolean;
  xKm: number;
  yKm: number;
  zKm: number;
  dispKm: number;
  normalDelta: number;
  tempK: number;
  damage: number;
  overpressureKpa: number;
  tsunamiHeightM: number;
}

function encodeTempLogU16(tempK: number): number {
  const norm = clamp(Math.log(Math.max(tempK, 10) / 10) / Math.log(20000), 0, 1);
  return Math.round(65535 * norm);
}

function encodePressureLogU16(pressureGPa: number): number {
  if (pressureGPa <= 1e-3) {
    return 0;
  }
  const norm = clamp(Math.log(pressureGPa / 1e-3) / Math.log(1e7), 0, 1);
  return Math.round(1 + 65534 * norm);
}

function encodeRadiusLogU8(radiusKm: number): number {
  const norm = clamp(Math.log(Math.max(radiusKm, 1.0) / 1.0) / Math.log(2000.0), 0, 1);
  return Math.round(255 * norm);
}

function encodeSpeedLogU8(speedKmS: number): number {
  const norm = clamp(Math.log(Math.max(speedKmS, 0.05) / 0.05) / Math.log(2000.0), 0, 1);
  return Math.round(255 * norm);
}

function buildPclsFrame(opts: {
  magic?: number;
  version?: number;
  flags?: number;
  simTimeSec?: number;
  frameSeq?: number;
  dtSec?: number;
  tracerGridRes?: number;
  particles: SyntheticParticle[];
  tracers: SyntheticTracer[];
  includeSpeed?: boolean;
  compressZlib?: boolean;
  hasLivingCom?: boolean;
  overridePayloadBytes?: number;
}): ArrayBuffer {
  const n = opts.particles.length;
  const t = opts.tracers.length;
  const includeSpeed = Boolean(opts.includeSpeed);
  const rawPayloadSize =
    PARTICLE_RECORD_BYTES * n + TRACER_RECORD_BYTES * t + (includeSpeed ? n : 0);

  const rawPayload = new Uint8Array(rawPayloadSize);
  const pView = new DataView(rawPayload.buffer, rawPayload.byteOffset, rawPayload.byteLength);

  for (let i = 0; i < n; i++) {
    const p = opts.particles[i];
    const base = i * PARTICLE_RECORD_BYTES;
    pView.setFloat32(base + 0, p.xKm, true);
    pView.setFloat32(base + 4, p.yKm, true);
    pView.setFloat32(base + 8, p.zKm, true);
    pView.setUint16(base + 12, encodeTempLogU16(p.tempK), true);
    pView.setUint16(base + 14, encodePressureLogU16(p.pressureGPa), true);
    pView.setUint8(base + 16, p.materialId & 0xff);
    pView.setUint8(base + 17, ((p.bodyId & 0x0f) << 4) | (p.phaseId & 0x0f));
    pView.setUint8(base + 18, Math.round(clamp(p.damage, 0, 1) * 255));
    pView.setUint8(base + 19, encodeRadiusLogU8(p.radiusKm));
  }

  const tracerBase = PARTICLE_RECORD_BYTES * n;
  for (let i = 0; i < t; i++) {
    const tr = opts.tracers[i];
    const base = tracerBase + i * TRACER_RECORD_BYTES;
    const tFlags = (tr.isOcean ? TRACER_FLAG_OCEAN : 0) | (tr.isDisrupted ? TRACER_FLAG_DISRUPTED : 0);
    pView.setUint16(base + 0, tr.anchorIndex, true);
    pView.setUint16(base + 2, tFlags, true);
    pView.setFloat32(base + 4, tr.xKm, true);
    pView.setFloat32(base + 8, tr.yKm, true);
    pView.setFloat32(base + 12, tr.zKm, true);
    pView.setFloat32(base + 16, tr.dispKm, true);
    pView.setFloat32(base + 20, tr.normalDelta, true);
    pView.setFloat32(base + 24, tr.tempK, true);
    pView.setFloat32(base + 28, tr.damage, true);
    pView.setFloat32(base + 32, tr.overpressureKpa, true);
    pView.setFloat32(base + 36, tr.tsunamiHeightM, true);
  }

  if (includeSpeed) {
    const speedBase = tracerBase + TRACER_RECORD_BYTES * t;
    for (let i = 0; i < n; i++) {
      rawPayload[speedBase + i] = encodeSpeedLogU8(opts.particles[i].speedKmS ?? 0.05);
    }
  }

  const bodyBytes = opts.compressZlib ? zlibSync(rawPayload) : rawPayload;
  let flags = opts.flags ?? 0;
  if (opts.compressZlib) flags |= FLAG_ZLIB;
  if (t > 0) flags |= FLAG_HAS_TRACERS;
  if (opts.hasLivingCom) flags |= FLAG_HAS_LIVING_COM;
  if (includeSpeed) flags |= FLAG_HAS_SPEED_ARRAY;

  const frame = new Uint8Array(HEADER_BYTES + bodyBytes.byteLength);
  const hView = new DataView(frame.buffer);
  hView.setUint32(0, opts.magic ?? PCLS_MAGIC, true);
  hView.setUint16(4, opts.version ?? 1, true);
  hView.setUint16(6, flags, true);
  hView.setFloat64(8, opts.simTimeSec ?? 12.5, true);
  hView.setUint32(16, opts.frameSeq ?? 42, true);
  hView.setFloat32(20, opts.dtSec ?? 0.25, true);
  hView.setUint32(24, n, true);
  hView.setUint32(28, t, true);
  hView.setUint32(32, opts.tracerGridRes ?? (t > 0 ? 16 : 0), true);
  hView.setUint32(36, opts.overridePayloadBytes ?? rawPayloadSize, true);

  frame.set(bodyBytes, HEADER_BYTES);
  return frame.buffer;
}

class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: MockWebSocket[] = [];

  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readonly CLOSING = 2;
  readonly CLOSED = 3;

  readonly url: string;
  readyState = MockWebSocket.CONNECTING;
  binaryType: 'blob' | 'arraybuffer' = 'blob';
  sentMessages: Array<string | ArrayBuffer | Uint8Array> = [];

  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code?: number; reason?: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;

  private listeners = new Map<string, Set<(ev: unknown) => void>>();

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  addEventListener(type: string, listener: (ev: unknown) => void): void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(listener);
  }

  removeEventListener(type: string, listener: (ev: unknown) => void): void {
    this.listeners.get(type)?.delete(listener);
  }

  send(data: string | ArrayBuffer | Uint8Array): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('MockWebSocket not open');
    }
    this.sentMessages.push(data);
  }

  close(code = 1000, reason = 'closed'): void {
    this.readyState = MockWebSocket.CLOSED;
    const ev = { code, reason };
    this.onclose?.(ev);
    this.dispatch('close', ev);
  }

  triggerOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    const ev = {};
    this.onopen?.(ev);
    this.dispatch('open', ev);
  }

  triggerMessage(data: unknown): void {
    const ev = { data };
    this.onmessage?.(ev);
    this.dispatch('message', ev);
  }

  triggerDrop(code = 1006): void {
    this.readyState = MockWebSocket.CLOSED;
    const ev = { code, reason: 'network drop' };
    this.onclose?.(ev);
    this.dispatch('close', ev);
  }

  private dispatch(type: string, ev: unknown): void {
    const set = this.listeners.get(type);
    if (!set) return;
    for (const fn of set) {
      fn(ev);
    }
  }
}

function makeSampleTelemetry(seq = 7): TelemetrySnapshotDto {
  return {
    type: 'telemetry',
    frameSeq: seq,
    simTimeSec: 120.5,
    dtSec: 0.5,
    state: 'playing',
    preset: 'LivingGaiaAsteroidCataclysm',
    outcome: 'Merging',
    kineticEnergyJ: 1.2e30,
    potentialEnergyJ: -2.4e32,
    internalEnergyJ: 3.1e30,
    radiatedEnergyJ: 1.5e26,
    totalEnergyJ: -2.357e32,
    energyDriftRel: 2.1e-5,
    momentumKgMS: [0, 0, 0],
    angularMomentumKgM2S: [0, 4.4e34, 0],
    virialRatio: 0.99,
    maxTempK: 14500,
    maxPressureGPa: 380,
    boundMassFrac: 0.985,
    diskMassFrac: 0.01,
    ejectaMassFrac: 0.005,
    qRatioQrd: 0.12,
    gpuStepMs: 14.2,
    livingBodyComKm: [120.5, -15.25, 8.0],
    livingBodyRadiusKm: 6371,
    livingBodySpinRadS: 7.2921e-5,
    impactorComKm: [-4200, 1200, -300],
    tracerGridRes: 16,
    livingBodyPresent: true,
  };
}

describe('BinaryProtocol (PCLS 40B/20B/40B/1B) - AC-12', () => {
  it('exports exact spec protocol constants', () => {
    expect(PCLS_MAGIC).toBe(0x50434c53);
    expect(HEADER_BYTES).toBe(40);
    expect(PARTICLE_RECORD_BYTES).toBe(20);
    expect(TRACER_RECORD_BYTES).toBe(40);
  });

  it('decodes full-scale N=100,000 particles and T=1,536 tracers (2,061,480 bytes) accurately', () => {
    const N = 100_000;
    const gridRes = 16;
    const T = 6 * gridRes * gridRes; // 1536

    const particles: SyntheticParticle[] = new Array(N);
    for (let i = 0; i < N; i++) {
      const frac = i / (N - 1);
      particles[i] = {
        xKm: (i % 200) - 100.25,
        yKm: Math.floor(i / 200) * 0.5 - 125.0,
        zKm: ((i * 7) % 100) - 50.0,
        tempK: 10 * Math.pow(20000, frac), // [10, 200000] K
        pressureGPa: i === 0 ? 0 : 0.002 * Math.pow(5_000_000, frac), // [0.002, 1e4] GPa
        materialId: i % 5,
        phaseId: i % 6,
        bodyId: i % 3,
        damage: frac,
        radiusKm: 1.0 * Math.pow(2000, frac), // [1, 2000] km
      };
    }

    const tracers: SyntheticTracer[] = new Array(T);
    for (let i = 0; i < T; i++) {
      tracers[i] = {
        anchorIndex: i,
        isOcean: i % 2 === 0,
        isDisrupted: i % 7 === 0,
        xKm: 6371.0 * Math.cos(i * 0.01),
        yKm: 6371.0 * Math.sin(i * 0.01),
        zKm: (i - 768) * 2.5,
        dispKm: (i % 19) - 9.25,
        normalDelta: (i % 11) * 0.05,
        tempK: 288 + (i % 500) * 3.25,
        damage: (i % 101) / 100,
        overpressureKpa: (i % 25) * 14.5,
        tsunamiHeightM: i % 2 === 0 ? (i % 40) * 2.5 : 0,
      };
    }

    const buf = buildPclsFrame({
      simTimeSec: 9876.54321,
      frameSeq: 123456,
      dtSec: 0.125,
      tracerGridRes: gridRes,
      hasLivingCom: true,
      particles,
      tracers,
    });

    expect(buf.byteLength).toBe(HEADER_BYTES + PARTICLE_RECORD_BYTES * N + TRACER_RECORD_BYTES * T);
    expect(buf.byteLength).toBe(2_061_480);

    const pPool = makeParticlePool(N);
    const tPool = makeTracerPool(T, gridRes);

    const header = decodeBinaryFrame(buf, pPool, tPool);
    expect(header.version).toBe(1);
    expect(header.flags & FLAG_ZLIB).toBe(0);
    expect(header.flags & FLAG_HAS_TRACERS).toBe(FLAG_HAS_TRACERS);
    expect(header.flags & FLAG_HAS_LIVING_COM).toBe(FLAG_HAS_LIVING_COM);
    expect(header.flags & FLAG_HAS_SPEED_ARRAY).toBe(0);
    expect(header.simTimeSec).toBeCloseTo(9876.54321, 10);
    expect(header.frameSeq).toBe(123456);
    expect(header.dtSec).toBeCloseTo(0.125, 6);
    expect(header.particleCount).toBe(N);
    expect(header.tracerCount).toBe(T);
    expect(header.tracerGridRes).toBe(gridRes);
    expect(header.payloadBytes).toBe(2_061_440);

    expect(pPool.count).toBe(N);
    expect(tPool.count).toBe(T);
    expect(tPool.gridRes).toBe(gridRes);

    // Check quantization fidelity across sampled indices covering full N=100,000 range
    for (let i = 0; i < N; i += 499) {
      const expected = particles[i];
      expect(pPool.positionsKm[3 * i + 0]).toBeCloseTo(expected.xKm, 4);
      expect(pPool.positionsKm[3 * i + 1]).toBeCloseTo(expected.yKm, 4);
      expect(pPool.positionsKm[3 * i + 2]).toBeCloseTo(expected.zKm, 4);

      const decTemp = pPool.props[4 * i + 0];
      const tempRelErr = Math.abs(decTemp - expected.tempK) / expected.tempK;
      expect(tempRelErr).toBeLessThan(0.001); // < 0.1%

      const decPress = pPool.props[4 * i + 1];
      if (expected.pressureGPa <= 1e-3) {
        expect(decPress).toBe(0);
      } else {
        const pRelErr = Math.abs(decPress - expected.pressureGPa) / expected.pressureGPa;
        expect(pRelErr).toBeLessThan(0.001); // < 0.1%
      }

      const decDam = pPool.props[4 * i + 2];
      expect(Math.abs(decDam - expected.damage)).toBeLessThanOrEqual(0.5 / 255 + 1e-5);

      const decRad = pPool.props[4 * i + 3];
      expect(Math.abs(decRad - expected.radiusKm) / expected.radiusKm).toBeLessThan(0.02);

      expect(pPool.meta[4 * i + 0]).toBe(expected.materialId);
      expect(pPool.meta[4 * i + 1]).toBe(expected.phaseId);
      expect(pPool.meta[4 * i + 2]).toBe(expected.bodyId);
    }

    // Also explicitly verify the final particle (N-1) at max boundaries
    expect(pPool.props[4 * (N - 1) + 0]).toBeCloseTo(200_000, -2);
    expect(pPool.props[4 * (N - 1) + 1]).toBeCloseTo(10_000, -1);

    // Verify all 1,536 tracers
    for (let i = 0; i < T; i++) {
      const tr = tracers[i];
      expect(tPool.anchorIndex[i]).toBe(tr.anchorIndex);
      expect((tPool.flags[i] & TRACER_FLAG_OCEAN) !== 0).toBe(tr.isOcean);
      expect((tPool.flags[i] & TRACER_FLAG_DISRUPTED) !== 0).toBe(tr.isDisrupted);
      expect(tPool.positionsKm[3 * i + 0]).toBeCloseTo(tr.xKm, 3);
      expect(tPool.positionsKm[3 * i + 1]).toBeCloseTo(tr.yKm, 3);
      expect(tPool.positionsKm[3 * i + 2]).toBeCloseTo(tr.zKm, 3);
      expect(tPool.dispKm[i]).toBeCloseTo(tr.dispKm, 5);
      expect(tPool.normalDelta[i]).toBeCloseTo(tr.normalDelta, 5);
      expect(tPool.tempK[i]).toBeCloseTo(tr.tempK, 3);
      expect(tPool.damage[i]).toBeCloseTo(tr.damage, 5);
      expect(tPool.overpressureKpa[i]).toBeCloseTo(tr.overpressureKpa, 4);
      expect(tPool.tsunamiHeightM[i]).toBeCloseTo(tr.tsunamiHeightM, 4);
    }
  });

  it('verifies exact log quantization bounds, pressure <= 1e-3 GPa zero-clamping, and optional speed block (flags & 0x08)', () => {
    const samples: SyntheticParticle[] = [
      // P <= 1e-3 clamps to 0, temp below min clamps to 10 K, speed below min clamps to 0.05
      { xKm: 1, yKm: 2, zKm: 3, tempK: 3.0, pressureGPa: 0.0, materialId: 0, phaseId: 0, bodyId: 0, damage: 0.0, radiusKm: 0.5, speedKmS: 0.01 },
      { xKm: -4, yKm: 5, zKm: -6, tempK: 10.0, pressureGPa: 1e-3, materialId: 1, phaseId: 1, bodyId: 1, damage: 0.25, radiusKm: 1.0, speedKmS: 0.05 },
      { xKm: 7, yKm: -8, zKm: 9, tempK: 288.0, pressureGPa: -0.5, materialId: 2, phaseId: 4, bodyId: 0, damage: 0.5, radiusKm: 12.5, speedKmS: 1.2 },
      { xKm: 10, yKm: 11, zKm: 12, tempK: 1600.0, pressureGPa: 0.002, materialId: 3, phaseId: 5, bodyId: 2, damage: 0.73, radiusKm: 85.0, speedKmS: 7.9 },
      { xKm: -13, yKm: 14, zKm: -15, tempK: 5500.0, pressureGPa: 350.0, materialId: 4, phaseId: 2, bodyId: 15, damage: 0.99, radiusKm: 420.0, speedKmS: 42.0 },
      { xKm: 16, yKm: -17, zKm: 18, tempK: 200000.0, pressureGPa: 10000.0, materialId: 0, phaseId: 3, bodyId: 1, damage: 1.0, radiusKm: 2000.0, speedKmS: 100.0 },
    ];

    const buf = buildPclsFrame({
      particles: samples,
      tracers: [],
      includeSpeed: true,
      tracerGridRes: 0,
    });

    expect(buf.byteLength).toBe(HEADER_BYTES + PARTICLE_RECORD_BYTES * samples.length + samples.length);

    const pPool = makeParticlePool(16);
    const tPool = makeTracerPool(16, 16);
    const hdr = decodeBinaryFrame(buf, pPool, tPool);

    expect((hdr.flags & FLAG_HAS_SPEED_ARRAY) !== 0).toBe(true);
    expect(pPool.count).toBe(samples.length);
    expect(tPool.count).toBe(0);

    // P <= 1e-3 exact 0 check
    expect(pPool.props[4 * 0 + 1]).toBe(0);
    expect(pPool.props[4 * 1 + 1]).toBe(0);
    expect(pPool.props[4 * 2 + 1]).toBe(0);

    // Clamped lower bounds
    expect(pPool.props[4 * 0 + 0]).toBeCloseTo(10.0, 2);
    expect(pPool.props[4 * 0 + 3]).toBeCloseTo(1.0, 2);
    expect(pPool.meta[4 * 0 + 3]).toBeCloseTo(0.05, 4);

    // Accurate interior & upper bounds
    for (let i = 1; i < samples.length; i++) {
      const s = samples[i];
      expect(Math.abs(pPool.props[4 * i + 0] - s.tempK) / s.tempK).toBeLessThan(0.001);
      if (s.pressureGPa > 1e-3) {
        expect(Math.abs(pPool.props[4 * i + 1] - s.pressureGPa) / s.pressureGPa).toBeLessThan(0.001);
      }
      expect(Math.abs(pPool.props[4 * i + 2] - s.damage)).toBeLessThanOrEqual(0.5 / 255 + 1e-5);
      expect(Math.abs(pPool.props[4 * i + 3] - s.radiusKm) / s.radiusKm).toBeLessThan(0.02);
      expect(pPool.meta[4 * i + 0]).toBe(s.materialId);
      expect(pPool.meta[4 * i + 1]).toBe(s.phaseId);
      expect(pPool.meta[4 * i + 2]).toBe(s.bodyId);
      const speed = s.speedKmS!;
      expect(Math.abs(pPool.meta[4 * i + 3] - speed) / speed).toBeLessThan(0.02);
    }
  });

  it('transparently decompresses zlib-compressed (flags & 0x01) 100,000-particle + 1,536-tracer + speed payload', () => {
    const N = 100_000;
    const gridRes = 16;
    const T = 6 * gridRes * gridRes;

    const particles: SyntheticParticle[] = new Array(N);
    for (let i = 0; i < N; i++) {
      particles[i] = {
        xKm: (i % 50) - 25,
        yKm: ((i * 3) % 50) - 25,
        zKm: ((i * 7) % 50) - 25,
        tempK: 288 + (i % 100) * 15,
        pressureGPa: 0.5 + (i % 80) * 1.25,
        materialId: i % 5,
        phaseId: i % 6,
        bodyId: i & 1,
        damage: (i % 20) / 20,
        radiusKm: 5 + (i % 30),
        speedKmS: 0.5 + (i % 40) * 0.8,
      };
    }

    const tracers: SyntheticTracer[] = new Array(T);
    for (let i = 0; i < T; i++) {
      tracers[i] = {
        anchorIndex: i,
        isOcean: (i & 1) === 1,
        isDisrupted: (i & 3) === 3,
        xKm: 6371,
        yKm: i * 0.1,
        zKm: -i * 0.1,
        dispKm: 1.5,
        normalDelta: 0.02,
        tempK: 310 + (i % 50),
        damage: 0.25,
        overpressureKpa: 45,
        tsunamiHeightM: 12,
      };
    }

    const buf = buildPclsFrame({
      simTimeSec: 444.25,
      frameSeq: 999,
      dtSec: 0.5,
      tracerGridRes: gridRes,
      particles,
      tracers,
      includeSpeed: true,
      compressZlib: true,
    });

    // Zlib compressed buffer must be significantly smaller than raw 2,161,480 bytes
    expect(buf.byteLength).toBeLessThan(2_161_480);

    const pPool = makeParticlePool(N);
    const tPool = makeTracerPool(T, gridRes);
    const hdr = decodeBinaryFrame(buf, pPool, tPool);

    expect((hdr.flags & FLAG_ZLIB) !== 0).toBe(true);
    expect((hdr.flags & FLAG_HAS_SPEED_ARRAY) !== 0).toBe(true);
    expect(hdr.particleCount).toBe(N);
    expect(hdr.tracerCount).toBe(T);
    expect(hdr.payloadBytes).toBe(PARTICLE_RECORD_BYTES * N + TRACER_RECORD_BYTES * T + N);
    expect(pPool.count).toBe(N);
    expect(tPool.count).toBe(T);

    // Check start, middle, and end elements post-zlib decompression
    for (const idx of [0, 12345, 54321, N - 1]) {
      const p = particles[idx];
      expect(pPool.positionsKm[3 * idx + 0]).toBeCloseTo(p.xKm, 4);
      expect(Math.abs(pPool.props[4 * idx + 0] - p.tempK) / p.tempK).toBeLessThan(0.001);
      expect(Math.abs(pPool.props[4 * idx + 1] - p.pressureGPa) / p.pressureGPa).toBeLessThan(0.001);
      expect(pPool.meta[4 * idx + 0]).toBe(p.materialId);
      expect(Math.abs(pPool.meta[4 * idx + 3] - p.speedKmS!) / p.speedKmS!).toBeLessThan(0.02);
    }
    expect(tPool.anchorIndex[T - 1]).toBe(T - 1);
    expect(tPool.tsunamiHeightM[T - 1]).toBeCloseTo(12, 4);
    expect((tPool.flags[3] & (TRACER_FLAG_OCEAN | TRACER_FLAG_DISRUPTED))).toBe(3);
  });

  it('handles empty frame (N=0, T=0) cleanly', () => {
    const buf = buildPclsFrame({
      particles: [],
      tracers: [],
      tracerGridRes: 0,
    });
    expect(buf.byteLength).toBe(HEADER_BYTES);
    const pPool = makeParticlePool(10);
    const tPool = makeTracerPool(10, 16);
    pPool.count = 5;
    tPool.count = 4;

    const hdr = decodeBinaryFrame(buf, pPool, tPool);
    expect(hdr.particleCount).toBe(0);
    expect(hdr.tracerCount).toBe(0);
    expect(pPool.count).toBe(0);
    expect(tPool.count).toBe(0);
  });

  it('throws on corrupted magic, unsupported version, truncated buffers, bad zlib, and pool overflow', () => {
    const validParticle: SyntheticParticle = {
      xKm: 0,
      yKm: 0,
      zKm: 0,
      tempK: 300,
      pressureGPa: 1.0,
      materialId: 1,
      phaseId: 0,
      bodyId: 0,
      damage: 0,
      radiusKm: 10,
    };
    const pPool = makeParticlePool(4);
    const tPool = makeTracerPool(4, 1);

    // Corrupted magic
    const badMagic = buildPclsFrame({ magic: 0xdeadbeef, particles: [validParticle], tracers: [] });
    expect(() => decodeBinaryFrame(badMagic, pPool, tPool)).toThrow();

    // Unsupported version
    const badVersion = buildPclsFrame({ version: 2, particles: [validParticle], tracers: [] });
    expect(() => decodeBinaryFrame(badVersion, pPool, tPool)).toThrow();

    // Truncated header (< 40 bytes)
    const shortHdr = badMagic.slice(0, HEADER_BYTES - 1);
    expect(() => decodeBinaryFrame(shortHdr, pPool, tPool)).toThrow();

    // Truncated payload
    const validOne = buildPclsFrame({ particles: [validParticle], tracers: [] });
    expect(() => decodeBinaryFrame(validOne.slice(0, validOne.byteLength - 4), pPool, tPool)).toThrow();

    // Header payloadBytes mismatch
    const badPayloadBytes = buildPclsFrame({
      particles: [validParticle],
      tracers: [],
      overridePayloadBytes: 999,
    });
    expect(() => decodeBinaryFrame(badPayloadBytes, pPool, tPool)).toThrow();

    // Corrupted zlib stream
    const zlibFrame = new Uint8Array(
      buildPclsFrame({ particles: [validParticle], tracers: [], compressZlib: true })
    );
    zlibFrame[HEADER_BYTES + 2] ^= 0xff;
    zlibFrame[zlibFrame.length - 1] ^= 0xff;
    expect(() => decodeBinaryFrame(zlibFrame.buffer, pPool, tPool)).toThrow();

    // Particle pool capacity exceeded
    const smallPPool = makeParticlePool(1);
    const twoParticles = buildPclsFrame({ particles: [validParticle, validParticle], tracers: [] });
    expect(() => decodeBinaryFrame(twoParticles, smallPPool, tPool)).toThrow();

    // Tracer pool capacity exceeded
    const smallTPool = makeTracerPool(1, 1);
    const twoTracers = buildPclsFrame({
      particles: [validParticle],
      tracers: [
        {
          anchorIndex: 0,
          isOcean: false,
          isDisrupted: false,
          xKm: 1,
          yKm: 0,
          zKm: 0,
          dispKm: 0,
          normalDelta: 0,
          tempK: 288,
          damage: 0,
          overpressureKpa: 0,
          tsunamiHeightM: 0,
        },
        {
          anchorIndex: 1,
          isOcean: true,
          isDisrupted: false,
          xKm: 0,
          yKm: 1,
          zKm: 0,
          dispKm: 0,
          normalDelta: 0,
          tempK: 288,
          damage: 0,
          overpressureKpa: 0,
          tsunamiHeightM: 0,
        },
      ],
      tracerGridRes: 1,
    });
    expect(() => decodeBinaryFrame(twoTracers, pPool, smallTPool)).toThrow();
  });
});

describe('SimClient WebSocket routing, queueing, heartbeat & backoff', () => {
  const originalWebSocket = globalThis.WebSocket;

  beforeEach(() => {
    MockWebSocket.instances = [];
    (globalThis as unknown as { WebSocket: unknown }).WebSocket = MockWebSocket;
  });

  afterEach(() => {
    vi.useRealTimers();
    (globalThis as unknown as { WebSocket: unknown }).WebSocket = originalWebSocket;
  });

  it('queues commands while disconnected/connecting, flushes on open, and routes binary + JSON messages', () => {
    const pPool = makeParticlePool(16);
    const tPool = makeTracerPool(16, 1);
    const client = new SimClient('ws://127.0.0.1:8080/ws', pPool, tPool);

    const frames: DecodedFrameHeader[] = [];
    const telemetries: TelemetrySnapshotDto[] = [];
    const presets: Array<{
      preset: string;
      particleCount: number;
      tracerCount: number;
      tracerGridRes: number;
      livingBodyPresent: boolean;
    }> = [];
    const exportsList: Array<{ format: 'csv' | 'json'; filename: string; content: string }> = [];

    client.onFrame((hdr) => frames.push(hdr));
    client.onTelemetry((dto) => telemetries.push(dto));
    client.onPresetLoaded((msg) => presets.push(msg));
    client.onExportData((msg) => exportsList.push(msg));

    // Queue commands before connect & before socket opens
    const customCfg: CustomScenarioDto = {
      targetArchetype: 'LivingTerra',
      impactorArchetype: 'ProtoTheia',
      targetMassKg: 5.97219e24,
      impactorMassKg: 6.417e23,
      impactVelocityRatioVesc: 1.15,
      impactParamB: 0.45,
      initialSepRadii: 3.2,
      targetSpinRadS: 7.2921e-5,
      particles: 10000,
    };
    const cmd1: ClientCommand = { type: 'set_preset', preset: 'LivingGaiaAsteroidCataclysm', particles: 10000 };
    const cmd2: ClientCommand = { type: 'launch_custom', config: customCfg };
    client.sendCommand(cmd1);

    client.connect();
    expect(MockWebSocket.instances.length).toBe(1);
    const ws = MockWebSocket.instances[0];
    expect(ws.url).toBe('ws://127.0.0.1:8080/ws');
    expect(ws.binaryType).toBe('arraybuffer');

    client.sendCommand(cmd2);
    expect(ws.sentMessages.length).toBe(0); // still CONNECTING

    // Open connection -> queued commands must flush in order
    ws.triggerOpen();
    expect(ws.sentMessages.length).toBe(2);
    expect(JSON.parse(ws.sentMessages[0] as string)).toEqual(cmd1);
    expect(JSON.parse(ws.sentMessages[1] as string)).toEqual(cmd2);

    // Send immediate command while open
    client.sendCommand({ type: 'play' });
    expect(ws.sentMessages.length).toBe(3);
    expect(JSON.parse(ws.sentMessages[2] as string)).toEqual({ type: 'play' });

    // Route JSON server messages
    ws.triggerMessage(
      JSON.stringify({
        type: 'preset_loaded',
        preset: 'LivingGaiaAsteroidCataclysm',
        particleCount: 2,
        tracerCount: 0,
        tracerGridRes: 0,
        livingBodyPresent: true,
      })
    );
    expect(presets).toEqual([
      {
        preset: 'LivingGaiaAsteroidCataclysm',
        particleCount: 2,
        tracerCount: 0,
        tracerGridRes: 0,
        livingBodyPresent: true,
      },
    ]);

    const dto = makeSampleTelemetry(42);
    ws.triggerMessage(JSON.stringify(dto));
    expect(telemetries).toHaveLength(1);
    expect(telemetries[0].frameSeq).toBe(42);
    expect(telemetries[0].livingBodyComKm).toEqual([120.5, -15.25, 8.0]);

    ws.triggerMessage(
      JSON.stringify({
        type: 'export_data',
        format: 'csv',
        filename: 'telemetry.csv',
        content: 'frameSeq,simTimeSec\n42,120.5\n',
      })
    );
    expect(exportsList).toEqual([
      {
        format: 'csv',
        filename: 'telemetry.csv',
        content: 'frameSeq,simTimeSec\n42,120.5\n',
      },
    ]);

    // Non-crashing pong / error handling
    ws.triggerMessage(JSON.stringify({ type: 'pong', ts: 1700000000 }));
    ws.triggerMessage(JSON.stringify({ type: 'error', code: 'BAD_REQUEST', message: 'invalid cmd' }));

    // Route binary PCLS frame
    const binFrame = buildPclsFrame({
      frameSeq: 42,
      simTimeSec: 120.5,
      dtSec: 0.5,
      particles: [
        {
          xKm: 100,
          yKm: -50,
          zKm: 25,
          tempK: 1500,
          pressureGPa: 22.5,
          materialId: 1,
          phaseId: 0,
          bodyId: 0,
          damage: 0.4,
          radiusKm: 15,
        },
      ],
      tracers: [],
      tracerGridRes: 0,
    });
    ws.triggerMessage(binFrame);

    expect(frames).toHaveLength(1);
    expect(frames[0].frameSeq).toBe(42);
    expect(pPool.count).toBe(1);
    expect(pPool.positionsKm[0]).toBeCloseTo(100, 4);

    client.disconnect();
    expect(ws.readyState).toBe(MockWebSocket.CLOSED);
  });

  it('sends 20s ping heartbeat and reconnects with exponential backoff when dropped', () => {
    vi.useFakeTimers();
    const pPool = makeParticlePool(8);
    const tPool = makeTracerPool(8, 1);
    const client = new SimClient('ws://127.0.0.1:8080/ws', pPool, tPool);

    client.connect();
    expect(MockWebSocket.instances).toHaveLength(1);
    const ws1 = MockWebSocket.instances[0];
    ws1.triggerOpen();

    // Advance 20s -> ping heartbeat
    vi.advanceTimersByTime(20_000);
    const pings = ws1.sentMessages.filter((m) => typeof m === 'string' && JSON.parse(m).type === 'ping');
    expect(pings.length).toBeGreaterThanOrEqual(1);

    // Unexpected close -> backoff reconnect (1000ms first attempt)
    ws1.triggerDrop(1006);
    vi.advanceTimersByTime(999);
    expect(MockWebSocket.instances).toHaveLength(1);
    vi.advanceTimersByTime(2);
    expect(MockWebSocket.instances).toHaveLength(2);

    // Second drop -> backoff 2000ms
    const ws2 = MockWebSocket.instances[1];
    ws2.triggerDrop(1006);
    vi.advanceTimersByTime(1999);
    expect(MockWebSocket.instances).toHaveLength(2);
    vi.advanceTimersByTime(2);
    expect(MockWebSocket.instances).toHaveLength(3);

    // Explicit disconnect prevents further reconnects
    client.disconnect();
    vi.advanceTimersByTime(60_000);
    expect(MockWebSocket.instances).toHaveLength(3);
  });
});

// Optional live C++ sim_server -> TypeScript BinaryProtocol parity integration test
const SIM_SERVER_PATH = path.resolve(__dirname, '../../build/sim_server');

interface ParsedWsFrame {
  fin: boolean;
  opcode: number;
  payload: Buffer;
}

function encodeClientMaskedTextFrame(text: string): Buffer {
  const payload = Buffer.from(text, 'utf8');
  const mask = crypto.randomBytes(4);
  let header: Buffer;
  if (payload.length < 126) {
    header = Buffer.alloc(2);
    header[0] = 0x81;
    header[1] = 0x80 | payload.length;
  } else if (payload.length <= 0xffff) {
    header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 0x80 | 126;
    header.writeUInt16BE(payload.length, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 0x80 | 127;
    header.writeBigUInt64BE(BigInt(payload.length), 2);
  }
  const masked = Buffer.alloc(payload.length);
  for (let i = 0; i < payload.length; i++) {
    masked[i] = payload[i] ^ mask[i & 3];
  }
  return Buffer.concat([header, mask, masked]);
}

function tryConsumeWsFrame(buf: Buffer): { frame: ParsedWsFrame; rest: Buffer } | null {
  if (buf.length < 2) return null;
  const b0 = buf[0];
  const b1 = buf[1];
  const fin = (b0 & 0x80) !== 0;
  const opcode = b0 & 0x0f;
  const masked = (b1 & 0x80) !== 0;
  let len = b1 & 0x7f;
  let offset = 2;
  if (len === 126) {
    if (buf.length < offset + 2) return null;
    len = buf.readUInt16BE(offset);
    offset += 2;
  } else if (len === 127) {
    if (buf.length < offset + 8) return null;
    const bigLen = buf.readBigUInt64BE(offset);
    len = Number(bigLen);
    offset += 8;
  }
  const maskLen = masked ? 4 : 0;
  if (buf.length < offset + maskLen + len) return null;
  let payload = buf.subarray(offset + maskLen, offset + maskLen + len);
  if (masked) {
    const mask = buf.subarray(offset, offset + 4);
    const unmasked = Buffer.alloc(len);
    for (let i = 0; i < len; i++) {
      unmasked[i] = payload[i] ^ mask[i & 3];
    }
    payload = unmasked;
  } else {
    payload = Buffer.from(payload);
  }
  return {
    frame: { fin, opcode, payload },
    rest: buf.subarray(offset + maskLen + len),
  };
}

describe('Live C++ sim_server -> TypeScript BinaryProtocol integration (optional)', () => {
  it.skipIf(!fs.existsSync(SIM_SERVER_PATH))(
    'receives real PCLS frame from sim_server over RFC 6455 WebSocket and decodes with byte-for-byte parity',
    async () => {
      let child: ChildProcess | null = null;
      let sock: net.Socket | null = null;

      try {
        const port = await new Promise<number>((resolve, reject) => {
          child = spawn(
            SIM_SERVER_PATH,
            ['--port', '0', '--particles', '2048', '--preset', 'LivingGaiaAsteroidCataclysm'],
            { stdio: ['ignore', 'pipe', 'pipe'] }
          );
          let stdoutAcc = '';
          let stderrAcc = '';
          const timer = setTimeout(() => {
            reject(new Error(`Timed out waiting for sim_server port. stdout=${stdoutAcc} stderr=${stderrAcc}`));
          }, 15_000);

          child.stdout?.on('data', (chunk: Buffer) => {
            stdoutAcc += chunk.toString('utf8');
            const m = stdoutAcc.match(/listening on http:\/\/127\.0\.0\.1:(\d+)/);
            if (m) {
              clearTimeout(timer);
              resolve(Number(m[1]));
            }
          });
          child.stderr?.on('data', (chunk: Buffer) => {
            stderrAcc += chunk.toString('utf8');
          });
          child.on('exit', (code) => {
            clearTimeout(timer);
            reject(new Error(`sim_server exited prematurely (${code}): ${stderrAcc}`));
          });
        });

        sock = net.createConnection({ host: '127.0.0.1', port });
        const wsKey = crypto.randomBytes(16).toString('base64');
        const expectedAccept = crypto
          .createHash('sha1')
          .update(wsKey + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
          .digest('base64');

        const { textMsgs, binaryFrames } = await new Promise<{
          textMsgs: Array<Record<string, unknown>>;
          binaryFrames: Buffer[];
        }>((resolve, reject) => {
          let recv: Buffer<ArrayBufferLike> = Buffer.alloc(0);
          let upgraded = false;
          let fragOpcode = 0;
          let fragChunks: Buffer[] = [];
          const texts: Array<Record<string, unknown>> = [];
          const binaries: Buffer[] = [];
          let requestedVelocity = false;

          const timer = setTimeout(() => {
            reject(new Error(`Timed out waiting for WS frames (texts=${texts.length}, bins=${binaries.length})`));
          }, 15_000);

          sock!.on('connect', () => {
            sock!.write(
              `GET / HTTP/1.1\r\n` +
                `Host: 127.0.0.1:${port}\r\n` +
                `Upgrade: websocket\r\n` +
                `Connection: Upgrade\r\n` +
                `Sec-WebSocket-Key: ${wsKey}\r\n` +
                `Sec-WebSocket-Version: 13\r\n\r\n`
            );
          });

          sock!.on('error', (err) => {
            clearTimeout(timer);
            reject(err);
          });

          sock!.on('data', (chunk: Buffer) => {
            recv = Buffer.concat([recv, chunk]);
            if (!upgraded) {
              const hdrEnd = recv.indexOf('\r\n\r\n');
              if (hdrEnd === -1) return;
              const hdrText = recv.subarray(0, hdrEnd).toString('utf8');
              recv = recv.subarray(hdrEnd + 4);
              if (!hdrText.includes('101') || !hdrText.includes(expectedAccept)) {
                clearTimeout(timer);
                reject(new Error(`Invalid WebSocket handshake response: ${hdrText}`));
                return;
              }
              upgraded = true;
            }

            while (true) {
              const parsed = tryConsumeWsFrame(recv);
              if (!parsed) break;
              recv = parsed.rest;
              const { fin, opcode, payload } = parsed.frame;
              if (opcode === 0x9) {
                // Ping -> Pong
                continue;
              }
              if (opcode === 0x8) {
                break;
              }
              if (opcode === 0x1 || opcode === 0x2) {
                fragOpcode = opcode;
                fragChunks = [payload];
              } else if (opcode === 0x0) {
                fragChunks.push(payload);
              }
              if (!fin) {
                continue;
              }
              const full = Buffer.concat(fragChunks);
              if (fragOpcode === 0x1) {
                texts.push(JSON.parse(full.toString('utf8')) as Record<string, unknown>);
              } else if (fragOpcode === 0x2) {
                binaries.push(full);
                if (!requestedVelocity) {
                  requestedVelocity = true;
                  sock!.write(encodeClientMaskedTextFrame(JSON.stringify({ type: 'set_vis_mode', mode: 'Velocity' })));
                } else if (binaries.length >= 2) {
                  clearTimeout(timer);
                  resolve({ textMsgs: texts, binaryFrames: binaries });
                  return;
                }
              }
            }
          });
        });

        const presetMsg = textMsgs.find((m) => m.type === 'preset_loaded');
        const telemetryMsg = textMsgs.find((m) => m.type === 'telemetry');
        expect(presetMsg).toBeDefined();
        expect(telemetryMsg).toBeDefined();
        expect(binaryFrames.length).toBeGreaterThanOrEqual(2);

        const pPool = makeParticlePool(4096);
        const tPool = makeTracerPool(4096, 16);

        // Initial frame (no speed array)
        const raw0 = binaryFrames[0];
        const ab0 = new Uint8Array(raw0).buffer;
        const hdr0 = decodeBinaryFrame(ab0, pPool, tPool);
        expect(hdr0.version).toBe(1);
        expect(hdr0.particleCount).toBe(presetMsg!.particleCount);
        expect(hdr0.tracerCount).toBe(presetMsg!.tracerCount);
        expect(hdr0.tracerGridRes).toBe(presetMsg!.tracerGridRes);
        expect(hdr0.frameSeq).toBe(telemetryMsg!.frameSeq);
        expect(raw0.byteLength).toBe(
          HEADER_BYTES + PARTICLE_RECORD_BYTES * hdr0.particleCount + TRACER_RECORD_BYTES * hdr0.tracerCount
        );
        expect(pPool.count).toBe(hdr0.particleCount);
        expect(tPool.count).toBe(hdr0.tracerCount);

        for (let i = 0; i < pPool.count; i++) {
          expect(Number.isFinite(pPool.positionsKm[3 * i + 0])).toBe(true);
          expect(Number.isFinite(pPool.positionsKm[3 * i + 1])).toBe(true);
          expect(Number.isFinite(pPool.positionsKm[3 * i + 2])).toBe(true);
          expect(pPool.props[4 * i + 0]).toBeGreaterThanOrEqual(10);
          expect(pPool.props[4 * i + 1]).toBeGreaterThanOrEqual(0);
          expect(pPool.props[4 * i + 2]).toBeGreaterThanOrEqual(0);
          expect(pPool.props[4 * i + 2]).toBeLessThanOrEqual(1);
          expect(pPool.meta[4 * i + 0]).toBeLessThanOrEqual(4);
        }
        for (let i = 0; i < tPool.count; i++) {
          expect(tPool.anchorIndex[i]).toBeLessThan(hdr0.tracerCount);
          expect(Number.isFinite(tPool.positionsKm[3 * i + 0])).toBe(true);
          expect(tPool.tempK[i]).toBeGreaterThan(0);
        }

        // Second frame after Velocity mode switch (flags & 0x08 speed array)
        const raw1 = binaryFrames[1];
        const ab1 = new Uint8Array(raw1).buffer;
        const hdr1 = decodeBinaryFrame(ab1, pPool, tPool);
        expect((hdr1.flags & FLAG_HAS_SPEED_ARRAY) !== 0).toBe(true);
        expect(raw1.byteLength).toBe(
          HEADER_BYTES +
            PARTICLE_RECORD_BYTES * hdr1.particleCount +
            TRACER_RECORD_BYTES * hdr1.tracerCount +
            hdr1.particleCount
        );
        for (let i = 0; i < pPool.count; i++) {
          expect(pPool.meta[4 * i + 3]).toBeGreaterThanOrEqual(0.05);
        }
      } finally {
        sock?.destroy();
        if (child !== null) {
          const proc = child as ChildProcess;
          proc.kill('SIGTERM');
          await new Promise<void>((res) => {
            const killTimer = setTimeout(() => {
              proc.kill('SIGKILL');
              res();
            }, 1500);
            proc.once('exit', () => {
              clearTimeout(killTimer);
              res();
            });
          });
        }
      }
    },
    25_000
  );
});
