import {
  decodeBinaryFrame,
  type DecodedFrameHeader,
  type ParticleBufferPool,
  type TracerBufferPool,
} from './BinaryProtocol';

export interface TelemetrySnapshotDto {
  type: 'telemetry';
  frameSeq: number;
  simTimeSec: number;
  dtSec: number;
  state: 'paused' | 'playing';
  preset: string;
  outcome:
    | 'Approaching'
    | 'Merging'
    | 'GrazeAndMerge'
    | 'HitAndRun'
    | 'CatastrophicDisruption';
  kineticEnergyJ: number;
  potentialEnergyJ: number;
  internalEnergyJ: number;
  radiatedEnergyJ: number;
  totalEnergyJ: number;
  energyDriftRel: number;
  momentumKgMS: [number, number, number];
  angularMomentumKgM2S: [number, number, number];
  virialRatio: number;
  maxTempK: number;
  maxPressureGPa: number;
  boundMassFrac: number;
  diskMassFrac: number;
  ejectaMassFrac: number;
  qRatioQrd: number;
  gpuStepMs: number;
  livingBodyComKm: [number, number, number];
  livingBodyRadiusKm: number;
  livingBodySpinRadS: number;
  impactorComKm: [number, number, number];
  tracerGridRes: number;
  livingBodyPresent: boolean;
}

export interface CustomScenarioDto {
  targetArchetype: string;
  impactorArchetype: string;
  targetMassKg: number;
  impactorMassKg: number;
  impactVelocityRatioVesc: number;
  impactParamB: number;
  initialSepRadii: number;
  targetSpinRadS: number;
  particles: number;
}

export type ClientCommand =
  | { type: 'set_preset'; preset: string; particles?: number }
  | { type: 'launch_custom'; config: CustomScenarioDto }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'step'; count: number }
  | { type: 'set_timescale'; scale: number }
  | {
      type: 'set_slice_plane';
      enabled: boolean;
      normal: [number, number, number];
      offsetKm: number;
    }
  | { type: 'set_vis_mode'; mode: string }
  | { type: 'request_export'; format: 'csv' | 'json' }
  | { type: 'ping'; ts: number };

export interface PresetLoadedDto {
  type?: 'preset_loaded';
  preset: string;
  particleCount: number;
  tracerCount: number;
  tracerGridRes: number;
  livingBodyPresent: boolean;
}

export interface ExportDataDto {
  type?: 'export_data';
  format: 'csv' | 'json';
  filename: string;
  content: string;
}

export interface ServerErrorDto {
  type?: 'error';
  code: string;
  message: string;
}

export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting';

const HEARTBEAT_INTERVAL_MS = 20_000;
const RECONNECT_MIN_DELAY_MS = 1_000;
const RECONNECT_MAX_DELAY_MS = 30_000;
const MAX_QUEUED_COMMANDS = 256;
const WS_CONNECTING = 0;
const WS_OPEN = 1;

export class SimClient {
  private readonly url: string;
  private readonly particlePool: ParticleBufferPool;
  private readonly tracerPool: TracerBufferPool;

  private ws: WebSocket | null = null;
  private explicitDisconnect = false;
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private connectionState: ConnectionState = 'disconnected';
  private readonly commandQueue: ClientCommand[] = [];

  private readonly frameListeners: Array<(header: DecodedFrameHeader) => void> = [];
  private readonly telemetryListeners: Array<(dto: TelemetrySnapshotDto) => void> = [];
  private readonly presetLoadedListeners: Array<(msg: PresetLoadedDto) => void> = [];
  private readonly exportDataListeners: Array<(msg: ExportDataDto) => void> = [];
  private readonly errorListeners: Array<(err: ServerErrorDto) => void> = [];
  private readonly stateListeners: Array<(state: ConnectionState) => void> = [];

  constructor(
    url: string,
    particlePool: ParticleBufferPool,
    tracerPool: TracerBufferPool
  ) {
    this.url = url;
    this.particlePool = particlePool;
    this.tracerPool = tracerPool;
  }

  getState(): ConnectionState {
    return this.connectionState;
  }

  connect(): void {
    const wsConnecting =
      typeof WebSocket !== 'undefined' && typeof WebSocket.CONNECTING === 'number'
        ? WebSocket.CONNECTING
        : WS_CONNECTING;
    const wsOpen =
      typeof WebSocket !== 'undefined' && typeof WebSocket.OPEN === 'number'
        ? WebSocket.OPEN
        : WS_OPEN;
    if (
      this.ws !== null &&
      (this.ws.readyState === wsConnecting || this.ws.readyState === wsOpen)
    ) {
      return;
    }

    this.explicitDisconnect = false;
    this.clearReconnectTimer();
    this.setConnectionState(this.reconnectAttempt > 0 ? 'reconnecting' : 'connecting');

    const socket = new WebSocket(this.url);
    socket.binaryType = 'arraybuffer';
    this.ws = socket;

    socket.onopen = () => {
      if (this.ws !== socket) {
        return;
      }
      this.reconnectAttempt = 0;
      this.setConnectionState('connected');
      this.flushQueue();
      this.startHeartbeat();
    };

    socket.onmessage = (event: MessageEvent) => {
      if (this.ws !== socket) {
        return;
      }
      if (event.data instanceof ArrayBuffer) {
        this.handleBinaryFrame(event.data);
      } else if (typeof event.data === 'string') {
        this.handleJsonMessage(event.data);
      }
    };

    socket.onerror = () => {
      if (this.ws !== socket) {
        return;
      }
      this.emitError({ code: 'WS_ERROR', message: 'WebSocket connection error' });
    };

    socket.onclose = () => {
      if (this.ws !== socket) {
        return;
      }
      this.stopHeartbeat();
      this.ws = null;

      if (this.explicitDisconnect) {
        this.setConnectionState('disconnected');
        return;
      }

      this.scheduleReconnect();
    };
  }

  disconnect(): void {
    this.explicitDisconnect = true;
    this.reconnectAttempt = 0;
    this.clearReconnectTimer();
    this.stopHeartbeat();

    const socket = this.ws;
    this.ws = null;
    if (socket !== null) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      const wsConnecting =
        typeof WebSocket !== 'undefined' && typeof WebSocket.CONNECTING === 'number'
          ? WebSocket.CONNECTING
          : WS_CONNECTING;
      const wsOpen =
        typeof WebSocket !== 'undefined' && typeof WebSocket.OPEN === 'number'
          ? WebSocket.OPEN
          : WS_OPEN;
      if (socket.readyState === wsConnecting || socket.readyState === wsOpen) {
        socket.close(1000, 'Client disconnect');
      }
    }

    this.setConnectionState('disconnected');
  }

  sendCommand(cmd: ClientCommand): void {
    const wsOpen =
      typeof WebSocket !== 'undefined' && typeof WebSocket.OPEN === 'number'
        ? WebSocket.OPEN
        : WS_OPEN;
    if (this.ws !== null && this.ws.readyState === wsOpen) {
      this.ws.send(JSON.stringify(cmd));
      return;
    }

    if (this.commandQueue.length >= MAX_QUEUED_COMMANDS) {
      this.commandQueue.shift();
    }
    this.commandQueue.push(cmd);
  }

  onFrame(cb: (header: DecodedFrameHeader) => void): () => void {
    this.frameListeners.push(cb);
    return () => this.removeListener(this.frameListeners, cb);
  }

  onTelemetry(cb: (dto: TelemetrySnapshotDto) => void): () => void {
    this.telemetryListeners.push(cb);
    return () => this.removeListener(this.telemetryListeners, cb);
  }

  onPresetLoaded(cb: (msg: PresetLoadedDto) => void): () => void {
    this.presetLoadedListeners.push(cb);
    return () => this.removeListener(this.presetLoadedListeners, cb);
  }

  onExportData(cb: (msg: ExportDataDto) => void): () => void {
    this.exportDataListeners.push(cb);
    return () => this.removeListener(this.exportDataListeners, cb);
  }

  onError(cb: (err: ServerErrorDto) => void): () => void {
    this.errorListeners.push(cb);
    return () => this.removeListener(this.errorListeners, cb);
  }

  onConnectionState(cb: (state: ConnectionState) => void): () => void {
    this.stateListeners.push(cb);
    return () => this.removeListener(this.stateListeners, cb);
  }

  private removeListener<T>(arr: Array<(arg: T) => void>, cb: (arg: T) => void): void {
    const idx = arr.indexOf(cb);
    if (idx !== -1) {
      arr.splice(idx, 1);
    }
  }

  private setConnectionState(state: ConnectionState): void {
    if (this.connectionState === state) {
      return;
    }
    this.connectionState = state;
    for (const cb of this.stateListeners) {
      cb(state);
    }
  }

  private handleBinaryFrame(buffer: ArrayBuffer): void {
    try {
      const header = decodeBinaryFrame(buffer, this.particlePool, this.tracerPool);
      for (const cb of this.frameListeners) {
        cb(header);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Binary decode failure';
      this.emitError({ code: 'BINARY_DECODE_ERROR', message });
    }
  }

  private handleJsonMessage(rawText: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(rawText);
    } catch {
      this.emitError({ code: 'INVALID_JSON', message: 'Failed to parse server JSON message' });
      return;
    }

    if (typeof parsed !== 'object' || parsed === null) {
      return;
    }

    const msg = parsed as { type?: string };
    switch (msg.type) {
      case 'telemetry': {
        const dto = parsed as TelemetrySnapshotDto;
        for (const cb of this.telemetryListeners) {
          cb(dto);
        }
        break;
      }
      case 'preset_loaded': {
        const dto = parsed as PresetLoadedDto;
        for (const cb of this.presetLoadedListeners) {
          cb(dto);
        }
        break;
      }
      case 'export_data': {
        const dto = parsed as ExportDataDto;
        for (const cb of this.exportDataListeners) {
          cb(dto);
        }
        break;
      }
      case 'pong':
        break;
      case 'error': {
        const err = parsed as ServerErrorDto;
        this.emitError({
          code: err.code ?? 'SERVER_ERROR',
          message: err.message ?? 'Unknown server error',
        });
        break;
      }
      default:
        break;
    }
  }

  private emitError(err: ServerErrorDto): void {
    for (const cb of this.errorListeners) {
      cb(err);
    }
  }

  private flushQueue(): void {
    const wsOpen =
      typeof WebSocket !== 'undefined' && typeof WebSocket.OPEN === 'number'
        ? WebSocket.OPEN
        : WS_OPEN;
    if (this.ws === null || this.ws.readyState !== wsOpen) {
      return;
    }
    while (this.commandQueue.length > 0 && this.ws.readyState === wsOpen) {
      const next = this.commandQueue.shift();
      if (next !== undefined) {
        this.ws.send(JSON.stringify(next));
      }
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    const wsOpen =
      typeof WebSocket !== 'undefined' && typeof WebSocket.OPEN === 'number'
        ? WebSocket.OPEN
        : WS_OPEN;
    this.heartbeatTimer = setInterval(() => {
      if (this.ws !== null && this.ws.readyState === wsOpen) {
        this.sendCommand({ type: 'ping', ts: Date.now() });
      }
    }, HEARTBEAT_INTERVAL_MS);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private scheduleReconnect(): void {
    this.clearReconnectTimer();
    this.setConnectionState('reconnecting');
    const delayMs = Math.min(
      RECONNECT_MIN_DELAY_MS * Math.pow(2, this.reconnectAttempt),
      RECONNECT_MAX_DELAY_MS
    );
    this.reconnectAttempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.explicitDisconnect) {
        this.connect();
      }
    }, delayMs);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
