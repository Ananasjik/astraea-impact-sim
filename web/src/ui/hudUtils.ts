export function getOwnerDoc(container: HTMLElement): Document | null {
  if (container && container.ownerDocument) {
    return container.ownerDocument;
  }
  if (typeof document !== 'undefined') {
    return document;
  }
  return null;
}

export interface ElementOptions {
  className?: string;
  id?: string;
  text?: string;
  testId?: string;
  attrs?: Record<string, string>;
}

export function makeEl<K extends keyof HTMLElementTagNameMap>(
  doc: Document,
  tag: K,
  opts?: ElementOptions
): HTMLElementTagNameMap[K] {
  const el = doc.createElement(tag);
  if (opts !== undefined) {
    if (opts.className !== undefined) {
      el.className = opts.className;
    }
    if (opts.id !== undefined) {
      el.id = opts.id;
    }
    if (opts.text !== undefined) {
      el.textContent = opts.text;
    }
    if (typeof el.setAttribute === 'function') {
      if (opts.testId !== undefined) {
        el.setAttribute('data-testid', opts.testId);
      }
      if (opts.attrs !== undefined) {
        for (const [k, v] of Object.entries(opts.attrs)) {
          el.setAttribute(k, v);
        }
      }
    }
  }
  return el;
}

export function clampNum(v: number, minVal: number, maxVal: number): number {
  if (!Number.isFinite(v)) {
    return minVal;
  }
  return v < minVal ? minVal : v > maxVal ? maxVal : v;
}

export function formatFixed(v: number, decimals: number): string {
  if (!Number.isFinite(v)) {
    return '--';
  }
  return v.toFixed(decimals);
}

export function formatCompactNum(v: number, decimals = 1): string {
  if (!Number.isFinite(v)) {
    return '--';
  }
  const abs = Math.abs(v);
  if (abs >= 1.0e6 || (abs > 0.0 && abs < 1.0e-3)) {
    return v.toExponential(2);
  }
  return v.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatAltitude(altitudeKm: number): string {
  if (!Number.isFinite(altitudeKm)) {
    return '--';
  }
  if (altitudeKm < 1.0) {
    const meters = Math.max(0.0, altitudeKm * 1000.0);
    return `${meters.toFixed(1)} m`;
  }
  if (altitudeKm < 100.0) {
    return `${altitudeKm.toFixed(2)} km`;
  }
  return `${Math.round(altitudeKm).toLocaleString('en-US')} km`;
}

const HUD_CSS = `
:root {
  --ast-bg: #030610;
  --ast-panel: rgba(9, 15, 29, 0.80);
  --ast-panel-alt: rgba(15, 23, 42, 0.72);
  --ast-border: rgba(108, 152, 224, 0.22);
  --ast-border-strong: rgba(125, 180, 255, 0.42);
  --ast-text: #e2e8f0;
  --ast-muted: #8fa3c0;
  --ast-accent: #38bdf8;
  --ast-accent-2: #818cf8;
  --ast-good: #22c55e;
  --ast-warn: #f59e0b;
  --ast-danger: #ef4444;
  --ast-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}

.astraea-hud-overlay {
  position: fixed;
  inset: 0;
  pointer-events: none;
  display: grid;
  grid-template-rows: auto 1fr;
  grid-template-columns: 332px 1fr 376px;
  grid-template-areas:
    "top top top"
    "left center right";
  gap: 10px;
  padding: 10px;
  box-sizing: border-box;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ast-text);
  z-index: 10;
}

.astraea-topbar {
  grid-area: top;
  pointer-events: auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 14px;
  border-radius: 10px;
  background: var(--ast-panel);
  border: 1px solid var(--ast-border);
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
}

.astraea-brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.astraea-brand-title {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #f8fafc;
}

.astraea-brand-sub {
  font-size: 11px;
  color: var(--ast-muted);
}

.astraea-status-row {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--ast-mono);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.astraea-panel-left,
.astraea-panel-right {
  pointer-events: auto;
  overflow-y: auto;
  max-height: calc(100vh - 76px);
  border-radius: 10px;
  background: var(--ast-panel);
  border: 1px solid var(--ast-border);
  backdrop-filter: blur(12px);
  box-shadow: 0 10px 32px rgba(0, 0, 0, 0.5);
  scrollbar-width: thin;
  scrollbar-color: rgba(108, 152, 224, 0.35) transparent;
}

.astraea-panel-left {
  grid-area: left;
}

.astraea-panel-right {
  grid-area: right;
}

.astraea-controls-root,
.astraea-telemetry-root {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  box-sizing: border-box;
  font-size: 12px;
}

.astraea-section {
  display: flex;
  flex-direction: column;
  gap: 7px;
  padding: 9px 10px;
  border-radius: 8px;
  background: var(--ast-panel-alt);
  border: 1px solid rgba(148, 163, 184, 0.13);
}

.astraea-section-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--ast-accent);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.astraea-btn-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 5px;
}

.astraea-btn-row {
  display: flex;
  gap: 6px;
  align-items: center;
}

.astraea-btn {
  appearance: none;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: rgba(30, 41, 59, 0.78);
  color: var(--ast-text);
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  text-align: left;
  transition: background 120ms ease, border-color 120ms ease, transform 80ms ease;
}

.astraea-btn:hover {
  background: rgba(51, 65, 85, 0.9);
  border-color: var(--ast-border-strong);
}

.astraea-btn:active {
  transform: translateY(1px);
}

.astraea-btn-center {
  text-align: center;
  flex: 1;
}

.astraea-btn-primary {
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.35), rgba(99, 102, 241, 0.35));
  border-color: rgba(56, 189, 248, 0.55);
  color: #f8fafc;
}

.astraea-btn-active {
  background: rgba(14, 165, 233, 0.28);
  border-color: var(--ast-accent);
  color: #f0f9ff;
}

.astraea-field {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.astraea-field-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: var(--ast-muted);
}

.astraea-field-val {
  font-family: var(--ast-mono);
  font-variant-numeric: tabular-nums;
  color: var(--ast-text);
}

.astraea-select,
.astraea-input {
  width: 100%;
  box-sizing: border-box;
  background: rgba(15, 23, 42, 0.92);
  color: var(--ast-text);
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 5px;
  padding: 5px 7px;
  font-size: 11px;
  font-family: var(--ast-mono);
}

.astraea-range {
  width: 100%;
  accent-color: var(--ast-accent);
  cursor: pointer;
  margin: 2px 0;
}

.astraea-checkbox-label {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 11px;
  cursor: pointer;
  color: var(--ast-text);
}

.astraea-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-family: var(--ast-mono);
  border: 1px solid rgba(148, 163, 184, 0.28);
  background: rgba(30, 41, 59, 0.85);
}

.astraea-pill-ok {
  color: #4ade80;
  border-color: rgba(74, 222, 128, 0.4);
  background: rgba(20, 83, 45, 0.4);
}

.astraea-pill-warn {
  color: #fbbf24;
  border-color: rgba(251, 191, 36, 0.45);
  background: rgba(120, 53, 15, 0.42);
}

.astraea-pill-danger {
  color: #f87171;
  border-color: rgba(248, 113, 113, 0.48);
  background: rgba(127, 29, 29, 0.45);
}

.astraea-pill-info {
  color: #38bdf8;
  border-color: rgba(56, 189, 248, 0.45);
  background: rgba(12, 74, 110, 0.42);
}

.astraea-metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
}

.astraea-metric-card {
  padding: 6px 8px;
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.75);
  border: 1px solid rgba(148, 163, 184, 0.14);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.astraea-metric-label {
  font-size: 10px;
  color: var(--ast-muted);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 4px;
}

.astraea-metric-value {
  font-family: var(--ast-mono);
  font-size: 12px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: #f8fafc;
}

.astraea-chart-box {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px;
  border-radius: 7px;
  background: rgba(6, 11, 22, 0.85);
  border: 1px solid rgba(148, 163, 184, 0.15);
}

.astraea-chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 10px;
  font-family: var(--ast-mono);
  color: var(--ast-muted);
}

.astraea-chart-canvas {
  width: 100%;
  height: 128px;
  display: block;
  border-radius: 4px;
}

.astraea-bars-grid {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.astraea-bar-row {
  display: grid;
  grid-template-columns: 112px 1fr 46px;
  align-items: center;
  gap: 6px;
  font-size: 10px;
}

.astraea-bar-name {
  color: var(--ast-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.astraea-bar-track {
  height: 6px;
  border-radius: 999px;
  background: rgba(51, 65, 85, 0.6);
  overflow: hidden;
}

.astraea-bar-fill {
  height: 100%;
  width: 100%;
  border-radius: 999px;
  background: #22c55e;
  transition: width 120ms ease-out;
}

.astraea-bar-pct {
  text-align: right;
  font-family: var(--ast-mono);
  font-variant-numeric: tabular-nums;
  color: var(--ast-text);
}

.astraea-error-toast {
  color: #fecaca;
  background: rgba(127, 29, 29, 0.8);
  border: 1px solid rgba(248, 113, 113, 0.5);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 11px;
  display: none;
}

@media (max-width: 980px) {
  .astraea-hud-overlay {
    grid-template-columns: 290px 1fr 310px;
    padding: 6px;
    gap: 6px;
  }
}
`;

export function ensureHudStyles(doc: Document | null): void {
  if (doc === null || !doc.head || typeof doc.createElement !== 'function') {
    return;
  }
  if (typeof doc.getElementById === 'function' && doc.getElementById('astraea-hud-styles') !== null) {
    return;
  }
  const styleEl = doc.createElement('style');
  styleEl.id = 'astraea-hud-styles';
  styleEl.textContent = HUD_CSS;
  doc.head.appendChild(styleEl);
}
