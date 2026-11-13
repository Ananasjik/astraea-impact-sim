import type { BiomeId } from '../biosphere/BiomeSystem';
import {
  ImpactBiosphereCoupler,
  type BiosphereMetrics,
  type ExtinctionTier,
} from '../biosphere/ImpactBiosphereCoupler';
import type { FaunaGuild } from '../biosphere/ProceduralFauna';
import type { TelemetrySnapshotDto } from '../net/SimClient';
import {
  clampNum,
  ensureHudStyles,
  formatCompactNum,
  formatFixed,
  getOwnerDoc,
  makeEl,
} from './hudUtils';

const MAX_HISTORY_SAMPLES = 240;
const SHANNON_MAX_LN35 = Math.log(35.0);
const DEFAULT_CANVAS_WIDTH = 340;
const DEFAULT_CANVAS_HEIGHT = 128;

interface TelemetryHistorySample {
  simTimeSec: number;
  eKin: number;
  ePot: number;
  eInt: number;
  eRad: number;
  eTot: number;
  driftAbs: number;
  survivalPct: number;
  biomassPct: number;
  shannonNormPct: number;
}

interface SurvivalBarRow {
  fill: HTMLElement;
  pctText: HTMLElement;
}

const BIOME_LIST: ReadonlyArray<{ id: BiomeId; label: string }> = [
  { id: 'TropicalRainforest', label: 'Tropical Forest' },
  { id: 'Savanna', label: 'Savanna' },
  { id: 'TemperateForest', label: 'Temperate Forest' },
  { id: 'BorealTaiga', label: 'Boreal Taiga' },
  { id: 'AlpineTundra', label: 'Alpine Tundra' },
  { id: 'AridSteppe', label: 'Arid Steppe' },
  { id: 'OceanReef', label: 'Ocean & Reef' },
];

const GUILD_LIST: ReadonlyArray<{ id: FaunaGuild | 'Flora'; label: string }> = [
  { id: 'Flora', label: 'Flora' },
  { id: 'HerbivoreHerd', label: 'Herbivores' },
  { id: 'PursuitPredator', label: 'Predators' },
  { id: 'AvianFlock', label: 'Avian Flocks' },
  { id: 'PelagicMarine', label: 'Pelagic Marine' },
];

const ENERGY_SERIES_COLORS = {
  eKin: '#38bdf8',
  ePot: '#a78bfa',
  eInt: '#fb923c',
  eRad: '#f472b6',
  eTot: '#facc15',
  drift: '#34d399',
} as const;

function outcomeBadgeClass(outcome: TelemetrySnapshotDto['outcome']): string {
  switch (outcome) {
    case 'CatastrophicDisruption':
      return 'astraea-pill astraea-pill-danger';
    case 'HitAndRun':
    case 'GrazeAndMerge':
      return 'astraea-pill astraea-pill-warn';
    case 'Merging':
      return 'astraea-pill astraea-pill-info';
    case 'Approaching':
    default:
      return 'astraea-pill astraea-pill-ok';
  }
}

function extinctionBadgeClass(tier: ExtinctionTier): string {
  switch (tier) {
    case 'Sterilization':
      return 'astraea-pill astraea-pill-danger';
    case 'MassExtinction':
      return 'astraea-pill astraea-pill-warn';
    case 'RegionalCrisis':
      return 'astraea-pill astraea-pill-info';
    case 'Nominal':
    default:
      return 'astraea-pill astraea-pill-ok';
  }
}

function survivalBarColor(pct: number): string {
  if (pct >= 85.0) {
    return '#22c55e';
  }
  if (pct >= 45.0) {
    return '#eab308';
  }
  if (pct >= 15.0) {
    return '#f97316';
  }
  return '#ef4444';
}

export class TelemetryCharts {
  private readonly container: HTMLElement;
  private readonly root: HTMLElement;
  private readonly history: TelemetryHistorySample[] = [];

  private readonly outcomePill: HTMLElement;
  private readonly extinctionPill: HTMLElement;
  private readonly qRatioVal: HTMLElement;
  private readonly energyDriftVal: HTMLElement;
  private readonly energyBudgetPill: HTMLElement;
  private readonly virialVal: HTMLElement;
  private readonly massFracsVal: HTMLElement;
  private readonly maxTempVal: HTMLElement;
  private readonly maxPressureVal: HTMLElement;
  private readonly gpuStepVal: HTMLElement;
  private readonly simTimeVal: HTMLElement;
  private readonly biomassVal: HTMLElement;
  private readonly survivalVal: HTMLElement;
  private readonly shannonVal: HTMLElement;
  private readonly opticalTempVal: HTMLElement;

  private readonly energyCanvas: HTMLCanvasElement;
  private readonly bioCanvas: HTMLCanvasElement;
  private readonly biomeBars = new Map<BiomeId, SurvivalBarRow>();
  private readonly guildBars = new Map<FaunaGuild | 'Flora', SurvivalBarRow>();

  constructor(container: HTMLElement) {
    this.container = container;
    const doc = getOwnerDoc(container);
    if (doc === null) {
      throw new Error('TelemetryCharts requires a Document or container.ownerDocument');
    }

    ensureHudStyles(doc);
    this.root = makeEl(doc, 'div', {
      className: 'astraea-telemetry-root',
      testId: 'telemetry-charts',
    });

    // Header + Outcome / ExtinctionTier Pills
    const header = makeEl(doc, 'div', { className: 'astraea-section' });
    const titleRow = makeEl(doc, 'div', { className: 'astraea-section-title' });
    titleRow.appendChild(makeEl(doc, 'span', { text: 'Telemetry & Biosphere' }));
    const pillRow = makeEl(doc, 'div', { className: 'astraea-btn-row' });
    this.outcomePill = makeEl(doc, 'span', {
      className: 'astraea-pill astraea-pill-ok',
      text: 'Approaching',
      testId: 'outcome-pill',
    });
    this.extinctionPill = makeEl(doc, 'span', {
      className: 'astraea-pill astraea-pill-ok',
      text: 'Nominal',
      testId: 'extinction-tier',
    });
    pillRow.appendChild(this.outcomePill);
    pillRow.appendChild(this.extinctionPill);
    header.appendChild(titleRow);
    header.appendChild(pillRow);
    this.root.appendChild(header);

    // Summary Metrics Grid (12 pre-allocated cards, zero per-sample DOM growth)
    const grid = makeEl(doc, 'div', { className: 'astraea-metrics-grid' });

    const makeCard = (
      label: string,
      testId: string,
      extraBadge?: HTMLElement
    ): HTMLElement => {
      const card = makeEl(doc, 'div', { className: 'astraea-metric-card' });
      const lbl = makeEl(doc, 'div', { className: 'astraea-metric-label' });
      lbl.appendChild(makeEl(doc, 'span', { text: label }));
      if (extraBadge !== undefined) {
        lbl.appendChild(extraBadge);
      }
      const val = makeEl(doc, 'div', {
        className: 'astraea-metric-value',
        text: '--',
        testId,
      });
      card.appendChild(lbl);
      card.appendChild(val);
      grid.appendChild(card);
      return val;
    };

    this.energyBudgetPill = makeEl(doc, 'span', {
      className: 'astraea-pill astraea-pill-ok',
      text: '< 1% OK',
      testId: 'energy-budget-status',
    });

    this.qRatioVal = makeCard('Q / Q*RD', 'metric-q-ratio');
    this.energyDriftVal = makeCard(
      'Drift |dE/E0|',
      'metric-energy-drift',
      this.energyBudgetPill
    );
    this.virialVal = makeCard('Virial -2K/U', 'metric-virial');
    this.massFracsVal = makeCard('Bound / Disk / Ejecta', 'metric-mass-fracs');
    this.maxTempVal = makeCard('Peak T_max', 'metric-tmax');
    this.maxPressureVal = makeCard('Peak P_max', 'metric-pmax');
    this.gpuStepVal = makeCard('GPU Step', 'metric-gpu-ms');
    this.simTimeVal = makeCard('Sim Time / Frame', 'metric-sim-time');
    this.biomassVal = makeCard('Biomass (GtC)', 'metric-biomass');
    this.survivalVal = makeCard('Survival %', 'metric-survival-pct');
    this.shannonVal = makeCard("Shannon H'", 'metric-shannon-h');
    this.opticalTempVal = makeCard('Optical τ / dT_glob', 'metric-optical-temp');

    this.root.appendChild(grid);

    // Top Chart: Multi-Series Energy + Drift
    const energyBox = makeEl(doc, 'div', { className: 'astraea-chart-box' });
    const energyHeader = makeEl(doc, 'div', { className: 'astraea-section-title' });
    energyHeader.appendChild(makeEl(doc, 'span', { text: 'Energy & Conservation (<= 240)' }));
    energyBox.appendChild(energyHeader);

    const energyLegend = makeEl(doc, 'div', { className: 'astraea-chart-legend' });
    for (const [label, color] of [
      ['E_kin', ENERGY_SERIES_COLORS.eKin],
      ['E_pot', ENERGY_SERIES_COLORS.ePot],
      ['E_int', ENERGY_SERIES_COLORS.eInt],
      ['E_rad', ENERGY_SERIES_COLORS.eRad],
      ['E_tot', ENERGY_SERIES_COLORS.eTot],
      ['|dE/E0|', ENERGY_SERIES_COLORS.drift],
    ] as const) {
      const item = makeEl(doc, 'span', { text: `■ ${label}` });
      item.style.color = color;
      energyLegend.appendChild(item);
    }
    energyBox.appendChild(energyLegend);

    this.energyCanvas = makeEl(doc, 'canvas', {
      className: 'astraea-chart-canvas',
      testId: 'energy-chart-canvas',
    });
    energyBox.appendChild(this.energyCanvas);
    this.root.appendChild(energyBox);

    // Bottom Chart: Biosphere Extinction Timeline
    const bioBox = makeEl(doc, 'div', { className: 'astraea-chart-box' });
    const bioHeader = makeEl(doc, 'div', { className: 'astraea-section-title' });
    bioHeader.appendChild(makeEl(doc, 'span', { text: 'Biosphere Extinction Timeline' }));
    bioBox.appendChild(bioHeader);

    const bioLegend = makeEl(doc, 'div', { className: 'astraea-chart-legend' });
    for (const [label, color] of [
      ['Survival %', '#22c55e'],
      ['Biomass %', '#38bdf8'],
      ["Shannon H'", '#f59e0b'],
    ] as const) {
      const item = makeEl(doc, 'span', { text: `■ ${label}` });
      item.style.color = color;
      bioLegend.appendChild(item);
    }
    bioBox.appendChild(bioLegend);

    this.bioCanvas = makeEl(doc, 'canvas', {
      className: 'astraea-chart-canvas',
      testId: 'biosphere-chart-canvas',
    });
    bioBox.appendChild(this.bioCanvas);
    this.root.appendChild(bioBox);

    // Biome Survival Bars (x7)
    const biomeSection = makeEl(doc, 'div', { className: 'astraea-section' });
    biomeSection.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-section-title',
        text: 'Biome Survival (7)',
      })
    );
    const biomeGrid = makeEl(doc, 'div', {
      className: 'astraea-bars-grid',
      testId: 'biome-survival-bars',
    });
    for (const b of BIOME_LIST) {
      const row = this.createBarRow(doc, b.label, `biome-bar-${b.id}`);
      this.biomeBars.set(b.id, row.refs);
      biomeGrid.appendChild(row.el);
    }
    biomeSection.appendChild(biomeGrid);
    this.root.appendChild(biomeSection);

    // Guild Survival Bars (x5)
    const guildSection = makeEl(doc, 'div', { className: 'astraea-section' });
    guildSection.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-section-title',
        text: 'Guild Survival (5)',
      })
    );
    const guildGrid = makeEl(doc, 'div', {
      className: 'astraea-bars-grid',
      testId: 'guild-survival-bars',
    });
    for (const g of GUILD_LIST) {
      const row = this.createBarRow(doc, g.label, `guild-bar-${g.id}`);
      this.guildBars.set(g.id, row.refs);
      guildGrid.appendChild(row.el);
    }
    guildSection.appendChild(guildGrid);
    this.root.appendChild(guildSection);

    this.container.appendChild(this.root);
    this.redrawCharts();
  }

  private createBarRow(
    doc: Document,
    label: string,
    testId: string
  ): { el: HTMLElement; refs: SurvivalBarRow } {
    const rowEl = makeEl(doc, 'div', { className: 'astraea-bar-row', testId });
    const nameEl = makeEl(doc, 'span', { className: 'astraea-bar-name', text: label });
    const trackEl = makeEl(doc, 'div', { className: 'astraea-bar-track' });
    const fillEl = makeEl(doc, 'div', { className: 'astraea-bar-fill' });
    fillEl.style.width = '100%';
    trackEl.appendChild(fillEl);
    const pctEl = makeEl(doc, 'span', { className: 'astraea-bar-pct', text: '100.0%' });
    rowEl.appendChild(nameEl);
    rowEl.appendChild(trackEl);
    rowEl.appendChild(pctEl);
    return { el: rowEl, refs: { fill: fillEl, pctText: pctEl } };
  }

  getHistoryLength(): number {
    return this.history.length;
  }

  pushSample(t: TelemetrySnapshotDto, bio: BiosphereMetrics): void {
    const driftAbs = Number.isFinite(t.energyDriftRel)
      ? Math.abs(t.energyDriftRel)
      : 0.0;
    const baseline =
      bio.baselineBiomassGtC > 1.0e-6
        ? bio.baselineBiomassGtC
        : ImpactBiosphereCoupler.BASELINE_BIOMASS_GTC;
    const biomassPct = clampNum((bio.totalBiomassGtC / baseline) * 100.0, 0.0, 100.0);
    const survivalPct = clampNum(bio.overallSurvivalPct, 0.0, 100.0);
    const shannonNormPct = clampNum(
      (bio.shannonDiversityH / SHANNON_MAX_LN35) * 100.0,
      0.0,
      100.0
    );

    if (this.history.length >= MAX_HISTORY_SAMPLES) {
      this.history.shift();
    }
    this.history.push({
      simTimeSec: Number.isFinite(t.simTimeSec) ? t.simTimeSec : 0.0,
      eKin: Number.isFinite(t.kineticEnergyJ) ? t.kineticEnergyJ : 0.0,
      ePot: Number.isFinite(t.potentialEnergyJ) ? t.potentialEnergyJ : 0.0,
      eInt: Number.isFinite(t.internalEnergyJ) ? t.internalEnergyJ : 0.0,
      eRad: Number.isFinite(t.radiatedEnergyJ) ? t.radiatedEnergyJ : 0.0,
      eTot: Number.isFinite(t.totalEnergyJ) ? t.totalEnergyJ : 0.0,
      driftAbs,
      survivalPct,
      biomassPct,
      shannonNormPct,
    });

    // Update summary cards in-place
    this.outcomePill.textContent = t.outcome;
    this.outcomePill.className = outcomeBadgeClass(t.outcome);

    this.extinctionPill.textContent = bio.extinctionTier;
    this.extinctionPill.className = extinctionBadgeClass(bio.extinctionTier);

    this.qRatioVal.textContent = formatFixed(t.qRatioQrd, 3);
    this.energyDriftVal.textContent = `${(driftAbs * 100.0).toFixed(4)}%`;

    const budgetOk = driftAbs < 0.01;
    this.energyBudgetPill.textContent = budgetOk ? '< 1% OK' : '>= 1% WARN';
    this.energyBudgetPill.className = budgetOk
      ? 'astraea-pill astraea-pill-ok'
      : 'astraea-pill astraea-pill-danger';

    this.virialVal.textContent = formatFixed(t.virialRatio, 3);
    this.massFracsVal.textContent = `${(t.boundMassFrac * 100.0).toFixed(1)}% / ${(t.diskMassFrac * 100.0).toFixed(1)}% / ${(t.ejectaMassFrac * 100.0).toFixed(1)}%`;
    this.maxTempVal.textContent = `${formatCompactNum(t.maxTempK, 0)} K`;
    this.maxPressureVal.textContent = `${formatCompactNum(t.maxPressureGPa, 1)} GPa`;
    this.gpuStepVal.textContent = `${formatFixed(t.gpuStepMs, 2)} ms`;
    this.simTimeVal.textContent = `${formatCompactNum(t.simTimeSec, 1)} s · #${t.frameSeq}`;
    this.biomassVal.textContent = `${formatFixed(bio.totalBiomassGtC, 1)} GtC`;
    this.survivalVal.textContent = `${survivalPct.toFixed(1)}%`;
    this.shannonVal.textContent = formatFixed(bio.shannonDiversityH, 3);
    this.opticalTempVal.textContent = `τ ${formatFixed(bio.globalOpticalDepthTau, 2)} · ${formatFixed(bio.globalDeltaTempK, 1)} K`;

    // Update horizontal survival bars in-place
    for (const b of BIOME_LIST) {
      const row = this.biomeBars.get(b.id);
      if (row !== undefined) {
        const pct = clampNum(bio.survivalByBiome[b.id] ?? 0.0, 0.0, 100.0);
        row.fill.style.width = `${pct.toFixed(1)}%`;
        row.fill.style.background = survivalBarColor(pct);
        row.pctText.textContent = `${pct.toFixed(1)}%`;
      }
    }

    for (const g of GUILD_LIST) {
      const row = this.guildBars.get(g.id);
      if (row !== undefined) {
        const pct = clampNum(bio.survivalByGuild[g.id] ?? 0.0, 0.0, 100.0);
        row.fill.style.width = `${pct.toFixed(1)}%`;
        row.fill.style.background = survivalBarColor(pct);
        row.pctText.textContent = `${pct.toFixed(1)}%`;
      }
    }

    this.redrawCharts();
  }

  clear(): void {
    this.history.length = 0;

    this.outcomePill.textContent = 'Approaching';
    this.outcomePill.className = 'astraea-pill astraea-pill-ok';
    this.extinctionPill.textContent = 'Nominal';
    this.extinctionPill.className = 'astraea-pill astraea-pill-ok';
    this.energyBudgetPill.textContent = '< 1% OK';
    this.energyBudgetPill.className = 'astraea-pill astraea-pill-ok';

    this.qRatioVal.textContent = '--';
    this.energyDriftVal.textContent = '--';
    this.virialVal.textContent = '--';
    this.massFracsVal.textContent = '--';
    this.maxTempVal.textContent = '--';
    this.maxPressureVal.textContent = '--';
    this.gpuStepVal.textContent = '--';
    this.simTimeVal.textContent = '--';
    this.biomassVal.textContent = '--';
    this.survivalVal.textContent = '--';
    this.shannonVal.textContent = '--';
    this.opticalTempVal.textContent = '--';

    for (const row of this.biomeBars.values()) {
      row.fill.style.width = '100%';
      row.fill.style.background = '#22c55e';
      row.pctText.textContent = '100.0%';
    }
    for (const row of this.guildBars.values()) {
      row.fill.style.width = '100%';
      row.fill.style.background = '#22c55e';
      row.pctText.textContent = '100.0%';
    }

    this.redrawCharts();
  }

  dispose(): void {
    this.history.length = 0;
    if (this.root.parentNode !== null) {
      this.root.parentNode.removeChild(this.root);
    }
  }

  private prepareCanvas(canvas: HTMLCanvasElement): {
    ctx: CanvasRenderingContext2D;
    width: number;
    height: number;
  } | null {
    const rect =
      typeof canvas.getBoundingClientRect === 'function'
        ? canvas.getBoundingClientRect()
        : null;
    const cssWidth =
      rect !== null && Number.isFinite(rect.width) && rect.width > 0
        ? rect.width
        : DEFAULT_CANVAS_WIDTH;
    const cssHeight =
      rect !== null && Number.isFinite(rect.height) && rect.height > 0
        ? rect.height
        : DEFAULT_CANVAS_HEIGHT;

    const rawDpr =
      typeof window !== 'undefined' && Number.isFinite(window.devicePixelRatio)
        ? window.devicePixelRatio
        : 1;
    const dpr = clampNum(rawDpr, 1, 2);
    const targetW = Math.max(1, Math.round(cssWidth * dpr));
    const targetH = Math.max(1, Math.round(cssHeight * dpr));

    if (canvas.width !== targetW) {
      canvas.width = targetW;
    }
    if (canvas.height !== targetH) {
      canvas.height = targetH;
    }

    const ctx =
      typeof canvas.getContext === 'function' ? canvas.getContext('2d') : null;
    if (ctx === null) {
      return null;
    }

    if (typeof ctx.setTransform === 'function') {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    return { ctx, width: cssWidth, height: cssHeight };
  }

  private drawGrid(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    rows: number
  ): void {
    ctx.fillStyle = 'rgba(4, 8, 18, 0.9)';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
    ctx.lineWidth = 1;
    for (let i = 1; i < rows; i++) {
      const y = (i / rows) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    for (let i = 1; i < 4; i++) {
      const x = (i / 4) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }

  private drawSeries(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    color: string,
    getValueY: (sample: TelemetryHistorySample) => number
  ): void {
    const n = this.history.length;
    if (n === 0) {
      return;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = n === 1 ? width * 0.5 : (i / (n - 1)) * (width - 8) + 4;
      const y = clampNum(getValueY(this.history[i]), 3, height - 3);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }

  private redrawCharts(): void {
    this.redrawEnergyChart();
    this.redrawBioChart();
  }

  private redrawEnergyChart(): void {
    const target = this.prepareCanvas(this.energyCanvas);
    if (target === null) {
      return;
    }
    const { ctx, width, height } = target;
    this.drawGrid(ctx, width, height, 4);

    // Zero energy reference line at midpoint
    const midY = height * 0.5;
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.24)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(width, midY);
    ctx.stroke();

    // 1% relative energy drift budget dashed threshold (mapped onto 0..2% scale from bottom)
    const driftOnePctY = height - (0.01 / 0.02) * (height - 12) - 6;
    ctx.save?.();
    ctx.setLineDash?.([4, 4]);
    ctx.strokeStyle = 'rgba(248, 113, 113, 0.55)';
    ctx.beginPath();
    ctx.moveTo(0, driftOnePctY);
    ctx.lineTo(width, driftOnePctY);
    ctx.stroke();
    ctx.restore?.();

    const n = this.history.length;
    if (n === 0) {
      ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
      ctx.font = '10px monospace';
      ctx.fillText('Awaiting telemetry stream...', 10, midY - 6);
      return;
    }

    let maxAbsE = 1.0;
    for (let i = 0; i < n; i++) {
      const s = this.history[i];
      maxAbsE = Math.max(
        maxAbsE,
        Math.abs(s.eKin),
        Math.abs(s.ePot),
        Math.abs(s.eInt),
        Math.abs(s.eRad),
        Math.abs(s.eTot)
      );
    }

    const halfAmp = height * 0.43;
    this.drawSeries(ctx, width, height, ENERGY_SERIES_COLORS.eKin, (s) => midY - (s.eKin / maxAbsE) * halfAmp);
    this.drawSeries(ctx, width, height, ENERGY_SERIES_COLORS.ePot, (s) => midY - (s.ePot / maxAbsE) * halfAmp);
    this.drawSeries(ctx, width, height, ENERGY_SERIES_COLORS.eInt, (s) => midY - (s.eInt / maxAbsE) * halfAmp);
    this.drawSeries(ctx, width, height, ENERGY_SERIES_COLORS.eRad, (s) => midY - (s.eRad / maxAbsE) * halfAmp);
    this.drawSeries(ctx, width, height, ENERGY_SERIES_COLORS.eTot, (s) => midY - (s.eTot / maxAbsE) * halfAmp);
    this.drawSeries(ctx, width, height, ENERGY_SERIES_COLORS.drift, (s) => {
      const frac = clampNum(s.driftAbs / 0.02, 0.0, 1.0);
      return height - frac * (height - 12) - 6;
    });

    ctx.fillStyle = 'rgba(148, 163, 184, 0.78)';
    ctx.font = '10px monospace';
    ctx.fillText(`±${maxAbsE.toExponential(1)} J`, 6, 12);
    ctx.fillText('1% drift limit', Math.max(6, width - 92), driftOnePctY - 4);
  }

  private redrawBioChart(): void {
    const target = this.prepareCanvas(this.bioCanvas);
    if (target === null) {
      return;
    }
    const { ctx, width, height } = target;
    this.drawGrid(ctx, width, height, 4);

    if (this.history.length === 0) {
      ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
      ctx.font = '10px monospace';
      ctx.fillText('Awaiting biosphere telemetry...', 10, height * 0.5);
      return;
    }

    const pad = 6;
    const span = Math.max(1, height - pad * 2);
    this.drawSeries(
      ctx,
      width,
      height,
      '#22c55e',
      (s) => height - pad - (s.survivalPct / 100.0) * span
    );
    this.drawSeries(
      ctx,
      width,
      height,
      '#38bdf8',
      (s) => height - pad - (s.biomassPct / 100.0) * span
    );
    this.drawSeries(
      ctx,
      width,
      height,
      '#f59e0b',
      (s) => height - pad - (s.shannonNormPct / 100.0) * span
    );

    ctx.fillStyle = 'rgba(148, 163, 184, 0.78)';
    ctx.font = '10px monospace';
    ctx.fillText('100%', 6, 12);
    ctx.fillText('0%', 6, height - 5);
  }
}
