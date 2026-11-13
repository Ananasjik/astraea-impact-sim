import type { BiosphereMetrics } from '../biosphere/ImpactBiosphereCoupler';
import type { CustomScenarioDto, TelemetrySnapshotDto } from '../net/SimClient';
import type { CameraMode, SurfaceAnchorCoord } from '../render/OrbitalSurfaceCamera';
import type { VisMode } from '../render/SphParticleRenderer';
import {
  clampNum,
  ensureHudStyles,
  formatAltitude,
  formatFixed,
  getOwnerDoc,
  makeEl,
} from './hudUtils';

export interface ControlPanelCallbacks {
  onPreset: (preset: string, particles: number) => void;
  onLaunchCustom: (cfg: CustomScenarioDto) => void;
  onPlayPause: (playing: boolean) => void;
  onStep: () => void;
  onTimescale: (scale: number) => void;
  onVisMode: (mode: VisMode) => void;
  onSliceChange: (
    enabled: boolean,
    normal: [number, number, number],
    offsetKm: number
  ) => void;
  onCameraMode: (mode: CameraMode, anchor?: SurfaceAnchorCoord) => void;
  onExport: (format: 'csv' | 'json') => void;
}

const EARTH_MASS_KG = 5.9722e24;

const PRESETS: ReadonlyArray<{ id: string; label: string }> = [
  { id: 'LivingGaiaAsteroidCataclysm', label: 'Living Gaia Cataclysm' },
  { id: 'EarthTheiaMoonForming', label: 'Earth–Theia Moon' },
  { id: 'MercuryMantleStrip', label: 'Mercury Mantle Strip' },
  { id: 'OceanWorldSteamPlume', label: 'Ocean World Steam' },
  { id: 'GasGiantCoreImpact', label: 'Gas Giant Core Impact' },
];

const PARTICLE_COUNTS: readonly number[] = [10000, 50000, 100000, 200000];

const ARCHETYPES: readonly string[] = [
  'LivingTerra',
  'IronCore',
  'OceanIce',
  'LavaIo',
  'GasGiant',
  'ProtoTheia',
];

const VIS_MODES: ReadonlyArray<{ id: VisMode; label: string }> = [
  { id: 'NaturalBlackbody', label: 'NaturalBlackbody' },
  { id: 'TemperatureK', label: 'TemperatureK' },
  { id: 'PressureGPa', label: 'PressureGPa' },
  { id: 'MaterialLayer', label: 'MaterialLayer' },
  { id: 'DamageFracture', label: 'DamageFracture' },
  { id: 'Velocity', label: 'Velocity' },
];

const SLICE_NORMALS: ReadonlyArray<{
  key: string;
  label: string;
  normal: [number, number, number];
}> = [
  { key: '0,1,0', label: 'Y Axis [0,1,0]', normal: [0, 1, 0] },
  { key: '1,0,0', label: 'X Axis [1,0,0]', normal: [1, 0, 0] },
  { key: '0,0,1', label: 'Z Axis [0,0,1]', normal: [0, 0, 1] },
];

export const BIOME_ANCHOR_PRESETS: ReadonlyArray<{
  id: string;
  label: string;
  anchor: SurfaceAnchorCoord;
}> = [
  {
    id: 'tropical-rainforest',
    label: 'Tropical Rainforest',
    anchor: { face: 0, u: 0.05, v: 0.6 },
  },
  {
    id: 'coastal-coral-reef',
    label: 'Coastal Coral Reef',
    anchor: { face: 0, u: 0.1, v: 0.6 },
  },
  {
    id: 'savanna-herd',
    label: 'Savanna Herd',
    anchor: { face: 0, u: 0.05, v: 0.3 },
  },
  {
    id: 'boreal-taiga',
    label: 'Boreal Taiga',
    anchor: { face: 2, u: 0.2, v: 0.35 },
  },
  {
    id: 'alpine-ridge',
    label: 'Alpine Ridge',
    anchor: { face: 0, u: 0.05, v: 0.45 },
  },
  {
    id: 'impactor-crater-rim',
    label: 'Impactor Crater Rim',
    anchor: { face: 0, u: 0.72, v: 0.52 },
  },
];

export class ControlPanel {
  private readonly callbacks: ControlPanelCallbacks;
  private readonly root: HTMLElement;
  private readonly cleanups: Array<() => void> = [];

  private isPlaying = false;
  private cameraMode: CameraMode = 'OrbitalFree';
  private activePreset = 'LivingGaiaAsteroidCataclysm';

  private readonly presetButtons = new Map<string, HTMLButtonElement>();
  private readonly particleSelect: HTMLSelectElement;
  private readonly playPauseBtn: HTMLButtonElement;
  private readonly timescaleSlider: HTMLInputElement;
  private readonly timescaleValLabel: HTMLElement;
  private readonly visModeSelect: HTMLSelectElement;
  private readonly sliceEnabledCheckbox: HTMLInputElement;
  private readonly sliceNormalSelect: HTMLSelectElement;
  private readonly sliceOffsetSlider: HTMLInputElement;
  private readonly sliceOffsetValLabel: HTMLElement;
  private readonly cameraModeToggleBtn: HTMLButtonElement;
  private readonly altitudeReadout: HTMLElement;
  private readonly bioStateBadge: HTMLElement;

  // Custom scenario fields
  private readonly targetArchSelect: HTMLSelectElement;
  private readonly impactorArchSelect: HTMLSelectElement;
  private readonly targetMassInput: HTMLInputElement;
  private readonly massRatioInput: HTMLInputElement;
  private readonly impactorMassInput: HTMLInputElement;
  private readonly vRatioSlider: HTMLInputElement;
  private readonly vRatioValLabel: HTMLElement;
  private readonly impactBSlider: HTMLInputElement;
  private readonly impactBValLabel: HTMLElement;
  private readonly initialSepSlider: HTMLInputElement;
  private readonly initialSepValLabel: HTMLElement;
  private readonly targetSpinInput: HTMLInputElement;

  constructor(container: HTMLElement, callbacks: ControlPanelCallbacks) {
    this.callbacks = callbacks;
    const doc = getOwnerDoc(container);
    if (doc === null) {
      throw new Error('ControlPanel requires a Document or container.ownerDocument');
    }

    ensureHudStyles(doc);
    this.root = makeEl(doc, 'div', {
      className: 'astraea-controls-root',
      testId: 'control-panel',
    });

    // 1. Scenario Presets & Particle Budget
    const presetSec = makeEl(doc, 'div', { className: 'astraea-section' });
    presetSec.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-section-title',
        text: 'Collision Presets',
      })
    );

    const pCountRow = makeEl(doc, 'div', { className: 'astraea-field-row' });
    pCountRow.appendChild(makeEl(doc, 'span', { text: 'Particles' }));
    this.particleSelect = makeEl(doc, 'select', {
      className: 'astraea-select',
      id: 'astraea-particle-count',
      testId: 'particle-count-select',
    });
    this.particleSelect.style.width = '128px';
    for (const count of PARTICLE_COUNTS) {
      const opt = makeEl(doc, 'option', {
        text: count.toLocaleString('en-US'),
        attrs: { value: String(count) },
      });
      if (count === 100000) {
        opt.selected = true;
      }
      this.particleSelect.appendChild(opt);
    }
    this.particleSelect.value = '100000';
    pCountRow.appendChild(this.particleSelect);
    presetSec.appendChild(pCountRow);

    const presetGrid = makeEl(doc, 'div', { className: 'astraea-btn-grid' });
    for (const p of PRESETS) {
      const btn = makeEl(doc, 'button', {
        className:
          p.id === this.activePreset
            ? 'astraea-btn astraea-btn-active'
            : 'astraea-btn',
        text: p.label,
        testId: `preset-${p.id}`,
        attrs: {
          type: 'button',
          'data-preset': p.id,
          title: p.id,
        },
      });
      this.listen(btn, 'click', () => {
        this.setActivePresetButton(p.id);
        this.callbacks.onPreset(p.id, this.getSelectedParticleCount());
      });
      this.presetButtons.set(p.id, btn);
      presetGrid.appendChild(btn);
    }
    presetSec.appendChild(presetGrid);
    this.root.appendChild(presetSec);

    // 2. Transport Controls
    const transportSec = makeEl(doc, 'div', { className: 'astraea-section' });
    transportSec.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-section-title',
        text: 'Simulation Transport',
      })
    );
    const transportRow = makeEl(doc, 'div', { className: 'astraea-btn-row' });
    this.playPauseBtn = makeEl(doc, 'button', {
      className: 'astraea-btn astraea-btn-primary astraea-btn-center',
      text: 'Play',
      testId: 'play-pause-btn',
      attrs: { type: 'button', 'aria-pressed': 'false' },
    });
    const stepBtn = makeEl(doc, 'button', {
      className: 'astraea-btn astraea-btn-center',
      text: 'Step +1',
      testId: 'step-btn',
      attrs: { type: 'button' },
    });
    this.listen(this.playPauseBtn, 'click', () => {
      this.isPlaying = !this.isPlaying;
      this.syncPlayPauseButton();
      this.callbacks.onPlayPause(this.isPlaying);
    });
    this.listen(stepBtn, 'click', () => {
      this.callbacks.onStep();
    });
    transportRow.appendChild(this.playPauseBtn);
    transportRow.appendChild(stepBtn);
    transportSec.appendChild(transportRow);

    const tsField = makeEl(doc, 'div', { className: 'astraea-field' });
    const tsRow = makeEl(doc, 'div', { className: 'astraea-field-row' });
    tsRow.appendChild(makeEl(doc, 'span', { text: 'Time Scale' }));
    this.timescaleValLabel = makeEl(doc, 'span', {
      className: 'astraea-field-val',
      text: '1.0x',
      testId: 'timescale-readout',
    });
    tsRow.appendChild(this.timescaleValLabel);
    this.timescaleSlider = makeEl(doc, 'input', {
      className: 'astraea-range',
      testId: 'timescale-slider',
      attrs: {
        type: 'range',
        min: '0.1',
        max: '100.0',
        step: '0.1',
        value: '1.0',
      },
    });
    this.timescaleSlider.value = '1.0';
    let lastEmittedTimescale = Number.NaN;
    const emitTimescale = (): void => {
      const val = clampNum(Number.parseFloat(this.timescaleSlider.value), 0.1, 100.0);
      this.timescaleValLabel.textContent = `${val.toFixed(1)}x`;
      if (val !== lastEmittedTimescale) {
        lastEmittedTimescale = val;
        this.callbacks.onTimescale(val);
      }
    };
    this.listen(this.timescaleSlider, 'input', emitTimescale);
    this.listen(this.timescaleSlider, 'change', emitTimescale);
    tsField.appendChild(tsRow);
    tsField.appendChild(this.timescaleSlider);
    transportSec.appendChild(tsField);
    this.root.appendChild(transportSec);

    // 3. Visualization & Cross-Section Slice
    const visSec = makeEl(doc, 'div', { className: 'astraea-section' });
    visSec.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-section-title',
        text: 'Visualization & Slice',
      })
    );

    const visRow = makeEl(doc, 'div', { className: 'astraea-field' });
    visRow.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-field-row',
        text: 'Shader Mode',
      })
    );
    this.visModeSelect = makeEl(doc, 'select', {
      className: 'astraea-select',
      testId: 'vis-mode-select',
    });
    for (const vm of VIS_MODES) {
      const opt = makeEl(doc, 'option', {
        text: vm.label,
        attrs: { value: vm.id },
      });
      if (vm.id === 'NaturalBlackbody') {
        opt.selected = true;
      }
      this.visModeSelect.appendChild(opt);
    }
    this.visModeSelect.value = 'NaturalBlackbody';
    this.listen(this.visModeSelect, 'change', () => {
      const selected = this.visModeSelect.value as VisMode;
      this.callbacks.onVisMode(selected);
    });
    visRow.appendChild(this.visModeSelect);
    visSec.appendChild(visRow);

    const sliceCheckLabel = makeEl(doc, 'label', {
      className: 'astraea-checkbox-label',
    });
    this.sliceEnabledCheckbox = makeEl(doc, 'input', {
      testId: 'slice-enabled-checkbox',
      attrs: { type: 'checkbox' },
    });
    this.sliceEnabledCheckbox.checked = false;
    sliceCheckLabel.appendChild(this.sliceEnabledCheckbox);
    sliceCheckLabel.appendChild(makeEl(doc, 'span', { text: 'Enable Cross-Section Slice' }));
    visSec.appendChild(sliceCheckLabel);

    const sliceNormField = makeEl(doc, 'div', { className: 'astraea-field' });
    sliceNormField.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-field-row',
        text: 'Slice Normal',
      })
    );
    this.sliceNormalSelect = makeEl(doc, 'select', {
      className: 'astraea-select',
      testId: 'slice-normal-select',
    });
    for (const sn of SLICE_NORMALS) {
      const opt = makeEl(doc, 'option', {
        text: sn.label,
        attrs: { value: sn.key },
      });
      if (sn.key === '0,1,0') {
        opt.selected = true;
      }
      this.sliceNormalSelect.appendChild(opt);
    }
    this.sliceNormalSelect.value = '0,1,0';
    sliceNormField.appendChild(this.sliceNormalSelect);
    visSec.appendChild(sliceNormField);

    const sliceOffField = makeEl(doc, 'div', { className: 'astraea-field' });
    const sliceOffRow = makeEl(doc, 'div', { className: 'astraea-field-row' });
    sliceOffRow.appendChild(makeEl(doc, 'span', { text: 'Slice Offset' }));
    this.sliceOffsetValLabel = makeEl(doc, 'span', {
      className: 'astraea-field-val',
      text: '0 km',
      testId: 'slice-offset-readout',
    });
    sliceOffRow.appendChild(this.sliceOffsetValLabel);
    this.sliceOffsetSlider = makeEl(doc, 'input', {
      className: 'astraea-range',
      testId: 'slice-offset-slider',
      attrs: {
        type: 'range',
        min: '-8000',
        max: '8000',
        step: '100',
        value: '0',
      },
    });
    this.sliceOffsetSlider.value = '0';
    sliceOffField.appendChild(sliceOffRow);
    sliceOffField.appendChild(this.sliceOffsetSlider);
    visSec.appendChild(sliceOffField);

    let lastSliceKey = '';
    const emitSlice = (force: boolean): void => {
      const enabled = Boolean(this.sliceEnabledCheckbox.checked);
      const normal = this.parseSliceNormal();
      const offsetKm = clampNum(
        Number.parseFloat(this.sliceOffsetSlider.value),
        -100000,
        100000
      );
      this.sliceOffsetValLabel.textContent = `${Math.round(offsetKm)} km`;
      const key = `${enabled}|${normal[0]},${normal[1]},${normal[2]}|${offsetKm}`;
      if (force || key !== lastSliceKey) {
        lastSliceKey = key;
        this.callbacks.onSliceChange(enabled, normal, offsetKm);
      }
    };
    this.listen(this.sliceEnabledCheckbox, 'change', () => emitSlice(true));
    this.listen(this.sliceNormalSelect, 'change', () => emitSlice(true));
    this.listen(this.sliceOffsetSlider, 'input', () => emitSlice(false));
    this.listen(this.sliceOffsetSlider, 'change', () => emitSlice(false));
    this.root.appendChild(visSec);

    // 4. Camera Orbital <-> Surface Biosphere Zoom
    const camSec = makeEl(doc, 'div', { className: 'astraea-section' });
    const camHeader = makeEl(doc, 'div', { className: 'astraea-section-title' });
    camHeader.appendChild(makeEl(doc, 'span', { text: 'Camera & Surface Biomes' }));
    this.bioStateBadge = makeEl(doc, 'span', {
      className: 'astraea-pill astraea-pill-ok',
      text: 'Nominal',
      testId: 'panel-biosphere-badge',
    });
    camHeader.appendChild(this.bioStateBadge);
    camSec.appendChild(camHeader);

    const altRow = makeEl(doc, 'div', { className: 'astraea-field-row' });
    altRow.appendChild(makeEl(doc, 'span', { text: 'Altitude' }));
    this.altitudeReadout = makeEl(doc, 'span', {
      className: 'astraea-field-val',
      text: '18,000 km',
      testId: 'camera-altitude-readout',
    });
    altRow.appendChild(this.altitudeReadout);
    camSec.appendChild(altRow);

    this.cameraModeToggleBtn = makeEl(doc, 'button', {
      className: 'astraea-btn astraea-btn-primary astraea-btn-center',
      text: 'Mode: OrbitalFree (Switch to SurfaceLock)',
      testId: 'camera-mode-toggle',
      attrs: { type: 'button', 'data-camera-mode': 'OrbitalFree' },
    });
    this.listen(this.cameraModeToggleBtn, 'click', () => {
      const nextMode: CameraMode =
        this.cameraMode === 'OrbitalFree' ? 'SurfaceLock' : 'OrbitalFree';
      this.setCameraModeDisplay(nextMode);
      this.callbacks.onCameraMode(nextMode);
    });
    camSec.appendChild(this.cameraModeToggleBtn);

    const biomeAnchorGrid = makeEl(doc, 'div', { className: 'astraea-btn-grid' });
    for (const bp of BIOME_ANCHOR_PRESETS) {
      const btn = makeEl(doc, 'button', {
        className: 'astraea-btn',
        text: bp.label,
        testId: `anchor-${bp.id}`,
        attrs: { type: 'button', 'data-biome-anchor': bp.id },
      });
      this.listen(btn, 'click', () => {
        this.setCameraModeDisplay('SurfaceLock');
        this.callbacks.onCameraMode('SurfaceLock', { ...bp.anchor });
      });
      biomeAnchorGrid.appendChild(btn);
    }
    camSec.appendChild(biomeAnchorGrid);
    this.root.appendChild(camSec);

    // 5. Custom Impactor Constructor
    const customSec = makeEl(doc, 'div', { className: 'astraea-section' });
    customSec.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-section-title',
        text: 'Custom Impactor Constructor',
      })
    );

    this.targetArchSelect = this.buildArchetypeSelect(
      doc,
      'LivingTerra',
      'custom-target-archetype'
    );
    this.impactorArchSelect = this.buildArchetypeSelect(
      doc,
      'ProtoTheia',
      'custom-impactor-archetype'
    );

    const archGrid = makeEl(doc, 'div', { className: 'astraea-btn-grid' });
    const tArchWrap = makeEl(doc, 'div', { className: 'astraea-field' });
    tArchWrap.appendChild(
      makeEl(doc, 'div', { className: 'astraea-field-row', text: 'Target' })
    );
    tArchWrap.appendChild(this.targetArchSelect);
    const iArchWrap = makeEl(doc, 'div', { className: 'astraea-field' });
    iArchWrap.appendChild(
      makeEl(doc, 'div', { className: 'astraea-field-row', text: 'Impactor' })
    );
    iArchWrap.appendChild(this.impactorArchSelect);
    archGrid.appendChild(tArchWrap);
    archGrid.appendChild(iArchWrap);
    customSec.appendChild(archGrid);

    const massGrid = makeEl(doc, 'div', { className: 'astraea-btn-grid' });
    const tMassWrap = makeEl(doc, 'div', { className: 'astraea-field' });
    tMassWrap.appendChild(
      makeEl(doc, 'div', { className: 'astraea-field-row', text: 'Target Mass (kg)' })
    );
    this.targetMassInput = makeEl(doc, 'input', {
      className: 'astraea-input',
      testId: 'custom-target-mass-kg',
      attrs: {
        type: 'number',
        step: 'any',
        min: '1e20',
        value: String(EARTH_MASS_KG),
      },
    });
    this.targetMassInput.value = String(EARTH_MASS_KG);
    tMassWrap.appendChild(this.targetMassInput);

    const ratioWrap = makeEl(doc, 'div', { className: 'astraea-field' });
    ratioWrap.appendChild(
      makeEl(doc, 'div', { className: 'astraea-field-row', text: 'Mass Ratio (M_i/M_t)' })
    );
    this.massRatioInput = makeEl(doc, 'input', {
      className: 'astraea-input',
      testId: 'custom-mass-ratio',
      attrs: {
        type: 'number',
        step: '0.005',
        min: '0.001',
        max: '2.0',
        value: '0.05',
      },
    });
    this.massRatioInput.value = '0.05';
    ratioWrap.appendChild(this.massRatioInput);
    massGrid.appendChild(tMassWrap);
    massGrid.appendChild(ratioWrap);
    customSec.appendChild(massGrid);

    const iMassWrap = makeEl(doc, 'div', { className: 'astraea-field' });
    iMassWrap.appendChild(
      makeEl(doc, 'div', { className: 'astraea-field-row', text: 'Impactor Mass (kg)' })
    );
    this.impactorMassInput = makeEl(doc, 'input', {
      className: 'astraea-input',
      testId: 'custom-impactor-mass-kg',
      attrs: {
        type: 'number',
        step: 'any',
        min: '1e18',
        value: String(EARTH_MASS_KG * 0.05),
      },
    });
    this.impactorMassInput.value = String(EARTH_MASS_KG * 0.05);
    iMassWrap.appendChild(this.impactorMassInput);
    customSec.appendChild(iMassWrap);

    const syncImpactorFromRatio = (): void => {
      const mT = this.readPositiveNumber(this.targetMassInput.value, EARTH_MASS_KG);
      const ratio = clampNum(
        this.readPositiveNumber(this.massRatioInput.value, 0.05),
        0.001,
        5.0
      );
      this.impactorMassInput.value = (mT * ratio).toExponential(4);
    };
    const syncRatioFromImpactor = (): void => {
      const mT = this.readPositiveNumber(this.targetMassInput.value, EARTH_MASS_KG);
      const mI = this.readPositiveNumber(
        this.impactorMassInput.value,
        EARTH_MASS_KG * 0.05
      );
      this.massRatioInput.value = (mI / mT).toFixed(4);
    };
    this.listen(this.targetMassInput, 'input', syncImpactorFromRatio);
    this.listen(this.targetMassInput, 'change', syncImpactorFromRatio);
    this.listen(this.massRatioInput, 'input', syncImpactorFromRatio);
    this.listen(this.massRatioInput, 'change', syncImpactorFromRatio);
    this.listen(this.impactorMassInput, 'input', syncRatioFromImpactor);
    this.listen(this.impactorMassInput, 'change', syncRatioFromImpactor);

    const vBuild = this.buildRangeField(
      doc,
      'v / v_esc',
      'custom-v-vesc',
      0.5,
      4.0,
      0.05,
      1.65,
      (v) => v.toFixed(2)
    );
    this.vRatioSlider = vBuild.input;
    this.vRatioValLabel = vBuild.valLabel;
    customSec.appendChild(vBuild.wrapper);

    const bBuild = this.buildRangeField(
      doc,
      'Impact Param b',
      'custom-impact-b',
      0.0,
      1.2,
      0.02,
      0.3,
      (v) => v.toFixed(2)
    );
    this.impactBSlider = bBuild.input;
    this.impactBValLabel = bBuild.valLabel;
    customSec.appendChild(bBuild.wrapper);

    const sepBuild = this.buildRangeField(
      doc,
      'Initial Sep (R_t+R_i)',
      'custom-initial-sep',
      1.5,
      6.0,
      0.1,
      3.2,
      (v) => `${v.toFixed(1)} R`
    );
    this.initialSepSlider = sepBuild.input;
    this.initialSepValLabel = sepBuild.valLabel;
    customSec.appendChild(sepBuild.wrapper);

    const spinWrap = makeEl(doc, 'div', { className: 'astraea-field-row' });
    spinWrap.appendChild(makeEl(doc, 'span', { text: 'Target Spin (rad/s)' }));
    this.targetSpinInput = makeEl(doc, 'input', {
      className: 'astraea-input',
      testId: 'custom-target-spin',
      attrs: {
        type: 'number',
        step: '0.00001',
        min: '-0.002',
        max: '0.002',
        value: '0.0000729',
      },
    });
    this.targetSpinInput.style.width = '128px';
    this.targetSpinInput.value = '0.0000729';
    spinWrap.appendChild(this.targetSpinInput);
    customSec.appendChild(spinWrap);

    const launchBtn = makeEl(doc, 'button', {
      className: 'astraea-btn astraea-btn-primary astraea-btn-center',
      text: 'Launch Custom Collision',
      testId: 'launch-custom-btn',
      attrs: { type: 'button' },
    });
    this.listen(launchBtn, 'click', () => {
      const targetMassKg = this.readPositiveNumber(
        this.targetMassInput.value,
        EARTH_MASS_KG
      );
      const impactorMassKg = Math.max(
        0.0,
        this.readPositiveNumber(this.impactorMassInput.value, EARTH_MASS_KG * 0.05)
      );
      const impactVelocityRatioVesc = clampNum(
        Number.parseFloat(this.vRatioSlider.value),
        0.5,
        4.0
      );
      const impactParamB = clampNum(
        Number.parseFloat(this.impactBSlider.value),
        0.0,
        1.2
      );
      const initialSepRadii = clampNum(
        Number.parseFloat(this.initialSepSlider.value),
        1.5,
        6.0
      );
      const rawSpin = Number.parseFloat(this.targetSpinInput.value);
      const targetSpinRadS = Number.isFinite(rawSpin) ? rawSpin : 0.0;

      this.callbacks.onLaunchCustom({
        targetArchetype: this.targetArchSelect.value || 'LivingTerra',
        impactorArchetype: this.impactorArchSelect.value || 'ProtoTheia',
        targetMassKg,
        impactorMassKg,
        impactVelocityRatioVesc,
        impactParamB,
        initialSepRadii,
        targetSpinRadS,
        particles: this.getSelectedParticleCount(),
      });
    });
    customSec.appendChild(launchBtn);
    this.root.appendChild(customSec);

    // 6. Scientific Export
    const exportSec = makeEl(doc, 'div', { className: 'astraea-section' });
    exportSec.appendChild(
      makeEl(doc, 'div', {
        className: 'astraea-section-title',
        text: 'Scientific Data Export',
      })
    );
    const exportRow = makeEl(doc, 'div', { className: 'astraea-btn-row' });
    const csvBtn = makeEl(doc, 'button', {
      className: 'astraea-btn astraea-btn-center',
      text: 'Export CSV',
      testId: 'export-csv-btn',
      attrs: { type: 'button', 'data-export': 'csv' },
    });
    const jsonBtn = makeEl(doc, 'button', {
      className: 'astraea-btn astraea-btn-center',
      text: 'Export JSON',
      testId: 'export-json-btn',
      attrs: { type: 'button', 'data-export': 'json' },
    });
    this.listen(csvBtn, 'click', () => this.callbacks.onExport('csv'));
    this.listen(jsonBtn, 'click', () => this.callbacks.onExport('json'));
    exportRow.appendChild(csvBtn);
    exportRow.appendChild(jsonBtn);
    exportSec.appendChild(exportRow);
    this.root.appendChild(exportSec);

    container.appendChild(this.root);
  }

  updateState(telemetry: TelemetrySnapshotDto, bio: BiosphereMetrics): void {
    this.isPlaying = telemetry.state === 'playing';
    this.syncPlayPauseButton();

    if (telemetry.preset && this.presetButtons.has(telemetry.preset)) {
      this.setActivePresetButton(telemetry.preset);
    }

    this.bioStateBadge.textContent = `${bio.extinctionTier} (${formatFixed(bio.overallSurvivalPct, 0)}%)`;
    this.bioStateBadge.className =
      bio.extinctionTier === 'Sterilization'
        ? 'astraea-pill astraea-pill-danger'
        : bio.extinctionTier === 'MassExtinction'
          ? 'astraea-pill astraea-pill-warn'
          : bio.extinctionTier === 'RegionalCrisis'
            ? 'astraea-pill astraea-pill-info'
            : 'astraea-pill astraea-pill-ok';

    // Sync custom slider readouts in case values were adjusted programmatically
    this.vRatioValLabel.textContent = clampNum(
      Number.parseFloat(this.vRatioSlider.value),
      0.5,
      4.0
    ).toFixed(2);
    this.impactBValLabel.textContent = clampNum(
      Number.parseFloat(this.impactBSlider.value),
      0.0,
      1.2
    ).toFixed(2);
    this.initialSepValLabel.textContent = `${clampNum(
      Number.parseFloat(this.initialSepSlider.value),
      1.5,
      6.0
    ).toFixed(1)} R`;
  }

  setCameraReadout(mode: CameraMode, altitudeKm: number): void {
    this.setCameraModeDisplay(mode);
    this.altitudeReadout.textContent = formatAltitude(altitudeKm);
  }

  dispose(): void {
    for (const cleanup of this.cleanups) {
      cleanup();
    }
    this.cleanups.length = 0;
    if (this.root.parentNode !== null) {
      this.root.parentNode.removeChild(this.root);
    }
  }

  private listen(
    target: EventTarget,
    event: string,
    handler: EventListenerOrEventListenerObject
  ): void {
    target.addEventListener(event, handler);
    this.cleanups.push(() => {
      target.removeEventListener(event, handler);
    });
  }

  private buildArchetypeSelect(
    doc: Document,
    defaultArch: string,
    testId: string
  ): HTMLSelectElement {
    const sel = makeEl(doc, 'select', {
      className: 'astraea-select',
      testId,
    });
    for (const arch of ARCHETYPES) {
      const opt = makeEl(doc, 'option', {
        text: arch,
        attrs: { value: arch },
      });
      if (arch === defaultArch) {
        opt.selected = true;
      }
      sel.appendChild(opt);
    }
    sel.value = defaultArch;
    return sel;
  }

  private buildRangeField(
    doc: Document,
    label: string,
    testId: string,
    min: number,
    max: number,
    step: number,
    initial: number,
    formatFn: (v: number) => string
  ): {
    wrapper: HTMLElement;
    input: HTMLInputElement;
    valLabel: HTMLElement;
  } {
    const wrapper = makeEl(doc, 'div', { className: 'astraea-field' });
    const row = makeEl(doc, 'div', { className: 'astraea-field-row' });
    row.appendChild(makeEl(doc, 'span', { text: label }));
    const valLabel = makeEl(doc, 'span', {
      className: 'astraea-field-val',
      text: formatFn(initial),
    });
    row.appendChild(valLabel);
    const input = makeEl(doc, 'input', {
      className: 'astraea-range',
      testId,
      attrs: {
        type: 'range',
        min: String(min),
        max: String(max),
        step: String(step),
        value: String(initial),
      },
    });
    input.value = String(initial);
    const syncLabel = (): void => {
      const parsed = clampNum(Number.parseFloat(input.value), min, max);
      valLabel.textContent = formatFn(parsed);
    };
    this.listen(input, 'input', syncLabel);
    this.listen(input, 'change', syncLabel);
    wrapper.appendChild(row);
    wrapper.appendChild(input);
    return { wrapper, input, valLabel };
  }

  private getSelectedParticleCount(): number {
    const parsed = Number.parseInt(this.particleSelect.value, 10);
    return Number.isFinite(parsed) && parsed >= 2 ? parsed : 100000;
  }

  private parseSliceNormal(): [number, number, number] {
    const val = this.sliceNormalSelect.value;
    const match = SLICE_NORMALS.find((s) => s.key === val);
    return match !== undefined ? match.normal : [0, 1, 0];
  }

  private readPositiveNumber(raw: string, fallback: number): number {
    const v = Number.parseFloat(raw);
    return Number.isFinite(v) && v > 0.0 ? v : fallback;
  }

  private setActivePresetButton(presetId: string): void {
    this.activePreset = presetId;
    for (const [id, btn] of this.presetButtons.entries()) {
      btn.className =
        id === presetId ? 'astraea-btn astraea-btn-active' : 'astraea-btn';
    }
  }

  private syncPlayPauseButton(): void {
    this.playPauseBtn.textContent = this.isPlaying ? 'Pause' : 'Play';
    if (typeof this.playPauseBtn.setAttribute === 'function') {
      this.playPauseBtn.setAttribute(
        'aria-pressed',
        this.isPlaying ? 'true' : 'false'
      );
    }
  }

  private setCameraModeDisplay(mode: CameraMode): void {
    this.cameraMode = mode;
    this.cameraModeToggleBtn.textContent =
      mode === 'OrbitalFree'
        ? 'Mode: OrbitalFree (Switch to SurfaceLock)'
        : 'Mode: SurfaceLock (Switch to OrbitalFree)';
    if (typeof this.cameraModeToggleBtn.setAttribute === 'function') {
      this.cameraModeToggleBtn.setAttribute('data-camera-mode', mode);
    }
  }
}
