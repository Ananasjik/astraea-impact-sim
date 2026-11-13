import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { BiomeId } from '../src/biosphere/BiomeSystem';
import type { BiosphereMetrics } from '../src/biosphere/ImpactBiosphereCoupler';
import type { FaunaGuild } from '../src/biosphere/ProceduralFauna';
import type {
  CustomScenarioDto,
  TelemetrySnapshotDto,
} from '../src/net/SimClient';
import type { CameraMode, SurfaceAnchorCoord } from '../src/render/OrbitalSurfaceCamera';
import type { VisMode } from '../src/render/SphParticleRenderer';
import { ControlPanel } from '../src/ui/ControlPanel';
import { SimulationApp } from '../src/ui/SimulationApp';
import { TelemetryCharts } from '../src/ui/TelemetryCharts';

const ALL_BIOMES: readonly BiomeId[] = [
  'TropicalRainforest',
  'Savanna',
  'TemperateForest',
  'BorealTaiga',
  'AlpineTundra',
  'AridSteppe',
  'OceanReef',
];

const ALL_GUILDS: readonly (FaunaGuild | 'Flora')[] = [
  'Flora',
  'HerbivoreHerd',
  'PursuitPredator',
  'AvianFlock',
  'PelagicMarine',
];

const ALL_VIS_MODES: readonly VisMode[] = [
  'NaturalBlackbody',
  'TemperatureK',
  'PressureGPa',
  'MaterialLayer',
  'DamageFracture',
  'Velocity',
];

// ============================================================================
// Lightweight Minimal Fake DOM + Canvas 2D / WebGL2 / WebSocket Harness
// ============================================================================

interface RecordedCanvasDrawCall {
  method: string;
  args: unknown[];
}

interface FakeEventInit {
  bubbles?: boolean;
  cancelable?: boolean;
  detail?: unknown;
  deltaY?: number;
  movementX?: number;
  movementY?: number;
  clientX?: number;
  clientY?: number;
  button?: number;
  key?: string;
  code?: string;
}

class FakeEvent {
  readonly type: string;
  readonly bubbles: boolean;
  readonly cancelable: boolean;
  defaultPrevented = false;
  propagationStopped = false;
  target: FakeElement | null = null;
  currentTarget: FakeElement | null = null;
  detail: unknown;
  deltaY: number;
  movementX: number;
  movementY: number;
  clientX: number;
  clientY: number;
  button: number;
  key: string;
  code: string;

  constructor(type: string, init: FakeEventInit = {}) {
    this.type = type;
    this.bubbles = init.bubbles ?? false;
    this.cancelable = init.cancelable ?? false;
    this.detail = init.detail;
    this.deltaY = init.deltaY ?? 0;
    this.movementX = init.movementX ?? 0;
    this.movementY = init.movementY ?? 0;
    this.clientX = init.clientX ?? 0;
    this.clientY = init.clientY ?? 0;
    this.button = init.button ?? 0;
    this.key = init.key ?? '';
    this.code = init.code ?? '';
  }

  preventDefault(): void {
    if (this.cancelable) {
      this.defaultPrevented = true;
    }
  }

  stopPropagation(): void {
    this.propagationStopped = true;
  }
}

class FakeClassList {
  private readonly owner: FakeElement;
  private readonly tokens = new Set<string>();

  constructor(owner: FakeElement) {
    this.owner = owner;
  }

  get length(): number {
    return this.tokens.size;
  }

  syncFromString(className: string): void {
    this.tokens.clear();
    for (const part of className.trim().split(/\s+/)) {
      if (part.length > 0) {
        this.tokens.add(part);
      }
    }
  }

  toString(): string {
    return Array.from(this.tokens).join(' ');
  }

  add(...names: string[]): void {
    for (const n of names) {
      if (n) {
        this.tokens.add(n);
      }
    }
    this.owner._syncClassNameFromList();
  }

  remove(...names: string[]): void {
    for (const n of names) {
      this.tokens.delete(n);
    }
    this.owner._syncClassNameFromList();
  }

  toggle(token: string, force?: boolean): boolean {
    const shouldAdd = force !== undefined ? force : !this.tokens.has(token);
    if (shouldAdd) {
      this.tokens.add(token);
    } else {
      this.tokens.delete(token);
    }
    this.owner._syncClassNameFromList();
    return shouldAdd;
  }

  contains(token: string): boolean {
    return this.tokens.has(token);
  }

  replace(oldToken: string, newToken: string): boolean {
    if (!this.tokens.has(oldToken)) {
      return false;
    }
    this.tokens.delete(oldToken);
    this.tokens.add(newToken);
    this.owner._syncClassNameFromList();
    return true;
  }
}

function createFakeStyle(): CSSStyleDeclaration {
  const store = new Map<string, string>();
  const target: Record<string, unknown> = {
    setProperty(prop: string, value: string | null) {
      if (value === null || value === '') {
        store.delete(prop);
      } else {
        store.set(prop, String(value));
      }
    },
    getPropertyValue(prop: string) {
      return store.get(prop) ?? '';
    },
    removeProperty(prop: string) {
      const prev = store.get(prop) ?? '';
      store.delete(prop);
      return prev;
    },
    get cssText() {
      return Array.from(store.entries())
        .map(([k, v]) => `${k}: ${v}`)
        .join('; ');
    },
    set cssText(val: string) {
      store.clear();
      for (const decl of val.split(';')) {
        const idx = decl.indexOf(':');
        if (idx > 0) {
          store.set(decl.slice(0, idx).trim(), decl.slice(idx + 1).trim());
        }
      }
    },
  };

  return new Proxy(target, {
    get(obj, prop) {
      if (typeof prop === 'symbol') {
        return undefined;
      }
      if (prop in obj) {
        return obj[prop];
      }
      return store.get(prop) ?? '';
    },
    set(obj, prop, value) {
      if (typeof prop === 'symbol') {
        return false;
      }
      if (prop === 'cssText') {
        obj.cssText = String(value ?? '');
        return true;
      }
      if (value === null || value === undefined || value === '') {
        store.delete(prop);
      } else {
        store.set(prop, String(value));
      }
      return true;
    },
  }) as unknown as CSSStyleDeclaration;
}

class FakeTextNode {
  readonly nodeType = 3;
  readonly nodeName = '#text';
  parentNode: FakeElement | null = null;
  data: string;

  constructor(data: string) {
    this.data = data;
  }

  get textContent(): string {
    return this.data;
  }

  set textContent(val: string | null) {
    this.data = val ?? '';
  }

  remove(): void {
    if (this.parentNode !== null) {
      this.parentNode.removeChild(this as unknown as FakeElement);
    }
  }
}

type FakeChildNode = FakeElement | FakeTextNode;

interface FakeListenerEntry {
  fn: (ev: FakeEvent) => void;
  once: boolean;
}

function toDataAttrName(prop: string): string {
  return 'data-' + prop.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`);
}

function toDatasetPropName(attr: string): string {
  return attr
    .slice(5)
    .replace(/-([a-z])/g, (_m, ch: string) => ch.toUpperCase());
}

const VOID_TAGS = new Set([
  'AREA',
  'BASE',
  'BR',
  'COL',
  'EMBED',
  'HR',
  'IMG',
  'INPUT',
  'LINK',
  'META',
  'PARAM',
  'SOURCE',
  'TRACK',
  'WBR',
]);

function decodeBasicHtmlEntities(s: string): string {
  return s
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&');
}

function parseAttributesInto(attrText: string, el: FakeElement): void {
  const attrRe = /([a-zA-Z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let match: RegExpExecArray | null;
  while ((match = attrRe.exec(attrText)) !== null) {
    const name = match[1];
    const rawVal = match[2] ?? match[3] ?? match[4] ?? '';
    el.setAttribute(name, decodeBasicHtmlEntities(rawVal));
  }
}

class FakeElement {
  readonly nodeType = 1;
  readonly tagName: string;
  readonly nodeName: string;
  ownerDocument: FakeDocument;
  parentNode: FakeElement | null = null;
  readonly _childNodes: FakeChildNode[] = [];
  readonly attributes = new Map<string, string>();
  readonly style: CSSStyleDeclaration = createFakeStyle();
  readonly classList: FakeClassList;
  readonly dataset: Record<string, string>;
  private readonly listeners = new Map<string, FakeListenerEntry[]>();

  id = '';
  private _className = '';
  type = '';
  name = '';
  protected _value: string | undefined = undefined;
  checked = false;
  disabled = false;
  selected = false;
  min = '';
  max = '';
  step = '';
  placeholder = '';
  title = '';
  htmlFor = '';
  tabIndex = 0;
  hidden = false;
  href = '';
  download = '';
  rel = '';
  target = '';
  private _clientWidth = 800;
  private _clientHeight = 600;
  offsetWidth = 800;
  offsetHeight = 600;
  scrollWidth = 800;
  scrollHeight = 600;

  get clientWidth(): number {
    return this._clientWidth;
  }
  set clientWidth(v: number) {
    this._clientWidth = v;
  }

  get clientHeight(): number {
    return this._clientHeight;
  }
  set clientHeight(v: number) {
    this._clientHeight = v;
  }

  constructor(tagName: string, ownerDocument: FakeDocument) {
    this.tagName = tagName.toUpperCase();
    this.nodeName = this.tagName;
    this.ownerDocument = ownerDocument;
    this.classList = new FakeClassList(this);
    const selfEl = this;
    this.dataset = new Proxy(
      {},
      {
        get(_t, prop) {
          if (typeof prop !== 'string') return undefined;
          return selfEl.attributes.get(toDataAttrName(prop));
        },
        set(_t, prop, val) {
          if (typeof prop !== 'string') return false;
          selfEl.setAttribute(toDataAttrName(prop), String(val));
          return true;
        },
        deleteProperty(_t, prop) {
          if (typeof prop !== 'string') return false;
          selfEl.removeAttribute(toDataAttrName(prop));
          return true;
        },
        has(_t, prop) {
          if (typeof prop !== 'string') return false;
          return selfEl.attributes.has(toDataAttrName(prop));
        },
      }
    );
  }

  get className(): string {
    return this._className;
  }

  set className(val: string) {
    this._className = val ?? '';
    this.classList.syncFromString(this._className);
    this.attributes.set('class', this._className);
  }

  _syncClassNameFromList(): void {
    this._className = this.classList.toString();
    this.attributes.set('class', this._className);
  }

  get parentElement(): FakeElement | null {
    return this.parentNode;
  }

  get childNodes(): FakeChildNode[] {
    return this._childNodes;
  }

  get children(): FakeElement[] {
    return this._childNodes.filter((n): n is FakeElement => n instanceof FakeElement);
  }

  get firstChild(): FakeChildNode | null {
    return this._childNodes[0] ?? null;
  }

  get lastChild(): FakeChildNode | null {
    return this._childNodes[this._childNodes.length - 1] ?? null;
  }

  get firstElementChild(): FakeElement | null {
    return this.children[0] ?? null;
  }

  get lastElementChild(): FakeElement | null {
    const ch = this.children;
    return ch[ch.length - 1] ?? null;
  }

  get nextSibling(): FakeChildNode | null {
    if (!this.parentNode) return null;
    const idx = this.parentNode._childNodes.indexOf(this as unknown as FakeChildNode);
    return idx >= 0 ? (this.parentNode._childNodes[idx + 1] ?? null) : null;
  }

  get previousSibling(): FakeChildNode | null {
    if (!this.parentNode) return null;
    const idx = this.parentNode._childNodes.indexOf(this as unknown as FakeChildNode);
    return idx > 0 ? (this.parentNode._childNodes[idx - 1] ?? null) : null;
  }

  get value(): string {
    return this._value ?? '';
  }

  set value(val: string) {
    this._value = String(val ?? '');
  }

  get textContent(): string {
    return this._childNodes.map((n) => n.textContent).join('');
  }

  set textContent(val: string | null) {
    this._clearChildren();
    const s = val ?? '';
    if (s.length > 0) {
      const tn = new FakeTextNode(String(s));
      tn.parentNode = this;
      this._childNodes.push(tn);
    }
  }

  get innerText(): string {
    return this.textContent;
  }

  set innerText(val: string | null) {
    this.textContent = val;
  }

  get innerHTML(): string {
    return this._childNodes
      .map((n) => (n instanceof FakeTextNode ? n.data : n.outerHTML))
      .join('');
  }

  set innerHTML(html: string) {
    this._clearChildren();
    if (!html) {
      return;
    }
    const stack: FakeElement[] = [this];
    const tokenRe = /<!--[\s\S]*?-->|<\/([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)([^<>]*?)\/?>|([^<]+)/g;
    let match: RegExpExecArray | null;
    while ((match = tokenRe.exec(html)) !== null) {
      const parent = stack[stack.length - 1];
      if (match[1]) {
        const closing = match[1].toUpperCase();
        for (let i = stack.length - 1; i >= 1; i--) {
          if (stack[i].tagName === closing) {
            stack.length = i;
            break;
          }
        }
      } else if (match[2]) {
        const tag = match[2];
        const attrText = match[3] ?? '';
        const child = this.ownerDocument.createElement(tag);
        if (attrText.trim().length > 0) {
          parseAttributesInto(attrText, child);
        }
        parent.appendChild(child);
        const selfClosing =
          match[0].endsWith('/>') || VOID_TAGS.has(child.tagName);
        if (!selfClosing) {
          stack.push(child);
        }
      } else if (match[4]) {
        const txt = decodeBasicHtmlEntities(match[4]);
        if (txt.length > 0) {
          const tn = new FakeTextNode(txt);
          tn.parentNode = parent;
          parent._childNodes.push(tn);
        }
      }
    }
  }

  get outerHTML(): string {
    const tag = this.tagName.toLowerCase();
    const attrs = Array.from(this.attributes.entries())
      .map(([k, v]) => (v === '' ? k : `${k}="${v}"`))
      .join(' ');
    const open = attrs.length > 0 ? `<${tag} ${attrs}>` : `<${tag}>`;
    if (VOID_TAGS.has(this.tagName)) {
      return open;
    }
    return `${open}${this.innerHTML}</${tag}>`;
  }

  private _clearChildren(): void {
    for (const c of this._childNodes) {
      c.parentNode = null;
    }
    this._childNodes.length = 0;
  }

  appendChild<T extends FakeChildNode>(child: T): T {
    if ((child as unknown as FakeDocumentFragment)._isFragment) {
      const frag = child as unknown as FakeDocumentFragment;
      for (const n of [...frag._childNodes]) {
        this.appendChild(n);
      }
      return child;
    }
    if (child.parentNode !== null) {
      child.parentNode.removeChild(child as unknown as FakeElement);
    }
    child.parentNode = this;
    this._childNodes.push(child);
    return child;
  }

  append(...nodes: Array<FakeChildNode | string>): void {
    for (const node of nodes) {
      if (typeof node === 'string') {
        this.appendChild(new FakeTextNode(node));
      } else {
        this.appendChild(node);
      }
    }
  }

  prepend(...nodes: Array<FakeChildNode | string>): void {
    const first = this._childNodes[0] ?? null;
    for (const node of nodes) {
      const child = typeof node === 'string' ? new FakeTextNode(node) : node;
      this.insertBefore(child, first);
    }
  }

  insertBefore<T extends FakeChildNode>(newNode: T, refNode: FakeChildNode | null): T {
    if (refNode === null) {
      return this.appendChild(newNode);
    }
    if (newNode.parentNode !== null) {
      newNode.parentNode.removeChild(newNode as unknown as FakeElement);
    }
    const idx = this._childNodes.indexOf(refNode);
    newNode.parentNode = this;
    if (idx === -1) {
      this._childNodes.push(newNode);
    } else {
      this._childNodes.splice(idx, 0, newNode);
    }
    return newNode;
  }

  removeChild<T extends FakeChildNode>(child: T): T {
    const idx = this._childNodes.indexOf(child);
    if (idx >= 0) {
      this._childNodes.splice(idx, 1);
      child.parentNode = null;
    }
    return child;
  }

  replaceChildren(...nodes: Array<FakeChildNode | string>): void {
    this._clearChildren();
    this.append(...nodes);
  }

  remove(): void {
    if (this.parentNode !== null) {
      this.parentNode.removeChild(this);
    }
  }

  contains(other: FakeChildNode | null): boolean {
    let cur: FakeChildNode | null = other;
    while (cur !== null) {
      if (cur === this) return true;
      cur = cur.parentNode;
    }
    return false;
  }

  setAttribute(name: string, value: string): void {
    const key = name.toLowerCase();
    const val = String(value);
    this.attributes.set(key, val);
    if (key === 'id') {
      this.id = val;
    } else if (key === 'class') {
      this.className = val;
    } else if (key === 'type') {
      this.type = val;
    } else if (key === 'name') {
      this.name = val;
    } else if (key === 'value') {
      this.value = val;
    } else if (key === 'checked') {
      this.checked = val !== 'false';
    } else if (key === 'disabled') {
      this.disabled = val !== 'false';
    } else if (key === 'selected') {
      this.selected = val !== 'false';
    } else if (key === 'min') {
      this.min = val;
    } else if (key === 'max') {
      this.max = val;
    } else if (key === 'step') {
      this.step = val;
    } else if (key === 'placeholder') {
      this.placeholder = val;
    } else if (key === 'title') {
      this.title = val;
    } else if (key === 'for') {
      this.htmlFor = val;
    }
    if (key.startsWith('data-')) {
      void toDatasetPropName(key);
    }
  }

  getAttribute(name: string): string | null {
    const key = name.toLowerCase();
    if (key === 'id' && this.id) return this.id;
    if (key === 'class' && this._className) return this._className;
    if (key === 'type' && this.type) return this.type;
    if (key === 'name' && this.name) return this.name;
    if (key === 'for' && this.htmlFor) return this.htmlFor;
    return this.attributes.get(key) ?? null;
  }

  hasAttribute(name: string): boolean {
    return this.getAttribute(name) !== null;
  }

  removeAttribute(name: string): void {
    const key = name.toLowerCase();
    this.attributes.delete(key);
    if (key === 'id') this.id = '';
    if (key === 'class') this.className = '';
    if (key === 'disabled') this.disabled = false;
    if (key === 'checked') this.checked = false;
    if (key === 'selected') this.selected = false;
  }

  getBoundingClientRect(): DOMRect {
    const w = this.clientWidth;
    const h = this.clientHeight;
    return {
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: w,
      bottom: h,
      width: w,
      height: h,
      toJSON: () => ({}),
    } as DOMRect;
  }

  addEventListener(
    type: string,
    listener: ((ev: FakeEvent) => void) | { handleEvent: (ev: FakeEvent) => void } | null,
    options?: boolean | { once?: boolean }
  ): void {
    if (!listener) return;
    const fn =
      typeof listener === 'function'
        ? listener
        : (ev: FakeEvent) => listener.handleEvent(ev);
    const once = typeof options === 'object' && options !== null ? Boolean(options.once) : false;
    const list = this.listeners.get(type) ?? [];
    list.push({ fn, once });
    this.listeners.set(type, list);
  }

  removeEventListener(
    type: string,
    listener: ((ev: FakeEvent) => void) | { handleEvent: (ev: FakeEvent) => void } | null
  ): void {
    if (!listener || typeof listener !== 'function') return;
    const list = this.listeners.get(type);
    if (!list) return;
    this.listeners.set(
      type,
      list.filter((entry) => entry.fn !== listener)
    );
  }

  dispatchEvent(event: FakeEvent): boolean {
    try {
      Object.defineProperty(event, 'target', {
        value: this,
        configurable: true,
        writable: true,
      });
    } catch {
      // ignore if sealed native event
    }
    let current: FakeElement | null = this;
    while (current !== null) {
      try {
        Object.defineProperty(event, 'currentTarget', {
          value: current,
          configurable: true,
          writable: true,
        });
      } catch {
        // ignore
      }
      const list = current.listeners.get(event.type);
      if (list && list.length > 0) {
        const snapshot = [...list];
        const remaining = list.filter((e) => !e.once);
        current.listeners.set(event.type, remaining);
        for (const entry of snapshot) {
          entry.fn.call(current, event);
        }
      }
      if (!event.bubbles || event.propagationStopped) {
        break;
      }
      current = current.parentNode;
    }
    return !event.defaultPrevented;
  }

  click(): void {
    if (this.disabled) {
      return;
    }
    const isCheckbox = this.tagName === 'INPUT' && this.type.toLowerCase() === 'checkbox';
    const isRadio = this.tagName === 'INPUT' && this.type.toLowerCase() === 'radio';
    if (isCheckbox) {
      this.checked = !this.checked;
    } else if (isRadio) {
      if (this.name) {
        const radios = this.ownerDocument.querySelectorAll(
          `input[type="radio"][name="${this.name}"]`
        );
        for (const r of radios) {
          r.checked = false;
        }
      }
      this.checked = true;
    }
    const notCancelled = this.dispatchEvent(
      new FakeEvent('click', { bubbles: true, cancelable: true })
    );
    if ((isCheckbox || isRadio) && notCancelled) {
      this.dispatchEvent(new FakeEvent('input', { bubbles: true }));
      this.dispatchEvent(new FakeEvent('change', { bubbles: true }));
    }
  }

  focus(): void {}
  blur(): void {}

  matches(selector: string): boolean {
    return selector
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .some((branch) => matchCompoundChain(this, branch));
  }

  closest(selector: string): FakeElement | null {
    let cur: FakeElement | null = this;
    while (cur !== null) {
      if (cur.matches(selector)) return cur;
      cur = cur.parentNode;
    }
    return null;
  }

  querySelector(selector: string): FakeElement | null {
    return this.querySelectorAll(selector)[0] ?? null;
  }

  querySelectorAll(selector: string): FakeElement[] {
    const out: FakeElement[] = [];
    const visit = (node: FakeElement): void => {
      for (const ch of node.children) {
        if (ch.matches(selector)) {
          out.push(ch);
        }
        visit(ch);
      }
    };
    visit(this);
    return out;
  }
}

class FakeDocumentFragment extends FakeElement {
  readonly _isFragment = true;
  constructor(ownerDocument: FakeDocument) {
    super('#document-fragment', ownerDocument);
  }
}

class FakeOptionElement extends FakeElement {
  override get value(): string {
    return this._value !== undefined ? this._value : this.textContent;
  }
  override set value(val: string) {
    this._value = String(val ?? '');
  }
}

class FakeSelectElement extends FakeElement {
  get options(): FakeOptionElement[] {
    return this.querySelectorAll('option') as FakeOptionElement[];
  }

  override get value(): string {
    if (this._value !== undefined) {
      return this._value;
    }
    const opts = this.options;
    const sel = opts.find((o) => o.selected);
    return sel ? sel.value : (opts[0]?.value ?? '');
  }

  override set value(val: string) {
    const s = String(val ?? '');
    this._value = s;
    for (const o of this.options) {
      o.selected = o.value === s;
    }
  }

  get selectedIndex(): number {
    const opts = this.options;
    const current = this.value;
    return opts.findIndex((o) => o.value === current);
  }

  set selectedIndex(idx: number) {
    const opts = this.options;
    if (idx >= 0 && idx < opts.length) {
      this.value = opts[idx].value;
    }
  }
}

class FakeInputElement extends FakeElement {}
class FakeButtonElement extends FakeElement {}

class FakeCanvas2DContext {
  readonly canvas: FakeCanvasElement;
  readonly drawCalls: RecordedCanvasDrawCall[] = [];
  fillStyle: string | CanvasGradient | CanvasPattern = '#000000';
  strokeStyle: string | CanvasGradient | CanvasPattern = '#000000';
  lineWidth = 1;
  lineCap: CanvasLineCap = 'butt';
  lineJoin: CanvasLineJoin = 'miter';
  font = '10px sans-serif';
  textAlign: CanvasTextAlign = 'start';
  textBaseline: CanvasTextBaseline = 'alphabetic';
  globalAlpha = 1;
  shadowBlur = 0;
  shadowColor = 'rgba(0,0,0,0)';

  constructor(canvas: FakeCanvasElement) {
    this.canvas = canvas;
  }

  private record(method: string, args: unknown[]): void {
    this.drawCalls.push({ method, args });
  }

  clearRect(x: number, y: number, w: number, h: number): void {
    this.record('clearRect', [x, y, w, h]);
  }
  fillRect(x: number, y: number, w: number, h: number): void {
    this.record('fillRect', [x, y, w, h]);
  }
  strokeRect(x: number, y: number, w: number, h: number): void {
    this.record('strokeRect', [x, y, w, h]);
  }
  beginPath(): void {
    this.record('beginPath', []);
  }
  closePath(): void {
    this.record('closePath', []);
  }
  moveTo(x: number, y: number): void {
    this.record('moveTo', [x, y]);
  }
  lineTo(x: number, y: number): void {
    this.record('lineTo', [x, y]);
  }
  arc(x: number, y: number, r: number, sa: number, ea: number, ccw?: boolean): void {
    this.record('arc', [x, y, r, sa, ea, ccw]);
  }
  rect(x: number, y: number, w: number, h: number): void {
    this.record('rect', [x, y, w, h]);
  }
  quadraticCurveTo(cpx: number, cpy: number, x: number, y: number): void {
    this.record('quadraticCurveTo', [cpx, cpy, x, y]);
  }
  bezierCurveTo(
    cp1x: number,
    cp1y: number,
    cp2x: number,
    cp2y: number,
    x: number,
    y: number
  ): void {
    this.record('bezierCurveTo', [cp1x, cp1y, cp2x, cp2y, x, y]);
  }
  stroke(): void {
    this.record('stroke', []);
  }
  fill(): void {
    this.record('fill', []);
  }
  clip(): void {
    this.record('clip', []);
  }
  save(): void {
    this.record('save', []);
  }
  restore(): void {
    this.record('restore', []);
  }
  translate(x: number, y: number): void {
    this.record('translate', [x, y]);
  }
  rotate(angle: number): void {
    this.record('rotate', [angle]);
  }
  scale(x: number, y: number): void {
    this.record('scale', [x, y]);
  }
  setTransform(...args: unknown[]): void {
    this.record('setTransform', args);
  }
  resetTransform(): void {
    this.record('resetTransform', []);
  }
  setLineDash(segments: number[]): void {
    this.record('setLineDash', [segments]);
  }
  fillText(text: string, x: number, y: number, maxWidth?: number): void {
    this.record('fillText', [text, x, y, maxWidth]);
  }
  strokeText(text: string, x: number, y: number, maxWidth?: number): void {
    this.record('strokeText', [text, x, y, maxWidth]);
  }
  drawImage(...args: unknown[]): void {
    this.record('drawImage', args);
  }
  measureText(text: string): TextMetrics {
    return { width: text.length * 7 } as TextMetrics;
  }
  createLinearGradient(): CanvasGradient {
    return { addColorStop() {} } as unknown as CanvasGradient;
  }
  createRadialGradient(): CanvasGradient {
    return { addColorStop() {} } as unknown as CanvasGradient;
  }
}

function createMockWebGL2Context(): Record<string, unknown> {
  const base: Record<string, unknown> = {
    VERSION: 0x1f02,
    RENDERER: 0x1f01,
    VENDOR: 0x1f00,
    SHADING_LANGUAGE_VERSION: 0x8b8c,
    MAX_VERTEX_ATTRIBS: 0x8869,
    MAX_TEXTURE_IMAGE_UNITS: 0x8872,
    MAX_VERTEX_TEXTURE_IMAGE_UNITS: 0x8b4c,
    MAX_TEXTURE_SIZE: 0x0d33,
    MAX_CUBE_MAP_TEXTURE_SIZE: 0x851c,
    MAX_VERTEX_UNIFORM_VECTORS: 0x8dfb,
    MAX_FRAGMENT_UNIFORM_VECTORS: 0x8dfd,
    MAX_VARYING_VECTORS: 0x8dfc,
    MAX_SAMPLES: 0x8d57,
    ALIASED_LINE_WIDTH_RANGE: 0x846e,
    ALIASED_POINT_SIZE_RANGE: 0x846d,
    VIEWPORT: 0x0ba2,
    SCISSOR_BOX: 0x0c10,
    HIGH_FLOAT: 0x8df2,
    MEDIUM_FLOAT: 0x8df1,
    LOW_FLOAT: 0x8df0,
    HIGH_INT: 0x8df5,
    MEDIUM_INT: 0x8df4,
    LOW_INT: 0x8df3,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    VALIDATE_STATUS: 0x8b83,
    ACTIVE_UNIFORMS: 0x8b86,
    ACTIVE_ATTRIBUTES: 0x8b89,
    FRAMEBUFFER_COMPLETE: 0x8cd5,
    getParameter(p: number) {
      if (p === 0x1f02) return 'WebGL 2.0 (Mock)';
      if (p === 0x8b8c) return 'WebGL GLSL ES 3.00 (Mock)';
      if (p === 0x0d33 || p === 0x851c) return 4096;
      if (p === 0x846e || p === 0x846d) return new Float32Array([1, 1]);
      if (p === 0x0ba2 || p === 0x0c10) return new Int32Array([0, 0, 800, 600]);
      return 16;
    },
    getContextAttributes() {
      return { alpha: true, antialias: true, depth: true, stencil: false };
    },
    getExtension() {
      return null;
    },
    getSupportedExtensions() {
      return [];
    },
    getShaderPrecisionFormat() {
      return { rangeMin: 127, rangeMax: 127, precision: 23 };
    },
    getShaderParameter(_s: unknown, p: number) {
      return p === 0x8b81 ? true : 0;
    },
    getProgramParameter(_p: unknown, p: number) {
      return p === 0x8b82 || p === 0x8b83 ? true : 0;
    },
    getShaderInfoLog() {
      return '';
    },
    getProgramInfoLog() {
      return '';
    },
    getUniformLocation() {
      return {};
    },
    getAttribLocation() {
      return 0;
    },
    getActiveUniform() {
      return { name: 'u', size: 1, type: 0x1406 };
    },
    getActiveAttrib() {
      return { name: 'a', size: 1, type: 0x1406 };
    },
    createBuffer() {
      return {};
    },
    createTexture() {
      return {};
    },
    createFramebuffer() {
      return {};
    },
    createRenderbuffer() {
      return {};
    },
    createVertexArray() {
      return {};
    },
    createShader() {
      return {};
    },
    createProgram() {
      return {};
    },
    checkFramebufferStatus() {
      return 0x8cd5;
    },
  };

  return new Proxy(base, {
    get(target, prop) {
      if (typeof prop === 'string' && prop in target) {
        return target[prop];
      }
      if (typeof prop === 'string' && prop === prop.toUpperCase()) {
        return 1;
      }
      return () => 0;
    },
  });
}

class FakeCanvasElement extends FakeElement {
  width = 300;
  height = 150;
  private ctx2d: FakeCanvas2DContext | null = null;
  private gl2: Record<string, unknown> | null = null;

  override get clientWidth(): number {
    return this.width || 640;
  }
  override set clientWidth(v: number) {
    this.width = v;
  }

  override get clientHeight(): number {
    return this.height || 320;
  }
  override set clientHeight(v: number) {
    this.height = v;
  }

  getContext(contextId: string): FakeCanvas2DContext | Record<string, unknown> | null {
    if (contextId === '2d') {
      if (this.ctx2d === null) {
        this.ctx2d = new FakeCanvas2DContext(this);
      }
      return this.ctx2d;
    }
    if (contextId === 'webgl2' || contextId === 'experimental-webgl2') {
      if (!this.ownerDocument.webgl2Enabled) {
        return null;
      }
      if (this.gl2 === null) {
        this.gl2 = createMockWebGL2Context();
      }
      return this.gl2;
    }
    return null;
  }

  getRecorded2DContext(): FakeCanvas2DContext | null {
    return this.ctx2d;
  }

  toDataURL(): string {
    return 'data:image/png;base64,';
  }
}

function matchSingleSimpleSelector(el: FakeElement, sel: string): boolean {
  let rest = sel.trim();
  if (!rest) return true;
  if (rest === '*') return true;

  // Handle :not(...)
  const notRe = /:not\(([^)]+)\)/g;
  let notMatch: RegExpExecArray | null;
  while ((notMatch = notRe.exec(rest)) !== null) {
    if (matchSingleSimpleSelector(el, notMatch[1])) {
      return false;
    }
  }
  rest = rest.replace(/:not\([^)]+\)/g, '');

  // Pseudo-classes
  if (rest.includes(':checked')) {
    if (!el.checked) return false;
    rest = rest.replace(/:checked/g, '');
  }
  if (rest.includes(':disabled')) {
    if (!el.disabled) return false;
    rest = rest.replace(/:disabled/g, '');
  }

  // Attribute selectors [attr], [attr="val"], [attr*=val], [attr^=val]
  const attrRe = /\[([a-zA-Z0-9_-]+)(?:(\*|\^)?=\s*(?:"([^"]*)"|'([^']*)'|([^\]]*)))?\]/g;
  let attrMatch: RegExpExecArray | null;
  while ((attrMatch = attrRe.exec(rest)) !== null) {
    const attrName = attrMatch[1];
    const op = attrMatch[2] ?? '';
    const expected = attrMatch[3] ?? attrMatch[4] ?? attrMatch[5];
    const actual = el.getAttribute(attrName);
    if (actual === null) return false;
    if (expected !== undefined) {
      if (op === '*') {
        if (!actual.includes(expected)) return false;
      } else if (op === '^') {
        if (!actual.startsWith(expected)) return false;
      } else {
        if (actual !== expected) return false;
      }
    }
  }
  rest = rest.replace(attrRe, '');

  // Tag prefix before # or .
  const tagMatch = rest.match(/^([a-zA-Z*][a-zA-Z0-9-]*)/);
  if (tagMatch) {
    const tag = tagMatch[1];
    if (tag !== '*' && el.tagName !== tag.toUpperCase()) {
      return false;
    }
    rest = rest.slice(tag.length);
  }

  // IDs & classes
  const parts = rest.match(/[#.][a-zA-Z0-9_-]+/g) ?? [];
  for (const part of parts) {
    if (part.startsWith('#')) {
      if (el.id !== part.slice(1)) return false;
    } else if (part.startsWith('.')) {
      if (!el.classList.contains(part.slice(1))) return false;
    }
  }
  return true;
}

function matchCompoundChain(el: FakeElement, chain: string): boolean {
  const tokens: Array<{ combinator: ' ' | '>'; selector: string }> = [];
  const norm = chain.replace(/\s*>\s*/g, ' > ').trim();
  const rawParts = norm.split(/\s+/);
  let nextComb: ' ' | '>' = ' ';
  for (const p of rawParts) {
    if (p === '>') {
      nextComb = '>';
      continue;
    }
    tokens.push({ combinator: nextComb, selector: p });
    nextComb = ' ';
  }
  if (tokens.length === 0) return false;

  const last = tokens[tokens.length - 1];
  if (!matchSingleSimpleSelector(el, last.selector)) {
    return false;
  }

  let cur: FakeElement = el;
  for (let i = tokens.length - 2; i >= 0; i--) {
    const comb = tokens[i + 1].combinator;
    const sel = tokens[i].selector;
    if (comb === '>') {
      const parent = cur.parentNode;
      if (!parent || !matchSingleSimpleSelector(parent, sel)) {
        return false;
      }
      cur = parent;
    } else {
      let anc: FakeElement | null = cur.parentNode;
      let matched = false;
      while (anc !== null) {
        if (matchSingleSimpleSelector(anc, sel)) {
          cur = anc;
          matched = true;
          break;
        }
        anc = anc.parentNode;
      }
      if (!matched) return false;
    }
  }
  return true;
}

class FakeDocument {
  webgl2Enabled = true;
  hidden = false;
  visibilityState = 'visible';
  readonly documentElement: FakeElement;
  readonly head: FakeElement;
  readonly body: FakeElement;
  private readonly listeners = new Map<string, Array<(ev: FakeEvent) => void>>();

  constructor() {
    this.documentElement = new FakeElement('html', this);
    this.head = new FakeElement('head', this);
    this.body = new FakeElement('body', this);
    this.documentElement.appendChild(this.head);
    this.documentElement.appendChild(this.body);
  }

  createElement(tagName: string): FakeElement {
    const upper = tagName.toUpperCase();
    if (upper === 'CANVAS') return new FakeCanvasElement(upper, this);
    if (upper === 'INPUT') return new FakeInputElement(upper, this);
    if (upper === 'SELECT') return new FakeSelectElement(upper, this);
    if (upper === 'OPTION') return new FakeOptionElement(upper, this);
    if (upper === 'BUTTON') return new FakeButtonElement(upper, this);
    return new FakeElement(upper, this);
  }

  createElementNS(_ns: string, tagName: string): FakeElement {
    return this.createElement(tagName);
  }

  createTextNode(text: string): FakeTextNode {
    return new FakeTextNode(text);
  }

  createDocumentFragment(): FakeDocumentFragment {
    return new FakeDocumentFragment(this);
  }

  getElementById(id: string): FakeElement | null {
    return this.documentElement.querySelector(`#${id}`);
  }

  querySelector(selector: string): FakeElement | null {
    return this.documentElement.querySelector(selector);
  }

  querySelectorAll(selector: string): FakeElement[] {
    return this.documentElement.querySelectorAll(selector);
  }

  addEventListener(type: string, cb: (ev: FakeEvent) => void): void {
    const list = this.listeners.get(type) ?? [];
    list.push(cb);
    this.listeners.set(type, list);
  }

  removeEventListener(type: string, cb: (ev: FakeEvent) => void): void {
    const list = this.listeners.get(type);
    if (!list) return;
    this.listeners.set(
      type,
      list.filter((fn) => fn !== cb)
    );
  }

  dispatchEvent(ev: FakeEvent): boolean {
    for (const fn of [...(this.listeners.get(ev.type) ?? [])]) {
      fn(ev);
    }
    return !ev.defaultPrevented;
  }
}

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  binaryType = 'blob';
  readyState = FakeWebSocket.CONNECTING;
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: string | ArrayBuffer }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  readonly sentMessages: Array<string | ArrayBuffer> = [];
  closedCode: number | undefined;
  closedReason: string | undefined;
  private readonly listeners = new Map<string, Array<(ev: unknown) => void>>();

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(
    type: string,
    listener: ((ev: unknown) => void) | { handleEvent: (ev: unknown) => void } | null,
    options?: boolean | { once?: boolean }
  ): void {
    if (!listener) return;
    const rawFn =
      typeof listener === 'function'
        ? listener
        : (ev: unknown) => listener.handleEvent(ev);
    const once = typeof options === 'object' && options !== null ? Boolean(options.once) : false;
    const fn = once
      ? (ev: unknown) => {
          this.removeEventListener(type, fn);
          rawFn(ev);
        }
      : rawFn;
    const list = this.listeners.get(type) ?? [];
    list.push(fn);
    this.listeners.set(type, list);
  }

  removeEventListener(
    type: string,
    listener: ((ev: unknown) => void) | { handleEvent: (ev: unknown) => void } | null
  ): void {
    if (!listener || typeof listener !== 'function') return;
    const list = this.listeners.get(type);
    if (!list) return;
    this.listeners.set(
      type,
      list.filter((fn) => fn !== listener)
    );
  }

  private emitWsEvent(type: string, ev: unknown): void {
    for (const fn of [...(this.listeners.get(type) ?? [])]) {
      fn(ev);
    }
  }

  send(data: string | ArrayBuffer): void {
    this.sentMessages.push(data);
  }

  close(code = 1000, reason = ''): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.closedCode = code;
    this.closedReason = reason;
    const ev = { code, reason };
    this.onclose?.(ev);
    this.emitWsEvent('close', ev);
  }

  triggerOpen(): void {
    this.readyState = FakeWebSocket.OPEN;
    const ev = {};
    this.onopen?.(ev);
    this.emitWsEvent('open', ev);
  }

  triggerMessage(data: string | ArrayBuffer): void {
    const ev = { data };
    this.onmessage?.(ev);
    this.emitWsEvent('message', ev);
  }

  triggerError(): void {
    const ev = {};
    this.onerror?.(ev);
    this.emitWsEvent('error', ev);
  }
}

interface DomHarness {
  document: FakeDocument;
  pendingRafs: Map<number, FrameRequestCallback>;
  flushAnimationFrames: (timestampMs?: number) => number;
  restore: () => void;
}

function installFakeDomHarness(): DomHarness {
  const g = globalThis as Record<string, unknown>;
  const savedKeys = [
    'document',
    'window',
    'self',
    'HTMLElement',
    'HTMLCanvasElement',
    'HTMLInputElement',
    'HTMLSelectElement',
    'HTMLButtonElement',
    'HTMLOptionElement',
    'Event',
    'CustomEvent',
    'MouseEvent',
    'PointerEvent',
    'WheelEvent',
    'KeyboardEvent',
    'requestAnimationFrame',
    'cancelAnimationFrame',
    'WebSocket',
    'ResizeObserver',
    'devicePixelRatio',
    'innerWidth',
    'innerHeight',
    'location',
    'getComputedStyle',
  ] as const;

  const prev = new Map<string, { had: boolean; val: unknown }>();
  for (const k of savedKeys) {
    prev.set(k, { had: Object.prototype.hasOwnProperty.call(g, k), val: g[k] });
  }

  const prevCreateObjectURL =
    typeof URL !== 'undefined' && 'createObjectURL' in URL
      ? (URL as unknown as Record<string, unknown>).createObjectURL
      : undefined;
  const prevRevokeObjectURL =
    typeof URL !== 'undefined' && 'revokeObjectURL' in URL
      ? (URL as unknown as Record<string, unknown>).revokeObjectURL
      : undefined;

  const doc = new FakeDocument();
  const pendingRafs = new Map<number, FrameRequestCallback>();
  let nextRafId = 1;
  let rafTimeMs = 0;

  const windowListeners = new Map<string, Array<(ev: FakeEvent) => void>>();
  const fakeWindow = {
    document: doc,
    performance: globalThis.performance,
    location: {
      protocol: 'http:',
      host: '127.0.0.1:5173',
      hostname: '127.0.0.1',
      port: '5173',
      href: 'http://127.0.0.1:5173/',
    },
    devicePixelRatio: 1,
    innerWidth: 1280,
    innerHeight: 720,
    addEventListener(type: string, fn: (ev: FakeEvent) => void) {
      const list = windowListeners.get(type) ?? [];
      list.push(fn);
      windowListeners.set(type, list);
    },
    removeEventListener(type: string, fn: (ev: FakeEvent) => void) {
      const list = windowListeners.get(type);
      if (!list) return;
      windowListeners.set(
        type,
        list.filter((l) => l !== fn)
      );
    },
    dispatchEvent(ev: FakeEvent) {
      for (const fn of [...(windowListeners.get(ev.type) ?? [])]) {
        fn(ev);
      }
      return !ev.defaultPrevented;
    },
    requestAnimationFrame: (cb: FrameRequestCallback): number => {
      const id = nextRafId++;
      pendingRafs.set(id, cb);
      return id;
    },
    cancelAnimationFrame: (id: number): void => {
      pendingRafs.delete(id);
    },
    getComputedStyle: () => createFakeStyle(),
  };

  FakeWebSocket.instances = [];

  g.document = doc;
  g.window = fakeWindow;
  g.self = fakeWindow;
  g.HTMLElement = FakeElement;
  g.HTMLCanvasElement = FakeCanvasElement;
  g.HTMLInputElement = FakeInputElement;
  g.HTMLSelectElement = FakeSelectElement;
  g.HTMLButtonElement = FakeButtonElement;
  g.HTMLOptionElement = FakeOptionElement;
  g.Event = FakeEvent;
  g.CustomEvent = FakeEvent;
  g.MouseEvent = FakeEvent;
  g.PointerEvent = FakeEvent;
  g.WheelEvent = FakeEvent;
  g.KeyboardEvent = FakeEvent;
  g.requestAnimationFrame = fakeWindow.requestAnimationFrame;
  g.cancelAnimationFrame = fakeWindow.cancelAnimationFrame;
  g.WebSocket = FakeWebSocket;
  g.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  g.devicePixelRatio = 1;
  g.innerWidth = 1280;
  g.innerHeight = 720;
  g.location = fakeWindow.location;
  g.getComputedStyle = fakeWindow.getComputedStyle;

  if (typeof URL !== 'undefined') {
    (URL as unknown as Record<string, unknown>).createObjectURL = () => 'blob:mock-url';
    (URL as unknown as Record<string, unknown>).revokeObjectURL = () => {};
  }

  return {
    document: doc,
    pendingRafs,
    flushAnimationFrames(timestampMs?: number): number {
      rafTimeMs = timestampMs ?? rafTimeMs + 16.6667;
      const batch = Array.from(pendingRafs.entries());
      pendingRafs.clear();
      for (const [, cb] of batch) {
        cb(rafTimeMs);
      }
      return batch.length;
    },
    restore() {
      pendingRafs.clear();
      FakeWebSocket.instances = [];
      for (const [k, entry] of prev.entries()) {
        if (entry.had) {
          g[k] = entry.val;
        } else {
          delete g[k];
        }
      }
      if (typeof URL !== 'undefined') {
        const urlObj = URL as unknown as Record<string, unknown>;
        if (prevCreateObjectURL !== undefined) {
          urlObj.createObjectURL = prevCreateObjectURL;
        } else {
          delete urlObj.createObjectURL;
        }
        if (prevRevokeObjectURL !== undefined) {
          urlObj.revokeObjectURL = prevRevokeObjectURL;
        } else {
          delete urlObj.revokeObjectURL;
        }
      }
    },
  };
}

// ============================================================================
// Test Data Builders & DOM Inspection Helpers
// ============================================================================

function makeTelemetryDto(overrides: Partial<TelemetrySnapshotDto> = {}): TelemetrySnapshotDto {
  return {
    type: 'telemetry',
    frameSeq: 1,
    simTimeSec: 120.0,
    dtSec: 1.5,
    state: 'playing',
    preset: 'LivingGaiaAsteroidCataclysm',
    outcome: 'Approaching',
    kineticEnergyJ: 2.4e31,
    potentialEnergyJ: -2.2e32,
    internalEnergyJ: 5.1e31,
    radiatedEnergyJ: 1.2e28,
    totalEnergyJ: -1.45e32,
    energyDriftRel: 0.00025,
    momentumKgMS: [1.0e24, -2.0e23, 0.0],
    angularMomentumKgM2S: [0.0, 7.1e33, 0.0],
    virialRatio: 0.995,
    maxTempK: 3400.0,
    maxPressureGPa: 290.0,
    boundMassFrac: 0.97,
    diskMassFrac: 0.02,
    ejectaMassFrac: 0.01,
    qRatioQrd: 0.14,
    gpuStepMs: 11.4,
    livingBodyComKm: [0.0, 0.0, 0.0],
    livingBodyRadiusKm: 6371.0,
    livingBodySpinRadS: 7.2921e-5,
    impactorComKm: [14000.0, 1800.0, 0.0],
    tracerGridRes: 16,
    livingBodyPresent: true,
    ...overrides,
  };
}

function makeBiosphereMetrics(overrides: Partial<BiosphereMetrics> = {}): BiosphereMetrics {
  return {
    totalBiomassGtC: 540.0,
    baselineBiomassGtC: 550.0,
    shannonDiversityH: 1.82,
    overallSurvivalPct: 98.2,
    globalOpticalDepthTau: 0.12,
    globalDeltaTempK: -0.4,
    survivalByBiome: {
      TropicalRainforest: 97.0,
      Savanna: 98.5,
      TemperateForest: 99.0,
      BorealTaiga: 99.4,
      AlpineTundra: 99.8,
      AridSteppe: 98.0,
      OceanReef: 96.5,
    },
    survivalByGuild: {
      Flora: 98.8,
      HerbivoreHerd: 97.6,
      PursuitPredator: 97.2,
      AvianFlock: 99.1,
      PelagicMarine: 97.0,
    },
    extinctionTier: 'Nominal',
    ...overrides,
  };
}

function countDomNodes(root: FakeElement): number {
  let count = 1;
  for (const ch of root.children) {
    count += countDomNodes(ch);
  }
  return count;
}

function totalCanvas2dDrawCalls(container: FakeElement): number {
  let total = 0;
  for (const c of container.querySelectorAll('canvas')) {
    const ctx = (c as FakeCanvasElement).getRecorded2DContext();
    if (ctx) {
      total += ctx.drawCalls.length;
    }
  }
  return total;
}

function totalSvgRenderNodes(container: FakeElement): number {
  let total = 0;
  for (const svg of container.querySelectorAll('svg')) {
    total += svg.querySelectorAll('*').length;
  }
  return total;
}

function getControlDescriptor(el: FakeElement, root: FakeElement): string {
  const parts: string[] = [
    el.textContent,
    el.id,
    el.name,
    el.className,
    el.title,
    el.placeholder,
    el.getAttribute('aria-label') ?? '',
    el.getAttribute('data-action') ?? '',
    el.getAttribute('data-preset') ?? '',
    el.getAttribute('data-mode') ?? '',
    el.getAttribute('data-vis-mode') ?? '',
    el.getAttribute('data-camera-mode') ?? '',
    el.getAttribute('data-export') ?? '',
    el.getAttribute('data-format') ?? '',
  ];

  if (el.id) {
    const matchingLabel = root.querySelector(`label[for="${el.id}"]`);
    if (matchingLabel) {
      parts.push(matchingLabel.textContent);
    }
  }

  const parentLabel = el.closest('label');
  if (parentLabel) {
    parts.push(parentLabel.textContent);
  }

  const skipParentFallback =
    el.tagName === 'BUTTON' && el.textContent.trim().length > 0;
  if (!skipParentFallback) {
    let parent = el.parentNode;
    for (let depth = 0; depth < 2 && parent && parent !== root; depth++) {
      if (parent.querySelectorAll('input, select, button').length === 1) {
        parts.push(parent.textContent);
        break;
      }
      parent = parent.parentNode;
    }
  }

  return parts.join(' ').trim();
}

function setInputValueAndEmit(input: FakeElement, value: string): void {
  input.value = value;
  input.dispatchEvent(new FakeEvent('input', { bubbles: true }));
  input.dispatchEvent(new FakeEvent('change', { bubbles: true }));
}

// ============================================================================
// Stage E6 Tests
// ============================================================================

describe('Stage E6 Scientific UI', () => {
  let harness: DomHarness;

  beforeEach(() => {
    harness = installFakeDomHarness();
  });

  afterEach(() => {
    harness.restore();
  });

  describe('TelemetryCharts', () => {
    it('mounts chart canvas elements inside the container cleanly', () => {
      const container = harness.document.createElement('div') as unknown as HTMLElement;
      const charts = new TelemetryCharts(container);
      expect(charts).toBeDefined();

      const fakeContainer = container as unknown as FakeElement;
      expect(fakeContainer.children.length).toBeGreaterThan(0);
      const canvases = fakeContainer.querySelectorAll('canvas');
      const svgs = fakeContainer.querySelectorAll('svg');
      expect(canvases.length + svgs.length).toBeGreaterThanOrEqual(1);
    });

    it('pushes 20+ rolling samples across biomes, guilds, drift, temp, and MassExtinction with bounded DOM nodes', () => {
      const container = harness.document.createElement('div') as unknown as HTMLElement;
      const fakeContainer = container as unknown as FakeElement;
      const charts = new TelemetryCharts(container);

      for (let i = 0; i < 20; i++) {
        const frac = i / 19;
        const survivalByBiome = {} as Record<BiomeId, number>;
        ALL_BIOMES.forEach((b, idx) => {
          survivalByBiome[b] = Math.max(5, 100 - frac * (35 + idx * 7));
        });
        const survivalByGuild = {} as Record<FaunaGuild | 'Flora', number>;
        ALL_GUILDS.forEach((g, idx) => {
          survivalByGuild[g] = Math.max(4, 100 - frac * (30 + idx * 9));
        });

        const telemetry = makeTelemetryDto({
          frameSeq: i + 1,
          simTimeSec: i * 150,
          outcome: i < 5 ? 'Approaching' : i < 14 ? 'Merging' : 'CatastrophicDisruption',
          energyDriftRel: (i % 2 === 0 ? 1 : -1) * (1e-4 + frac * 4e-3),
          maxTempK: 300 + frac * 6200,
          maxPressureGPa: 50 + frac * 420,
          boundMassFrac: 1.0 - frac * 0.35,
          diskMassFrac: frac * 0.2,
          ejectaMassFrac: frac * 0.15,
        });

        const bio = makeBiosphereMetrics({
          totalBiomassGtC: 550 * (1 - frac * 0.72),
          baselineBiomassGtC: 550,
          shannonDiversityH: Math.max(0.1, 1.85 - frac * 1.35),
          overallSurvivalPct: Math.max(8, 100 - frac * 76),
          globalOpticalDepthTau: 0.05 + frac * 7.5,
          globalDeltaTempK: frac < 0.3 ? 18 * frac : -22 * (frac - 0.25),
          survivalByBiome,
          survivalByGuild,
          extinctionTier:
            i < 6 ? 'Nominal' : i < 12 ? 'RegionalCrisis' : 'MassExtinction',
        });

        charts.pushSample(telemetry, bio);
      }

      harness.flushAnimationFrames();
      expect(
        totalCanvas2dDrawCalls(fakeContainer) + totalSvgRenderNodes(fakeContainer)
      ).toBeGreaterThan(0);

      const nodeCountAfter20 = countDomNodes(fakeContainer);
      for (let i = 20; i < 140; i++) {
        charts.pushSample(
          makeTelemetryDto({
            frameSeq: i + 1,
            simTimeSec: i * 150,
            energyDriftRel: 0.002 * Math.sin(i * 0.3),
            maxTempK: 4200 + 800 * Math.cos(i * 0.2),
          }),
          makeBiosphereMetrics({
            overallSurvivalPct: 24.0,
            extinctionTier: 'MassExtinction',
          })
        );
      }
      harness.flushAnimationFrames();

      const nodeCountAfter140 = countDomNodes(fakeContainer);
      expect(nodeCountAfter140).toBeLessThanOrEqual(nodeCountAfter20 + 4);
    });

    it('resets cleanly on clear() and accepts zero-baseline edge cases without throwing', () => {
      const container = harness.document.createElement('div') as unknown as HTMLElement;
      const fakeContainer = container as unknown as FakeElement;
      const charts = new TelemetryCharts(container);

      charts.pushSample(
        makeTelemetryDto({
          totalEnergyJ: 0,
          energyDriftRel: 0,
          maxTempK: 0,
        }),
        makeBiosphereMetrics({
          totalBiomassGtC: 0,
          baselineBiomassGtC: 0,
          overallSurvivalPct: 0,
          shannonDiversityH: 0,
          extinctionTier: 'Sterilization',
        })
      );
      harness.flushAnimationFrames();

      expect(() => charts.clear()).not.toThrow();
      harness.flushAnimationFrames();

      const callsBeforeReuse = totalCanvas2dDrawCalls(fakeContainer);
      expect(() =>
        charts.pushSample(
          makeTelemetryDto({ frameSeq: 2, simTimeSec: 60 }),
          makeBiosphereMetrics()
        )
      ).not.toThrow();
      harness.flushAnimationFrames();
      const canvasCallsAfterReuse = totalCanvas2dDrawCalls(fakeContainer);
      const svgNodesAfterReuse = totalSvgRenderNodes(fakeContainer);
      if (fakeContainer.querySelectorAll('canvas').length > 0) {
        expect(canvasCallsAfterReuse).toBeGreaterThanOrEqual(callsBeforeReuse);
      }
      expect(canvasCallsAfterReuse + svgNodesAfterReuse).toBeGreaterThan(0);
      expect(fakeContainer.querySelectorAll('canvas, svg').length).toBeGreaterThanOrEqual(
        1
      );
    });
  });

  describe('ControlPanel', () => {
    function setupPanel() {
      const container = harness.document.createElement('div') as unknown as HTMLElement;
      const fakeContainer = container as unknown as FakeElement;
      const callbacks = {
        onPreset: vi.fn<(preset: string, particles: number) => void>(),
        onLaunchCustom: vi.fn<(cfg: CustomScenarioDto) => void>(),
        onPlayPause: vi.fn<(playing: boolean) => void>(),
        onStep: vi.fn<() => void>(),
        onTimescale: vi.fn<(scale: number) => void>(),
        onVisMode: vi.fn<(mode: VisMode) => void>(),
        onSliceChange: vi.fn<
          (enabled: boolean, normal: [number, number, number], offsetKm: number) => void
        >(),
        onCameraMode: vi.fn<(mode: CameraMode, anchor?: SurfaceAnchorCoord) => void>(),
        onExport: vi.fn<(format: 'csv' | 'json') => void>(),
      };
      const panel = new ControlPanel(container, callbacks);
      return { panel, fakeContainer, callbacks };
    }

    it('mounts interactive controls inside the container and wires all 9 callbacks', () => {
      const { panel, fakeContainer, callbacks } = setupPanel();
      expect(fakeContainer.children.length).toBeGreaterThan(0);

      const allButtons = fakeContainer.querySelectorAll('button');
      const allSelects = fakeContainer.querySelectorAll('select') as FakeSelectElement[];
      const allInputs = fakeContainer.querySelectorAll('input');
      const radioInputs = allInputs.filter(
        (i) => i.type.toLowerCase() === 'radio'
      );

      // 1. onPreset
      const presetRe =
        /EarthTheiaMoonForming|MercuryMantleStrip|OceanWorldSteamPlume|GasGiantCoreImpact|LivingGaiaAsteroidCataclysm|Earth[\s–-]+Theia|Mercury\s+Mantle|Ocean\s+World|Gas\s+Giant|Living\s+Gaia/i;
      const presetSelect = allSelects.find(
        (s) =>
          s.options.filter((o) =>
            presetRe.test(`${o.value} ${o.textContent}`)
          ).length >= 2
      );
      if (presetSelect) {
        const targetOpt =
          presetSelect.options.find(
            (o) =>
              presetRe.test(`${o.value} ${o.textContent}`) &&
              o.value !== presetSelect.value
          ) ?? presetSelect.options[0];
        setInputValueAndEmit(presetSelect, targetOpt.value);
        if (callbacks.onPreset.mock.calls.length === 0) {
          const loadBtn = allButtons.find((b) =>
            /\b(load|apply|preset|set)\b/i.test(getControlDescriptor(b, fakeContainer))
          );
          loadBtn?.click();
        }
      } else {
        const presetBtn = allButtons.find((b) =>
          presetRe.test(getControlDescriptor(b, fakeContainer))
        );
        expect(presetBtn).toBeDefined();
        presetBtn!.click();
      }
      expect(callbacks.onPreset).toHaveBeenCalled();
      const [presetName, presetParticles] = callbacks.onPreset.mock.calls[0];
      expect(typeof presetName).toBe('string');
      expect(presetName.length).toBeGreaterThan(0);
      expect(Number.isFinite(presetParticles)).toBe(true);
      expect(presetParticles).toBeGreaterThan(0);

      // 2. onPlayPause
      const playPauseButtons = allButtons.filter((b) =>
        /play|pause|▶|⏸/i.test(getControlDescriptor(b, fakeContainer))
      );
      expect(playPauseButtons.length).toBeGreaterThanOrEqual(1);
      if (playPauseButtons.length === 1) {
        const toggleBtn = playPauseButtons[0];
        toggleBtn.click();
        expect(callbacks.onPlayPause).toHaveBeenCalledTimes(1);
        const firstArg = callbacks.onPlayPause.mock.calls[0][0];
        expect(typeof firstArg).toBe('boolean');

        // Drive second toggle via authoritative telemetry.state
        panel.updateState(
          makeTelemetryDto({ state: firstArg ? 'playing' : 'paused' }),
          makeBiosphereMetrics()
        );
        const secondToggleBtn = fakeContainer.contains(toggleBtn)
          ? toggleBtn
          : fakeContainer
              .querySelectorAll('button')
              .find((b) =>
                /play|pause|▶|⏸/i.test(getControlDescriptor(b, fakeContainer))
              );
        expect(secondToggleBtn).toBeDefined();
        secondToggleBtn!.click();
        expect(callbacks.onPlayPause).toHaveBeenCalledTimes(2);
        const secondArg = callbacks.onPlayPause.mock.calls[1][0];
        expect(secondArg).toBe(!firstArg);
      } else {
        const playBtn =
          playPauseButtons.find(
            (b) => !b.disabled && /play|▶/i.test(getControlDescriptor(b, fakeContainer))
          ) ?? playPauseButtons[0];
        playBtn.click();
        panel.updateState(
          makeTelemetryDto({ state: 'playing' }),
          makeBiosphereMetrics()
        );
        const pauseBtn =
          playPauseButtons.find(
            (b) =>
              b !== playBtn &&
              !b.disabled &&
              /pause|⏸/i.test(getControlDescriptor(b, fakeContainer))
          ) ?? playPauseButtons.find((b) => b !== playBtn);
        pauseBtn?.click();
        expect(callbacks.onPlayPause).toHaveBeenCalled();
        for (const [playing] of callbacks.onPlayPause.mock.calls) {
          expect(typeof playing).toBe('boolean');
        }
      }

      // 3. onStep
      panel.updateState(
        makeTelemetryDto({ state: 'paused' }),
        makeBiosphereMetrics()
      );
      const stepBtn = allButtons.find((b) => {
        const desc = getControlDescriptor(b, fakeContainer);
        return /\bstep\b|⏭|single.?step/i.test(desc) && !/steppe/i.test(desc);
      });
      expect(stepBtn).toBeDefined();
      stepBtn!.click();
      expect(callbacks.onStep).toHaveBeenCalledTimes(1);

      // 4. onTimescale
      const timescaleInput = allInputs.find(
        (i) =>
          i.type.toLowerCase() !== 'checkbox' &&
          i.type.toLowerCase() !== 'radio' &&
          /time\s*scale|speed|rate/i.test(getControlDescriptor(i, fakeContainer))
      );
      if (timescaleInput) {
        const minVal = Number.parseFloat(timescaleInput.min);
        const maxVal = Number.parseFloat(timescaleInput.max);
        let targetVal = timescaleInput.value === '10' ? '5' : '10';
        if (Number.isFinite(minVal) && Number.isFinite(maxVal)) {
          const candidate = Number.parseFloat(targetVal);
          const clamped = Math.min(maxVal, Math.max(minVal, candidate));
          targetVal = String(
            String(clamped) === timescaleInput.value ? minVal : clamped
          );
        }
        setInputValueAndEmit(timescaleInput, targetVal);
      } else {
        const timescaleSelect = allSelects.find((s) =>
          /time\s*scale|speed|rate/i.test(getControlDescriptor(s, fakeContainer))
        );
        if (timescaleSelect && timescaleSelect.options.length > 0) {
          const opt =
            timescaleSelect.options.find((o) => o.value !== timescaleSelect.value) ??
            timescaleSelect.options[0];
          setInputValueAndEmit(timescaleSelect, opt.value);
        } else {
          const speedBtn = allButtons.find((b) =>
            /^\s*(0\.1|0\.5|1|2|5|10|25|50|100)\s*[x×]?\s*$/i.test(b.textContent)
          );
          expect(speedBtn).toBeDefined();
          speedBtn!.click();
        }
      }
      expect(callbacks.onTimescale).toHaveBeenCalled();
      const [scaleArg] = callbacks.onTimescale.mock.calls[0];
      expect(Number.isFinite(scaleArg)).toBe(true);
      expect(scaleArg).toBeGreaterThan(0);

      // 5. onVisMode
      const visRe =
        /NaturalBlackbody|TemperatureK|PressureGPa|MaterialLayer|DamageFracture|\bVelocity\b|Natural\s*Blackbody|Temperature|Pressure|Material\s*Layer|Damage/i;
      const visSelect = allSelects.find((s) =>
        s.options.some((o) => visRe.test(`${o.value} ${o.textContent}`))
      );
      if (visSelect) {
        const targetOpt =
          visSelect.options.find(
            (o) =>
              /Temperature|Pressure/i.test(`${o.value} ${o.textContent}`) &&
              o.value !== visSelect.value
          ) ??
          visSelect.options.find((o) => o.value !== visSelect.value) ??
          visSelect.options[0];
        setInputValueAndEmit(visSelect, targetOpt.value);
      } else {
        const visControls = [...allButtons, ...radioInputs];
        const visControl =
          visControls.find((el) =>
            /Temperature|Pressure/i.test(getControlDescriptor(el, fakeContainer))
          ) ??
          visControls.find((el) =>
            visRe.test(getControlDescriptor(el, fakeContainer))
          );
        expect(visControl).toBeDefined();
        visControl!.click();
      }
      expect(callbacks.onVisMode).toHaveBeenCalled();
      expect(ALL_VIS_MODES).toContain(callbacks.onVisMode.mock.calls[0][0]);

      // 6. onSliceChange
      const checkboxes = allInputs.filter(
        (i) => i.type.toLowerCase() === 'checkbox'
      );
      const sliceRe = /slice|cross.?section|clip|cut/i;
      const sliceCheckbox =
        checkboxes.find((i) =>
          sliceRe.test(getControlDescriptor(i, fakeContainer))
        ) ?? (checkboxes.length === 1 ? checkboxes[0] : undefined);
      const sliceToggleBtn = sliceCheckbox
        ? undefined
        : allButtons.find((b) =>
            sliceRe.test(getControlDescriptor(b, fakeContainer))
          );
      const sliceControl = sliceCheckbox ?? sliceToggleBtn;
      expect(sliceControl).toBeDefined();
      sliceControl!.click();
      sliceControl!.click();
      expect(callbacks.onSliceChange.mock.calls.length).toBeGreaterThanOrEqual(2);
      const [enabled1, normal1, offset1] = callbacks.onSliceChange.mock.calls[0];
      const [enabled2] = callbacks.onSliceChange.mock.calls[1];
      expect(typeof enabled1).toBe('boolean');
      expect(enabled2).toBe(!enabled1);
      expect(normal1).toHaveLength(3);
      for (const n of normal1) {
        expect(Number.isFinite(n)).toBe(true);
      }
      expect(Number.isFinite(offset1)).toBe(true);

      // 7. onCameraMode
      const camRe = /OrbitalFree|SurfaceLock|Surface\s*Lock|Orbital/i;
      const camSelect = allSelects.find((s) =>
        s.options.some((o) => camRe.test(`${o.value} ${o.textContent}`))
      );
      if (camSelect) {
        const targetOpt =
          camSelect.options.find(
            (o) =>
              /Surface/i.test(`${o.value} ${o.textContent}`) &&
              o.value !== camSelect.value
          ) ?? camSelect.options[0];
        setInputValueAndEmit(camSelect, targetOpt.value);
      } else {
        const camControls = [...allButtons, ...radioInputs];
        const camControl =
          camControls.find((el) =>
            /Surface/i.test(getControlDescriptor(el, fakeContainer))
          ) ??
          camControls.find((el) =>
            camRe.test(getControlDescriptor(el, fakeContainer))
          );
        expect(camControl).toBeDefined();
        camControl!.click();
      }
      expect(callbacks.onCameraMode).toHaveBeenCalled();
      const [camMode, anchor] = callbacks.onCameraMode.mock.calls[0];
      expect(['OrbitalFree', 'SurfaceLock']).toContain(camMode);
      if (anchor !== undefined) {
        expect(anchor.face).toBeGreaterThanOrEqual(0);
        expect(anchor.face).toBeLessThanOrEqual(5);
        expect(anchor.u).toBeGreaterThanOrEqual(0);
        expect(anchor.u).toBeLessThanOrEqual(1);
        expect(anchor.v).toBeGreaterThanOrEqual(0);
        expect(anchor.v).toBeLessThanOrEqual(1);
      }

      // 8. onExport
      const csvBtn = allButtons.find((b) =>
        /\bcsv\b/i.test(getControlDescriptor(b, fakeContainer))
      );
      const jsonBtn = allButtons.find(
        (b) =>
          b !== csvBtn &&
          /\bjson\b/i.test(getControlDescriptor(b, fakeContainer))
      );
      if (csvBtn && jsonBtn) {
        csvBtn.click();
        jsonBtn.click();
      } else {
        const exportSelect = allSelects.find((s) =>
          s.options.some((o) => /csv|json/i.test(`${o.value} ${o.textContent}`))
        );
        const exportBtn = allButtons.find((b) =>
          /export/i.test(getControlDescriptor(b, fakeContainer))
        );
        expect(exportSelect).toBeDefined();
        expect(exportBtn).toBeDefined();
        setInputValueAndEmit(exportSelect!, 'csv');
        exportBtn!.click();
        setInputValueAndEmit(exportSelect!, 'json');
        exportBtn!.click();
      }
      expect(callbacks.onExport).toHaveBeenCalledWith('csv');
      expect(callbacks.onExport).toHaveBeenCalledWith('json');

      // 9. onLaunchCustom
      let launchBtn = allButtons.find((b) =>
        /launch/i.test(getControlDescriptor(b, fakeContainer))
      );
      if (!launchBtn) {
        const customToggle = allButtons.find((b) =>
          /custom/i.test(getControlDescriptor(b, fakeContainer))
        );
        customToggle?.click();
        const postButtons = fakeContainer
          .querySelectorAll('button')
          .filter((b) => b !== customToggle);
        launchBtn =
          postButtons.find((b) =>
            /launch|custom/i.test(getControlDescriptor(b, fakeContainer))
          ) ??
          postButtons.find((b) =>
            /run|start|apply|submit/i.test(getControlDescriptor(b, fakeContainer))
          );
      }
      expect(launchBtn).toBeDefined();
      launchBtn!.click();
      expect(callbacks.onLaunchCustom).toHaveBeenCalled();
      const [customCfg] = callbacks.onLaunchCustom.mock.calls[0];
      expect(typeof customCfg.targetArchetype).toBe('string');
      expect(customCfg.targetArchetype.length).toBeGreaterThan(0);
      expect(typeof customCfg.impactorArchetype).toBe('string');
      expect(customCfg.impactorArchetype.length).toBeGreaterThan(0);
      expect(customCfg.targetMassKg).toBeGreaterThan(0);
      expect(customCfg.impactorMassKg).toBeGreaterThan(0);
      expect(customCfg.impactVelocityRatioVesc).toBeGreaterThan(0);
      expect(customCfg.impactParamB).toBeGreaterThanOrEqual(0);
      expect(customCfg.initialSepRadii).toBeGreaterThan(0);
      expect(Number.isFinite(customCfg.targetSpinRadS)).toBe(true);
      expect(customCfg.particles).toBeGreaterThan(0);
    });

    it('updates live readout displays cleanly on updateState(telemetry, bio)', () => {
      const { panel, fakeContainer } = setupPanel();

      panel.updateState(
        makeTelemetryDto({
          state: 'paused',
          outcome: 'Approaching',
          energyDriftRel: 0.00012,
          maxTempK: 310,
        }),
        makeBiosphereMetrics({
          overallSurvivalPct: 98.4,
          extinctionTier: 'Nominal',
        })
      );
      const firstText = fakeContainer.textContent;

      panel.updateState(
        makeTelemetryDto({
          frameSeq: 42,
          simTimeSec: 1850,
          state: 'playing',
          outcome: 'CatastrophicDisruption',
          energyDriftRel: 0.0045,
          maxTempK: 7500,
        }),
        makeBiosphereMetrics({
          totalBiomassGtC: 128.7,
          overallSurvivalPct: 23.4,
          extinctionTier: 'MassExtinction',
        })
      );
      const secondText = fakeContainer.textContent;

      expect(secondText).not.toBe(firstText);
      expect(secondText).toMatch(/catastrophic|disruption/i);
      expect(secondText).toMatch(/extinction/i);
      expect(secondText).toMatch(/23/);
    });
  });

  describe('SimulationApp', () => {
    it('constructs with (canvasContainer, uiContainer, wsUrl), mounts UI/canvas, streams WebSocket state, and disposes cleanly', () => {
      expect(SimulationApp.length).toBeGreaterThanOrEqual(2);

      const canvasContainer = harness.document.createElement(
        'div'
      ) as unknown as HTMLElement;
      const uiContainer = harness.document.createElement(
        'div'
      ) as unknown as HTMLElement;
      harness.document.body.appendChild(canvasContainer as unknown as FakeElement);
      harness.document.body.appendChild(uiContainer as unknown as FakeElement);

      const app = new SimulationApp(
        canvasContainer,
        uiContainer,
        'ws://127.0.0.1:18080/ws'
      );
      expect(() => app.start()).not.toThrow();

      const fakeCanvasContainer = canvasContainer as unknown as FakeElement;
      const fakeUiContainer = uiContainer as unknown as FakeElement;
      expect(fakeCanvasContainer.querySelectorAll('canvas').length).toBeGreaterThanOrEqual(
        1
      );
      expect(fakeUiContainer.children.length).toBeGreaterThan(0);

      expect(FakeWebSocket.instances.length).toBeGreaterThanOrEqual(1);
      const ws = FakeWebSocket.instances[0];
      expect(ws.url).toBe('ws://127.0.0.1:18080/ws');

      ws.triggerOpen();
      ws.triggerMessage(
        JSON.stringify({
          type: 'preset_loaded',
          preset: 'LivingGaiaAsteroidCataclysm',
          particleCount: 1000,
          tracerCount: 1536,
          tracerGridRes: 16,
          livingBodyPresent: true,
        })
      );
      ws.triggerMessage(
        JSON.stringify(
          makeTelemetryDto({
            frameSeq: 10,
            simTimeSec: 240,
            outcome: 'GrazeAndMerge',
            maxTempK: 4200,
          })
        )
      );

      expect(() => harness.flushAnimationFrames(16.67)).not.toThrow();
      expect(() => harness.flushAnimationFrames(33.33)).not.toThrow();

      // Trigger a control action inside uiContainer and confirm JSON command emission
      const stepOrExportBtn = fakeUiContainer
        .querySelectorAll('button')
        .find(
          (b) =>
            !b.disabled &&
            /\bstep\b|\bcsv\b|play|pause/i.test(getControlDescriptor(b, fakeUiContainer))
        );
      if (stepOrExportBtn) {
        const beforeSent = ws.sentMessages.length;
        stepOrExportBtn.click();
        expect(ws.sentMessages.length).toBeGreaterThan(beforeSent);
        const lastRaw = ws.sentMessages[ws.sentMessages.length - 1];
        expect(typeof lastRaw).toBe('string');
        const parsed = JSON.parse(lastRaw as string) as { type?: string };
        expect(typeof parsed.type).toBe('string');
      }

      expect(() => app.dispose()).not.toThrow();
      expect(ws.readyState).toBe(FakeWebSocket.CLOSED);
      expect(harness.pendingRafs.size).toBe(0);
      expect(() => app.dispose()).not.toThrow();
    });

    it('supports omitted optional wsUrl using window.location and cleans up on dispose()', () => {
      const canvasContainer = harness.document.createElement(
        'div'
      ) as unknown as HTMLElement;
      const uiContainer = harness.document.createElement(
        'div'
      ) as unknown as HTMLElement;

      const app = new SimulationApp(canvasContainer, uiContainer);
      expect(() => app.start()).not.toThrow();
      expect(FakeWebSocket.instances.length).toBeGreaterThanOrEqual(1);
      expect(FakeWebSocket.instances[0].url).toMatch(/^wss?:\/\//);

      expect(() => app.dispose()).not.toThrow();
      expect(FakeWebSocket.instances[0].readyState).toBe(FakeWebSocket.CLOSED);
      expect(harness.pendingRafs.size).toBe(0);
    });
  });
});
