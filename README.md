# ASTRAEA — Scientific 3D GPU Planetary Collision & Biosphere Impact Simulator

**ASTRAEA** (`astraea-impact-sim`) is a GPU-accelerated 3D planetary collision simulator written in **C++20 + CUDA (`sm_89`)** with a streaming POSIX HTTP / binary WebSocket server and a **Three.js + TypeScript** WebGL2 client that zooms seamlessly across seven orders of magnitude—from orbital mega-impacts down to individual procedural biomes, flora, and fauna reacting with physical wave arrival delays.

---

## Key Scientific & Engineering Features

### 1. C++20 / CUDA SPH + Barnes-Hut Physics Engine (`engine/`)
- **Self-Gravity (`O(N log N)`):** 64-bit Morton-code (`cub::DeviceRadixSort`) + Karras (2012) Linear BVH octree/binary tree + Barnes-Hut monopole traversal (`theta = 0.5`) with particle-constant conservative Plummer softening.
- **Multi-Material Hydrodynamics:** 3D Wendland $C^2$ $\delta$-SPH continuity density (with same-material acoustic diffusion to preserve sharp Core–Mantle–Ocean–Atmosphere boundaries), Monaghan artificial viscosity + Balsara shear switch, Monaghan tensile instability mitigation, XSPH position drift, and adaptive Leapfrog (Kick–Drift–Kick) integration.
- **Equations of State & Strength (`engine/eos.cu`):**
  - Published **Tillotson EOS** for `Iron`, `Granite`, `WaterIce`, and `Regolith` across compressed, two-phase transition, and hot expanded vapor regimes with analytical sound speed $c_s$ and cold compression energy $u_{cold}(\rho)$.
  - Ideal/polytropic **`HydrogenHelium` EOS** for gas giants.
  - Jaumann deviatoric stress update, Von Mises + granular-friction yield return, **Grady–Kipp brittle damage** $D \in [0, 1]$, and Stefan–Boltzmann radiative surface cooling ($\sigma_{SB} T^4$).
- **Hydrostatic Equilibrium Planet Generator (`engine/planet_builder.cpp`):** 1D radial hydrostatic integration ($\frac{dP}{dr} = -\frac{G M(r) \rho(r)}{r^2}$) followed by GPU SPH damping relaxation across **6 archetypes** (`LivingTerra`, `IronCore`, `OceanIce`, `LavaIo`, `GasGiant`, `ProtoTheia`) and **5 preset scenarios** (`EarthTheiaMoonForming`, `MercuryMantleStrip`, `OceanWorldSteamPlume`, `GasGiantCoreImpact`, `LivingGaiaAsteroidCataclysm`).

### 2. Streaming C++20 Server & Headless Runner (`server/`, `engine/headless_main.cpp`)
- **Zero-dependency POSIX HTTP + RFC 6455 WebSocket server (`sim_server`):** serves the built Vite/Three.js bundle, accepts JSON control RPC commands, and broadcasts fixed-layout `PCLS` binary frames (`40 B` header + `20 B/particle` + `40 B/tracer` + optional speed byte array and optional `--zlib` payload compression).
- **Socket Backpressure Protection:** limits TCP send buffering (`SO_SNDBUF` + `TIOCOUTQ`) and skips intermediate streaming binary frames above `256 KiB` queued bytes so software WebGL clients stay responsive while still guaranteeing complete delivery for `SetPreset`, `LaunchCustom`, and `Step`.
- **Headless Mode (`sim_headless`):** deterministic physics execution, JSON/CSV telemetry export, and inline conservation verification (`--verify-conservation`).

### 3. Multi-Scale WebGL2 Renderer & Procedural Biosphere (`web/`)
- **7-Order Scale Camera & LOD (`OrbitalSurfaceCamera.ts`, `CubeSphereLod.ts`):** logarithmic depth buffer + camera-relative origin translation from `> 18,000 km` orbital overview down to sub-kilometer terrain anchored to `6 x 16 x 16 = 1,536` Lagrangian GPU surface tracers (`LivingTerra`), dynamically cracking and culling patches as crust damage approaches $D \ge 0.85$.
- **6 Visualization Modes (`SphParticleRenderer.ts`):** `NaturalBlackbody` (Planck thermal emission + material albedo), `TemperatureK`, `PressureGPa`, `PhaseTransition`, `DamageFracture`, and `VelocityKmS`, plus interactive arbitrary-plane cross-section slicing.
- **Procedural 3D Biosphere (`web/src/biosphere/`):**
  - **`BiomeSystem.ts`:** latitude, elevation, and moisture classification across all 7 biomes (`TropicalRainforest`, `Savanna`, `TemperateForest`, `BorealTaiga`, `AlpineTundra`, `AridSteppe`, `OceanReef`).
  - **`ProceduralFlora.ts`:** 9 L-system / procedural instanced flora species (`ButtressCanopyTree`, `CoastalPalm`, `AcaciaSavannaTree`, `BroadleafOak`, `BorealConifer`, `GiantFern`, `DesertSaguaroCactus`, `CoralColony`, `GiantKelp`) with blast-bend, scorch, and thermal incandescence shaders.
  - **`ProceduralFauna.ts`:** 4 behavioral guilds (`HerbivoreHerd`, `PursuitPredator`, `AvianFlock`, `PelagicMarine`) transitioning through a monotonic stress state machine (`Grazing` -> `Alert` -> `StampedeOrMassFlight` -> `ThermalEscapeOrBurrow` -> `Lethal`).
  - **`ImpactBiosphereCoupler.ts`:** filters interior molten core phases (`phaseId === 2`) and pre-contact approach/settling noise; couples localized seismic P/S-waves ($t_P < t_S$), radiative thermal flash with horizon line-of-sight occultation, atmospheric blast overpressure, ballistic ejecta, tsunami arrival, global dust optical depth ($\tau$), stratospheric $\mathrm{CO_2}$, and Shannon biodiversity ($H'$) extinction tracking.

---

## Repository Layout

```text
CMakeLists.txt               # Root CMake configuration (C++20, CUDA 20, sm_89)
engine/                      # CUDA SPH/Barnes-Hut solver, EOS, planet builder, sim_headless
server/                      # POSIX HTTP + RFC 6455 binary WebSocket server (sim_server)
tests/cpp/                   # GTest physics & protocol verification suites
web/                         # Vite + TypeScript + Three.js frontend & Vitest tests
.agent-specs/                # Architecture specification and stage plan
archive/factory-game/        # Preserved previous workspace prototype
```

---

## Build & Verification

### Prerequisites
- CMake `>= 3.24`, C++20 compiler, CUDA Toolkit (`nvcc` targeting `sm_89` / Ada Lovelace; adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` for other GPUs)
- Development headers/libraries: `GTest`, `zlib`
- Node.js & npm (for `web/`)

### Build & Run C++ Tests (`CTest`)
```bash
cmake -B build
cmake --build build -j12
ctest --test-dir build --output-on-failure
```

### Build & Run Frontend Tests (`Vitest`)
```bash
npm --prefix web ci
npm --prefix web run typecheck
npm --prefix web run build
npm --prefix web run test
```

---

## Running the Simulation

### Interactive Web UI + Live GPU Server
```bash
./build/sim_server \
  --port 8080 \
  --bind 127.0.0.1 \
  --web-root ./web/dist \
  --particles 100000 \
  --preset LivingGaiaAsteroidCataclysm \
  --target-fps 20
```
Then open `http://127.0.0.1:8080` in a WebGL2-capable browser. Use `--zlib` to compress binary payloads on slower links, and `--help` to view all options.

### Headless Scientific Run & Conservation Audit
```bash
./build/sim_headless \
  --preset LivingGaiaAsteroidCataclysm \
  --particles 100000 \
  --steps 40 \
  --verify-conservation \
  --dump-telemetry telemetry.json \
  --output-csv history.csv
```

---

## Verified Benchmark & Test Summary

Measured on an **NVIDIA GeForce RTX 4050 Laptop GPU (`sm_89`)**:
- **100,000-particle live CUDA step (`LivingGaiaAsteroidCataclysm`, `40` steps):**
  - `medianGpuStepMs = 18.22 ms` (`~55 steps/s`)
  - `energyDriftRel = 9.45e-6`, `dPRel = 2.81e-6`, `dLRel = 1.69e-4`
- **C++ / CUDA & Protocol Tests (`ctest`):** `30/30` passed (`100%`)
- **Frontend Tests (`vitest`, `tsc --noEmit`, `vite build`):** `5/5` test files, `31/31` tests passed
- **Headless Chromium 156 WebGL2 E2E (`AC-22`):** live `100,000` particles + `1,536` tracers, all 6 `VisMode` shaders, cross-section slicing, `SurfaceLock` biome/flora/fauna rendering, `Play`/`Pause`, and preset switching verified with `0` console/JS/WebGL errors.
