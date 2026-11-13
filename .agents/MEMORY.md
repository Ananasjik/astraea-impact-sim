# Project Memory

## 1. Environment & Toolchain
- OS: Linux (Arch), CPU: 12 threads, RAM: 15 GiB.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (`sm_89`, 6141 MiB VRAM).
- Compilers & CUDA: `g++` (GCC 16.2.1), `cmake` 4.4.5, `nvcc` CUDA 13.4 (`V13.4.92`), CUB `300500` (`/opt/cuda/include/cccl` / `/usr/include/cccl`).
- System libraries (`pkg-config`): `gtest`, `gtest_main`, `zlib`, `libdeflate`.
- Web & Browser: `node` v26.13.0, `npm` 12.2.2, `chromium` 156.0.8078.79, `python3` 3.14.10.

## 2. Verified Runbooks
- CUDA C++20 compilation probe verified with: `nvcc -std=c++20 -arch=sm_89 <file>.cu -o <out>`.
- C++20/CUDA build & test suite (`30` CTest cases across `engine/` + `server/`):
  `cmake -B build && cmake --build build -j12 && ctest --test-dir build --output-on-failure`
- Web frontend typecheck, Vite build, and Vitest suite (`web/`):
  `npm --prefix web run typecheck && npm --prefix web run build && npm --prefix web run test`
- Live web + C++ HTTP/Binary WebSocket server launch:
  `./build/sim_server --port 8080 --bind 127.0.0.1 --web-root ./web/dist --particles 100000 --preset LivingGaiaAsteroidCataclysm --target-fps 20`
- Headless CUDA benchmark & conservation check (`sim_headless`):
  `./build/sim_headless --preset LivingGaiaAsteroidCataclysm --particles 100000 --steps 40 --verify-conservation`
- Headless Chromium 156 SwiftShader WebGL2 E2E launch flags:
  `--headless=new --no-sandbox --disable-gpu-sandbox --disable-background-networking --disable-component-update --enable-webgl --ignore-gpu-blocklist --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`

## 3. Known Pitfalls & Workarounds
- `gh` CLI inside sandbox cannot access the GNOME/OS keyring or network (`HTTP 401` / invalid token error inside sandbox); running `gh` / `git push` outside sandbox authenticates cleanly as `Ananasjik` and pushed public repo `https://github.com/Ananasjik/astraea-impact-sim`.
- Model policy: ONLY use `Gemini 4 Hydrogen (XHigh)` via `TypeName: "self"` + `Model: "inherit"` (or `define_subagent` with `Model: "inherit"`); NEVER invoke `argon`, `flash`, `pro`, `opus`, `sonnet`, or `research`.
- Slow WebGL/SwiftShader clients must not trigger TCP kernel queue bloat (`30+ MiB` of queued binary frames causing stale UI/preset delays): `server/http_ws_server.cpp` caps `SO_SNDBUF` to `131072`, checks `TIOCOUTQ`, and drops intermediate streaming binary frames above `kStreamBackpressureBytes = 256 KiB` while preserving full `kMaxBroadcastQueueBytes` delivery for control transitions (`SetPreset`, `LaunchCustom`, `Step`).
- `ImpactBiosphereCoupler` filters out intact deep molten core particles (`phaseId === 2`, e.g. 5500 K `LivingTerra` / 1800 K `ProtoTheia` interiors) and suppresses pre-contact hyperbolic approach / SPH settling noise (`preContactNoiseFloor` before contact) so biosphere survival starts at `99.999999%+` (`Nominal`) until real impact arrival.

## 4. Architecture Invariants
- Previous game (*Browser Factory Game*) archived in `archive/factory-game/`.
- `.agent-specs/` is tracked in git and excluded from `.gitignore`.
- Planetary collision simulation (`astraea-impact-sim`) uses C++20/CUDA backend (`engine/`, `server/`) + Vite/TypeScript/Three.js frontend (`web/`).
- Binary WebSocket frames use 40-byte `PCLS` v1 header + 20 B/particle + 40 B/tracer (optional `zlib` compression via `--zlib` + optional per-particle speed array).
- `SimulationApp` exposes runtime/E2E telemetry via `window.__ASTRAEA_DEBUG__` updated both per RAF and on `onFrame`/`onTelemetry`/`onPresetLoaded`, and resets `SurfaceLock` back to `OrbitalFree` (`18,000 km`) when switching to a different collision preset.
