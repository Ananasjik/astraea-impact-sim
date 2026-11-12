# Project Memory

## 1. Environment & Toolchain
- OS: Linux (Arch), CPU: 12 threads, RAM: 15 GiB.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (`sm_89`, 6141 MiB VRAM).
- Compilers & CUDA: `g++` (GCC 16.2.1), `cmake` 4.4.5, `nvcc` CUDA 13.4 (`V13.4.92`), CUB `300500` (`/opt/cuda/include/cccl` / `/usr/include/cccl`).
- System libraries (`pkg-config`): `gtest`, `gtest_main`, `zlib`, `libdeflate`.
- Web & Browser: `node` v26.13.0, `npm` 12.2.2, `chromium` 156.0.8078.79, `python3` 3.14.10.

## 2. Verified Runbooks
- CUDA C++20 compilation probe verified with: `nvcc -std=c++20 -arch=sm_89 <file>.cu -o <out>`.

## 3. Known Pitfalls & Workarounds
- `gh auth status` in sandbox reported invalid token for account `Ananasjik` (re-check outside sandbox before push; never assume network inside sandbox).
- Model policy: ONLY use `Gemini 4 Hydrogen (XHigh)` via `TypeName: "self"` + `Model: "inherit"` (or `define_subagent` with `Model: "inherit"`); NEVER invoke `argon`, `flash`, `pro`, `opus`, `sonnet`, or `research`.

## 4. Architecture Invariants
- Previous game (*Browser Factory Game*) archived in `archive/factory-game/`.
- `.agent-specs/` is tracked in git and excluded from `.gitignore`.
- Planetary collision simulation uses C++20/CUDA backend (`engine/`, `server/`) + Vite/TypeScript/Three.js frontend (`web/`).
