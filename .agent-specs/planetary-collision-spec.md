# Technical Specification: 3D Scientific Planetary Collision Simulation with Procedural Biosphere

---

## 1. Objective

Deliver an end-to-end, real-time 3D GPU-accelerated scientific planetary collision simulation (`astraea-impact-sim`) featuring a **C++20 / CUDA (`sm_89`)** N-body + SPH + strength/damage + Tillotson EOS computational physics engine, an embedded **POSIX C++20 HTTP/1.1 + RFC 6455 Binary WebSocket server** (`sim_server`), a deterministic **CLI headless runner** (`sim_headless`), and a **Vite + TypeScript + Three.js** interactive 3D web client capable of zooming seamlessly across 7 orders of magnitude ($10^7\ \mathrm{m}$ orbital view down to $1\ \mathrm{m}$ planetary surface) onto a procedural living biosphere (`LivingTerra`) that reacts with physically delayed tsunami, seismic, radiative, ejecta, atmospheric blast, and climate extinction effects to real-time SPH impact states.

---

## 2. Boundaries

Explicitly **out of scope**:
- Multi-GPU / MPI distributed clusters (targeted strictly at single-GPU CUDA execution on NVIDIA RTX 4050 `sm_89`).
- Relativistic corrections, magnetic MHD fields, nuclear fusion reactions, or full radiative transfer raytracing inside optically thick planetary interiors.
- General-purpose networking dependencies (no Boost, Asio, OpenSSL, or `nlohmann/json`; uses C++20 STL + POSIX sockets + CUDA + CUB + optional `zlib`).
- Persistent user accounts, databases, or multi-player server rooms.
- Modifying archived files inside `archive/factory-game/`.

---

## 3. Project Context

### Workspace Layout & Conventions
- Workspace root: `/home/slava/antigravity/Web game 1` (git branch `main`).
- Legacy project is archived at `archive/factory-game/` and untouched.
- `.agent-specs/` and `.agents/MEMORY.md` are tracked in Git.
- Verified toolchain (from `.agents/MEMORY.md`):
  - **OS/Hardware:** Linux x86_64, 12 CPU threads, 15 GiB RAM, NVIDIA GeForce RTX 4050 Laptop GPU (`sm_89`, 6141 MiB VRAM).
  - **C++/CUDA:** `g++` (GCC 16.2.1), `cmake` 4.4.5, `nvcc` CUDA 13.4 (`V13.4.92`, flags `-std=c++20 -arch=sm_89`), CUB `300500` (`#include <cub/cub.cuh>`).
  - **System libs (`pkg-config`):** `gtest`, `gtest_main`, `zlib`, `libdeflate`.
  - **Web/Test:** `node` v26.13.0, `npm` 12.2.2, `chromium` 156.0.8078.79.

### Target File Placement & Build/Test Commands
```text
/home/slava/antigravity/Web game 1/
├── CMakeLists.txt
├── engine/
│   ├── units.hpp
│   ├── types.hpp
│   ├── eos.hpp
│   ├── eos.cu
│   ├── planet_builder.hpp
│   ├── planet_builder.cpp
│   ├── presets.hpp
│   ├── presets.cpp
│   ├── sph_gravity.cuh
│   ├── sph_gravity.cu
│   ├── simulation.hpp
│   ├── simulation.cpp
│   └── headless_main.cpp
├── server/
│   ├── ws_crypto.hpp
│   ├── ws_crypto.cpp
│   ├── json_codec.hpp
│   ├── json_codec.cpp
│   ├── binary_protocol.hpp
│   ├── binary_protocol.cpp
│   ├── http_ws_server.hpp
│   ├── http_ws_server.cpp
│   └── sim_server_main.cpp
├── tests/
│   └── cpp/
│       ├── test_eos.cpp
│       ├── test_planet_builder.cpp
│       ├── test_gravity_sph.cpp
│       ├── test_conservation.cpp
│       └── test_server_protocol.cpp
└── web/
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── index.html
    ├── src/
    │   ├── main.ts
    │   ├── net/
    │   │   ├── BinaryProtocol.ts
    │   │   └── SimClient.ts
    │   ├── render/
    │   │   ├── OrbitalSurfaceCamera.ts
    │   │   ├── SphParticleRenderer.ts
    │   │   ├── AtmosphereShader.ts
    │   │   └── CubeSphereLod.ts
    │   ├── biosphere/
    │   │   ├── BiomeSystem.ts
    │   │   ├── ProceduralFlora.ts
    │   │   ├── ProceduralFauna.ts
    │   │   └── ImpactBiosphereCoupler.ts
    │   └── ui/
    │       ├── TelemetryCharts.ts
    │       ├── ControlPanel.ts
    │       └── SimulationApp.ts
    └── tests/
        ├── binary_protocol.test.ts
        ├── camera_lod.test.ts
        ├── biosphere.test.ts
        └── impact_coupler.test.ts
```

- **CMake Targets (`CMakeLists.txt`):**
  - `sim_engine` (static CUDA/C++20 library compiling `engine/*.cpp`, `engine/*.cu` with `-std=c++20 -arch=sm_89 --expt-relaxed-constexpr`)
  - `sim_headless` (executable linking `sim_engine` + `engine/headless_main.cpp`)
  - `sim_server` (executable linking `sim_engine` + `server/*.cpp` + `ZLIB::ZLIB` + `Threads::Threads`)
  - `astraea_tests` (GoogleTest executable compiling `tests/cpp/*.cpp`, registered via `gtest_discover_tests`)
- **Web Scripts (`web/package.json`):** `dev` (`vite`), `build` (`vite build`), `typecheck` (`tsc --noEmit`), `test` (`vitest run`).
- **Build & C++ Tests:**
  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j12
  ctest --test-dir build --output-on-failure
  ```
- **Web Typecheck, Unit Tests & Production Build:**
  ```bash
  cd web
  npm run typecheck
  npx vitest run
  npm run build
  ```

---

## 4. Contract

### 4.1 Internal Unit System (`engine/units.hpp`)

To prevent IEEE-754 `float32` overflow on planetary masses ($m_i^2 \sim 10^{40}\ \mathrm{kg^2} > \mathrm{FLT\_MAX}$) while maintaining full FP32 arithmetic throughput on RTX 4050 (`sm_89`), the CUDA solver executes in **Planetary Simulation Units (PSU)** and converts to **exact SI** at public APIs and in `float64` telemetry accumulators.

```cpp
namespace astraea::units {

inline constexpr double G_SI          = 6.67430e-11;         // m^3 kg^-1 s^-2
inline constexpr double SIGMA_SB_SI   = 5.670374419e-8;      // W m^-2 K^-4
inline constexpr double R_GAS_SI      = 8.314462618;         // J mol^-1 K^-1
inline constexpr double SOLAR_FLUX_1AU = 1361.0;             // W m^-2
inline constexpr double M_EARTH_SI    = 5.97219e24;          // kg
inline constexpr double R_EARTH_SI    = 6.371e6;             // m

// Base PSU Scaling Constants
inline constexpr double L_UNIT_M      = 1.0e6;               // 1 PSU length = 1,000 km
inline constexpr double M_UNIT_KG     = 1.0e24;              // 1 PSU mass   = 10^24 kg
inline constexpr double T_UNIT_S      = 1.0e3;               // 1 PSU time   = 1,000 s

// Derived PSU -> SI Conversion Multipliers
inline constexpr double V_UNIT_MS     = L_UNIT_M / T_UNIT_S;                       // 1.0e3 m/s (1 km/s)
inline constexpr double A_UNIT_MS2    = L_UNIT_M / (T_UNIT_S * T_UNIT_S);          // 1.0 m/s^2
inline constexpr double RHO_UNIT_KGM3 = M_UNIT_KG / (L_UNIT_M * L_UNIT_M * L_UNIT_M); // 1.0e6 kg/m^3
inline constexpr double P_UNIT_PA     = M_UNIT_KG / (L_UNIT_M * T_UNIT_S * T_UNIT_S); // 1.0e12 Pa (1 TPa)
inline constexpr double U_UNIT_JKG    = (L_UNIT_M * L_UNIT_M) / (T_UNIT_S * T_UNIT_S); // 1.0e6 J/kg (1 MJ/kg)
inline constexpr double E_UNIT_J      = M_UNIT_KG * U_UNIT_JKG;                    // 1.0e30 J
inline constexpr double L_ANG_UNIT    = M_UNIT_KG * L_UNIT_M * V_UNIT_MS;          // 1.0e33 kg m^2/s
inline constexpr double G_PSU         = G_SI * M_UNIT_KG * T_UNIT_S * T_UNIT_S / (L_UNIT_M * L_UNIT_M * L_UNIT_M); // 66.7430

inline constexpr float  G_PSU_F       = static_cast<float>(G_PSU);                 // 66.7430f

} // namespace astraea::units
```

### 4.2 Core Data Types, Materials & SoA Layout (`engine/types.hpp`)

```cpp
namespace astraea {

enum class MaterialId : uint8_t {
    Iron           = 0,
    Granite        = 1,
    WaterIce       = 2,
    HydrogenHelium = 3,
    Regolith       = 4
};

enum class PhaseId : uint8_t {
    Solid       = 0,
    Molten      = 1,
    Vapor       = 2,
    Plasma      = 3,
    LiquidWater = 4,
    Gas         = 5
};

enum class PlanetArchetype : uint8_t {
    LivingTerra = 0,
    IronCore    = 1,
    OceanIce    = 2,
    LavaIo      = 3,
    GasGiant    = 4,
    ProtoTheia  = 5
};

enum class ScenarioPresetId : uint8_t {
    EarthTheiaMoonForming       = 0,
    MercuryMantleStrip          = 1,
    OceanWorldSteamPlume        = 2,
    GasGiantCoreImpact          = 3,
    LivingGaiaAsteroidCataclysm = 4
};

enum class CollisionOutcome : uint8_t {
    Approaching            = 0,
    Merging                = 1,
    GrazeAndMerge          = 2,
    HitAndRun              = 3,
    CatastrophicDisruption = 4
};

struct SlicePlaneConfig {
    bool  enabled  = false;
    float normalX  = 0.0f;
    float normalY  = 1.0f;
    float normalZ  = 0.0f;
    float offsetKm = 0.0f;
};

struct BodyInitSpec {
    PlanetArchetype archetype      = PlanetArchetype::LivingTerra;
    double          massKg         = units::M_EARTH_SI;
    double          surfaceTempK   = 288.0;
    double          coreTempK      = 5500.0;
    double          spinOmegaRadS  = 7.292115e-5;
    uint32_t        relaxSteps     = 80;
};

struct CollisionScenarioConfig {
    std::string presetName               = "LivingGaiaAsteroidCataclysm";
    BodyInitSpec target                  = {};
    BodyInitSpec impactor                = { PlanetArchetype::ProtoTheia, 0.025 * units::M_EARTH_SI, 240.0, 1800.0, 0.0, 80 };
    double       impactVelocityRatioVesc = 1.65;   // v_inf_contact / v_esc_mutual
    double       impactParamB            = 0.30;   // sin(theta_impact) in [0.0, 1.2]
    double       initialSepRadii         = 3.2;    // initial COM separation in (R_t + R_i)
    uint32_t     totalParticles          = 100000;
    uint32_t     tracerGridRes           = 16;     // 6 * 16 * 16 = 1536 tracers when LivingTerra is present
    uint64_t     randomSeed              = 42;
};

struct TelemetrySnapshot {
    uint64_t         frameSeq                = 0;
    double           simTimeSec              = 0.0;
    float            dtSec                   = 0.0f;
    CollisionOutcome outcome                 = CollisionOutcome::Approaching;
    double           kineticEnergyJ          = 0.0;
    double           potentialEnergyJ        = 0.0;
    double           internalEnergyJ         = 0.0;
    double           radiatedEnergyJ         = 0.0;
    double           totalEnergyJ            = 0.0;   // E_kin + E_pot + E_int
    double           initialTotalEnergyJ     = 0.0;
    double           energyDriftRel          = 0.0;   // (E_tot + E_rad - E_0) / |E_0|
    double           momentumKgMS[3]         = {0.0, 0.0, 0.0};
    double           angularMomentumKgM2S[3] = {0.0, 0.0, 0.0};
    double           virialRatio             = 0.0;   // -2 * E_kin / E_pot
    double           maxTempK                = 0.0;
    double           maxPressureGPa          = 0.0;
    double           boundMassFrac           = 1.0;
    double           diskMassFrac            = 0.0;
    double           ejectaMassFrac          = 0.0;
    double           qRatioQrd               = 0.0;
    double           gpuStepMs               = 0.0;
    double           livingBodyComKm[3]      = {0.0, 0.0, 0.0};
    double           livingBodyRadiusKm      = 0.0;
    double           livingBodySpinRadS      = 0.0;
    double           impactorComKm[3]        = {0.0, 0.0, 0.0};
    uint32_t         particleCount           = 0;
    uint32_t         tracerCount             = 0;
    uint32_t         tracerGridRes           = 0;
    bool             livingBodyPresent       = false;
};

// Host-mirror / GPU device SoA descriptor (allocated up to capacity maxParticles; 100% coalesced arrays)
struct ParticleBuffersDevice {
    uint32_t capacity = 0;
    uint32_t count    = 0;

    // Ping-pong spatial & state arrays (active pointers + alt pointers for Morton permutation)
    float*    posX = nullptr;      float* posY = nullptr;      float* posZ = nullptr;
    float*    velX = nullptr;      float* velY = nullptr;      float* velZ = nullptr;
    float*    velHalfX = nullptr;  float* velHalfY = nullptr;  float* velHalfZ = nullptr;
    float*    accX = nullptr;      float* accY = nullptr;      float* accZ = nullptr;
    float*    mass = nullptr;
    float*    h = nullptr;         // smoothing length h_i (support H_i = 2 * h_i)
    float*    h0 = nullptr;        // initial smoothing length (fixed softening eps_i = 0.8 * h0_i)
    float*    rho = nullptr;
    float*    drhoDt = nullptr;
    float*    pressure = nullptr;
    float*    soundSpeed = nullptr;
    float*    u = nullptr;
    float*    duDt = nullptr;
    float*    tempK = nullptr;
    float*    radPowerW = nullptr;

    // Symmetric traceless deviatoric stress tensor & damage
    float*    sXX = nullptr; float* sYY = nullptr; float* sZZ = nullptr;
    float*    sXY = nullptr; float* sXZ = nullptr; float* sYZ = nullptr;
    float*    dSXX = nullptr; float* dSYY = nullptr; float* dSZZ = nullptr;
    float*    dSXY = nullptr; float* dSXZ = nullptr; float* dSYZ = nullptr;
    float*    damage = nullptr;    // D in [0, 1]
    float*    damageCbrt = nullptr;// D^(1/3) in [0, 1]
    float*    dDamageCbrtDt = nullptr;
    float*    epsFail = nullptr;   // Weibull threshold
    float*    divV = nullptr;
    float*    curlVX = nullptr; float* curlVY = nullptr; float* curlVZ = nullptr;
    float*    shepardC = nullptr;
    float*    surfFactor = nullptr;
    float*    phiGrav = nullptr;

    // Metadata
    uint8_t*  materialId = nullptr;
    uint8_t*  phaseId = nullptr;
    uint8_t*  bodyId = nullptr;
    uint32_t* originalId = nullptr;
    int32_t*  tracerIndex = nullptr; // -1 or [0, tracerCount - 1]
    float*    tracerRestRadiusPsu = nullptr;
    float*    tracerRestNormalX = nullptr;
    float*    tracerRestNormalY = nullptr;
    float*    tracerRestNormalZ = nullptr;
    uint8_t*  tracerIsOcean = nullptr;
};

} // namespace astraea
```

### 4.3 Equations of State (`engine/eos.hpp`, `engine/eos.cu`)

```cpp
namespace astraea::eos {

struct TillotsonParams {
    double rho0KgM3;      // reference density (kg/m^3)
    double a;             // dimensionless high-energy polytropic factor
    double b;             // dimensionless thermal coupling factor
    double A_Pa;          // bulk modulus linear term (Pa)
    double B_Pa;          // quadratic compression term (Pa)
    double E0_JKg;        // characteristic specific energy (J/kg)
    double Eiv_JKg;       // incipient vaporization energy (J/kg)
    double Ecv_JKg;       // complete vaporization energy (J/kg)
    double alpha;         // exponential cutoff parameter in expanded regime
    double beta;          // exponential cutoff parameter in expanded regime
    double cv_JKgK;       // specific heat capacity (J / (kg K))
    double tMelt0K;       // zero-pressure melting temperature (K)
    double tRefK;         // STP reference temperature for rho0 (K)
    double muShear_Pa;    // intact shear modulus (Pa)
    double yieldY0_Pa;    // intact Von Mises yield strength (Pa)
    double epsFailMin;    // baseline Weibull tensile fracture strain
    double kH2_Poly;      // used only for HydrogenHelium (Pa m^6 / kg^2)
    double gammaIdeal;    // used only for HydrogenHelium
};

struct EosEvalResult {
    float pressurePsu;
    float soundSpeedPsu;
    float tempK;
    uint8_t phaseId;
};

// Host + Device callable exact parameter table accessor
__host__ __device__ const TillotsonParams& getParams(MaterialId mat);

// Evaluates EOS in PSU (rhoPsu, uPsu) -> pressure, soundSpeed, tempK, phaseId
// Stored uPsu is total specific internal energy from 0 K so T = max(3.0, (u - u_cold)/cv) and energy
// conservation remains exact; Tillotson P_c, P_e, two-phase blend, and c_s evaluate at thermal excess
// u_EOS = u_cold(rho) + max(0.0, u - u_cold(rho) - cv * tRefK).
__host__ __device__ EosEvalResult evaluateEosPsu(MaterialId mat, float rhoPsu, float uPsu, float damage);

// Cold compression energy u_cold(rho) in PSU (used for temperature extraction & hydrostatic init)
__host__ __device__ double computeColdEnergyPsu(MaterialId mat, double rhoPsu);

// Converts (rho_SI, T_K) -> (u_SI, P_SI) for 1D hydrostatic profile generation
// Outputs total outUSi = u_cold(rho) + cv * tempK from 0 K while evaluating Tillotson P_SI using
// u_EOS = u_cold(rho) + max(0.0, outUSi - u_cold(rho) - cv * tRefK).
__host__ void evaluateFromRhoTempSI(MaterialId mat, double rhoKgM3, double tempK, double& outUSi, double& outPSi);

// Bisects/Newtons P_SI(rho, T) == targetPSi for 1D planet builder
__host__ double invertDensityFromPressureTempSI(MaterialId mat, double targetPSi, double tempK, double rhoGuessKgM3);

} // namespace astraea::eos
```

#### Exact Published EOS Material Parameter Table (SI Units)

| Constant | `Iron` (Tillotson 1962 / Melosh 1989) | `Granite` (Benz & Asphaug 1999) | `WaterIce` (Allen 1967 / Tillotson Water) | `HydrogenHelium` (Polytropic $n=1$ + Ideal) | `Regolith` (Porous Silicate / Jutzi 2008) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\rho_0\ (\mathrm{kg/m^3})$ | `7800.0` | `2680.0` | `1000.0` | `0.178` (1-bar ref) | `2000.0` |
| $a$ | `0.5` | `0.5` | `0.7` | `0.4` ($\gamma-1$) | `0.5` |
| $b$ | `1.5` | `1.3` | `0.15` | `0.0` | `1.3` |
| $A\ (\mathrm{Pa})$ | `1.28e11` | `1.80e10` | `2.18e9` | `0.0` | `1.20e10` |
| $B\ (\mathrm{Pa})$ | `1.05e11` | `1.80e10` | `1.325e10` | `0.0` | `1.20e10` |
| $E_0\ (\mathrm{J/kg})$ | `9.50e6` | `1.60e7` | `7.00e6` | `0.0` | `1.60e7` |
| $E_{iv}\ (\mathrm{J/kg})$ | `2.40e6` | `3.50e6` | `4.19e5` | `0.0` | `3.50e6` |
| $E_{cv}\ (\mathrm{J/kg})$ | `8.67e6` | `1.80e7` | `2.68e6` | `0.0` | `1.80e7` |
| $\alpha$ | `5.0` | `5.0` | `10.0` | `0.0` | `5.0` |
| $\beta$ | `5.0` | `5.0` | `5.0` | `0.0` | `5.0` |
| $c_v\ (\mathrm{J/(kg\cdot K)})$ | `449.0` | `790.0` | `4184.0` | `8260.0` | `800.0` |
| $T_{melt,0}\ (\mathrm{K})$ | `1811.0` | `1600.0` | `273.15` | `0.0` (gas) | `1500.0` |
| $T_{ref}\ (\mathrm{K})$ | `293.0` | `293.0` | `273.15` | `0.0` | `293.0` |
| $\mu_{shear}\ (\mathrm{Pa})$ | `7.70e10` | `2.20e10` | `3.50e9` (only $T<273.15$) | `0.0` | `8.00e9` |
| $Y_0\ (\mathrm{Pa})$ | `6.80e8` | `3.50e9` | `1.00e7` (only $T<273.15$) | `0.0` | `5.00e7` |
| $\epsilon_{fail,min}$ | `1.0e-3` | `1.0e-4` | `5.0e-4` | `1.0` | `5.0e-5` |
| $K_{H_2}\ (\mathrm{Pa\cdot m^6\,kg^{-2}})$ | `0.0` | `0.0` | `0.0` | `2.08e5` | `0.0` |
| $\gamma_{ideal}$ | `1.0` | `1.0` | `1.0` | `1.4` | `1.0` |

### 4.4 1D Planet Builder, Presets & Simulation Engine (`engine/planet_builder.hpp`, `engine/presets.hpp`, `engine/simulation.hpp`)

```cpp
namespace astraea {

struct RadialProfileSample {
    double rM;
    double rhoKgM3;
    double pressurePa;
    double enclosedMassKg;
    double tempK;
    MaterialId material;
};

struct BuiltPlanetHost {
    double                  equatorialRadiusM = 0.0;
    double                  actualMassKg      = 0.0;
    double                  centralPressurePa = 0.0;
    std::vector<RadialProfileSample> radialProfile;
    std::vector<float>      posX, posY, posZ; // PSU
    std::vector<float>      velX, velY, velZ; // PSU
    std::vector<float>      mass, h0, rho, u, tempK, epsFail;
    std::vector<uint8_t>    materialId, bodyId;
    std::vector<int32_t>    tracerIndex;
    std::vector<float>      tracerRestRadiusPsu;
    std::vector<float>      tracerRestNormalX, tracerRestNormalY, tracerRestNormalZ;
    std::vector<uint8_t>    tracerIsOcean;
    uint32_t                tracerCount = 0;
};

class PlanetBuilder {
public:
    static std::vector<RadialProfileSample> solveHydrostatic1D(
        const BodyInitSpec& spec,
        double& outRadiusM,
        double& outCentralPressurePa);

    static BuiltPlanetHost buildPlanet(
        const BodyInitSpec& spec,
        uint32_t particleCount,
        uint8_t bodyId,
        uint32_t tracerGridRes,
        uint64_t seed);
};

CollisionScenarioConfig makePresetConfig(ScenarioPresetId preset, uint32_t totalParticles = 100000);
ScenarioPresetId parsePresetName(const std::string& name); // throws std::invalid_argument on unknown
std::string presetToString(ScenarioPresetId id);

// Shared Cube-Sphere Face Orientation (identical in C++ planet_builder and TS CubeSphereLod):
// face 0 (+X): ( 1,  sy, -sx), face 1 (-X): (-1,  sy,  sx)
// face 2 (+Y): ( sx,  1, -sy), face 3 (-Y): ( sx, -1,  sy)
// face 4 (+Z): ( sx, sy,   1), face 5 (-Z): (-sx, sy,  -1)
// where sx = 2*u - 1, sy = 2*v - 1, u,v in [0, 1], normalized to unit sphere.
// Coordinate Frame Alignment: CrustTracerRecord streams (xKm, yKm, zKm) directly in the authoritative
// system COM frame (already encoding planet rotation and tidal/impact deformation). CubeSphereLod and
// OrbitalSurfaceCamera resolve vertex and camera positions in this same system COM frame (interpolating
// tracer (xKm, yKm, zKm) when tracerCount > 0; falling back to spinAngleRad only when tracerCount == 0).
void cubeFaceUvToUnitNormal(uint32_t face, float u, float v, float& nx, float& ny, float& nz);
void unitNormalToCubeFaceUv(float nx, float ny, float nz, uint32_t& outFace, float& outU, float& outV);

class Simulation {
public:
    explicit Simulation(uint32_t maxParticles = 250000);
    ~Simulation();

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void loadScenario(const CollisionScenarioConfig& config);
    void relaxBody(uint8_t bodyId, uint32_t steps, double& outFinalVrmsOverVesc);

    TelemetrySnapshot step(double maxDtSec = 5.0);
    TelemetrySnapshot stepForDuration(double durationSec, double maxDtSec = 5.0);

    void setTimeScale(double scale);
    void setSlicePlane(const SlicePlaneConfig& slice);

    TelemetrySnapshot computeTelemetry();
    void extractRenderFrame(
        std::vector<uint8_t>& outParticleBytes, // 20 bytes/particle
        std::vector<uint8_t>& outTracerBytes,   // 40 bytes/tracer
        std::vector<uint8_t>* outSpeedBytes = nullptr); // optional N bytes

    std::string exportTelemetryCsv() const;
    std::string exportTelemetryJson() const;
    const std::vector<TelemetrySnapshot>& telemetryHistory() const;
    const CollisionScenarioConfig& activeConfig() const;
};

} // namespace astraea
```

#### Planet Archetype Layer Composition & Geotherm Defaults

| Archetype | Layer 1 (Inner Core) | Layer 2 (Mantle) | Layer 3 (Outer Shell) | Default Mass ($\mathrm{kg}$) | Default $T_{core}$ / $T_{surf}$ ($\mathrm{K}$) | Default Spin $\omega$ ($\mathrm{rad/s}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `LivingTerra` | `Iron` $32.0\%$ | `Granite` $67.5\%$ | `WaterIce` $0.5\%$ | $5.97219 \times 10^{24}$ | `5500` / `288` | `7.2921e-5` |
| `IronCore` | `Iron` $68.0\%$ | `Granite` $32.0\%$ | — | $3.3011 \times 10^{23}$ | `3200` / `440` | `1.24e-6` |
| `OceanIce` | `Iron` $15.0\%$ | `Granite` $45.0\%$ | `WaterIce` $40.0\%$ | $4.0 \times 10^{24}$ | `3800` / `275` | `5.0e-5` |
| `LavaIo` | `Iron` $25.0\%$ | `Granite` $75.0\%$ | — | $2.0 \times 10^{24}$ | `4200` / `1550` | `4.11e-5` |
| `GasGiant` | `Iron` $5.0\%$ | `Granite` $10.0\%$ | `HydrogenHelium` $85.0\%$ | $1.2 \times 10^{26}$ | `18000` / `165` | `1.76e-4` |
| `ProtoTheia` | `Iron` $22.0\%$ | `Granite` $58.0\%$ | `Regolith` $20.0\%$ | $6.417 \times 10^{23}$ | `2200` / `240` | `7.09e-5` |

#### Scientific Scenario Presets (`makePresetConfig`)

| `ScenarioPresetId` | Target Archetype & Mass | Impactor Archetype & Mass | $v_{imp} / v_{esc}$ | Impact Param $b$ | Initial Sep $(R_t + R_i)$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `EarthTheiaMoonForming` | `LivingTerra` ($1.0\,M_\oplus$) | `ProtoTheia` ($0.13\,M_\oplus$) | `1.05` | `0.72` | `3.0` |
| `MercuryMantleStrip` | `ProtoTheia` ($0.45\,M_\oplus$, $T_{surf}=440\,\mathrm{K}$) | `IronCore` ($0.22\,M_\oplus$) | `2.80` | `0.35` | `3.2` |
| `OceanWorldSteamPlume` | `OceanIce` ($0.75\,M_\oplus$) | `ProtoTheia` ($0.15\,M_\oplus$) | `1.35` | `0.25` | `3.2` |
| `GasGiantCoreImpact` | `GasGiant` ($20.0\,M_\oplus$) | `IronCore` ($1.0\,M_\oplus$) | `1.40` | `0.15` | `2.8` |
| `LivingGaiaAsteroidCataclysm` | `LivingTerra` ($1.0\,M_\oplus$) | `ProtoTheia` ($0.025\,M_\oplus$) | `1.65` | `0.30` | `3.2` |

### 4.5 CUDA Kernel & BVH Interfaces (`engine/sph_gravity.cuh`)

```cpp
namespace astraea::cuda {

struct RadixBvhDevice {
    uint32_t  capacity = 0;
    uint64_t* mortonCodes = nullptr;     // [N]
    uint64_t* mortonCodesSorted = nullptr;
    uint32_t* indicesIn = nullptr;       // [N]
    uint32_t* indicesSorted = nullptr;   // [N]

    // Internal nodes [0 .. N-2], Leaf nodes conceptually [N-1 .. 2N-2]
    int32_t*  leftChild = nullptr;       // >= 0: internal node, < 0: encoded leaf (~idx)
    int32_t*  rightChild = nullptr;
    int32_t*  parent = nullptr;          // [2N - 1]
    uint32_t* atomicVisited = nullptr;   // [N - 1]

    // Node bounding boxes & Barnes-Hut multipole moments (size 2N - 1)
    float*    aabbMinX = nullptr; float* aabbMinY = nullptr; float* aabbMinZ = nullptr;
    float*    aabbMaxX = nullptr; float* aabbMaxY = nullptr; float* aabbMaxZ = nullptr;
    float*    nodeMass = nullptr;
    float*    comX = nullptr;     float* comY = nullptr;     float* comZ = nullptr;
    float*    nodeMaxH = nullptr; // max smoothing support H_i in subtree (for SPH BVH query)

    void*     cubTempStorage = nullptr;
    size_t    cubTempBytes = 0;
};

void launchSortAndBuildBvh(ParticleBuffersDevice& p, ParticleBuffersDevice& alt, RadixBvhDevice& bvh);
void launchSphPass1Neighbors(ParticleBuffersDevice& p, const RadixBvhDevice& bvh);
void launchEvaluateEos(ParticleBuffersDevice& p);
void launchGravityBarnesHut(ParticleBuffersDevice& p, const RadixBvhDevice& bvh, float theta = 0.5f);
void launchSphPass2Forces(ParticleBuffersDevice& p, const RadixBvhDevice& bvh);
void launchLeapfrogKickHalf(ParticleBuffersDevice& p, float halfDtPsu, bool includeXsph, bool clampThermal);
void launchLeapfrogDrift(ParticleBuffersDevice& p, float dtPsu, const RadixBvhDevice& bvh);
float launchReduceDtPsu(const ParticleBuffersDevice& p, float prevDtPsu, float maxDtPsu);
void launchExtractTracers(const ParticleBuffersDevice& p, uint8_t* d_tracerRecords40B, uint32_t tracerCount, uint32_t tracerGridRes, float livingComX, float livingComY, float livingComZ);
void launchPackRenderRecords(const ParticleBuffersDevice& p, uint8_t* d_particleRecords20B, uint8_t* d_speedBytes, float sysComX, float sysComY, float sysComZ);

} // namespace astraea::cuda
```

### 4.6 Binary Frame & JSON Control Protocol (`server/binary_protocol.hpp`, `server/http_ws_server.hpp`, `web/src/net/BinaryProtocol.ts`)

#### Binary Frame Layout (Little-Endian, Zero Padding)

- **Header (`40 bytes`, offset `0..39`):**
  | Byte Offset | Type | Field | Description |
  | :--- | :--- | :--- | :--- |
  | `0..3` | `uint32` | `magic` | `0x50434C53` (`"PCLS"`) |
  | `4..5` | `uint16` | `version` | `1` |
  | `6..7` | `uint16` | `flags` | Bit 0: `zlib`, Bit 1: `hasTracers`, Bit 2: `hasLivingCom`, Bit 3: `hasSpeedArray` |
  | `8..15` | `float64` | `simTimeSec` | Simulation elapsed time in seconds |
  | `16..19` | `uint32` | `frameSeq` | Monotonic frame sequence id |
  | `20..23` | `float32` | `dtSec` | Current adaptive timestep in seconds |
  | `24..27` | `uint32` | `particleCount` | $N$ |
  | `28..31` | `uint32` | `tracerCount` | $T$ (`0` or `6 * tracerGridRes^2`) |
  | `32..35` | `uint32` | `tracerGridRes` | Cube-sphere tracer resolution per face (e.g. `16`) |
  | `36..39` | `uint32` | `payloadBytes` | Uncompressed payload byte length following header |

- **Particle Block (`20 bytes/particle`, starting at offset `40`):**
  | Byte Offset | Type | Field | Encoding / Scale |
  | :--- | :--- | :--- | :--- |
  | `0..3` | `float32` | `xKm` | COM-centered $x$ in km |
  | `4..7` | `float32` | `yKm` | COM-centered $y$ in km |
  | `8..11` | `float32` | `zKm` | COM-centered $z$ in km |
  | `12..13` | `uint16` | `tempLogU16` | $\mathrm{round}(65535 \cdot \mathrm{clamp}(\frac{\ln(T/10)}{\ln(20000)}, 0, 1))$ for $T \in [10, 200000]\ \mathrm{K}$ |
  | `14..15` | `uint16` | `pressureLogU16` | `0` if $P \le 10^{-3}\ \mathrm{GPa}$; else $\mathrm{round}(1 + 65534 \cdot \mathrm{clamp}(\frac{\ln(P_{GPa}/10^{-3})}{\ln(10^7)}, 0, 1))$ for $[10^{-3}, 10^4]\ \mathrm{GPa}$ |
  | `16` | `uint8` | `materialId` | `MaterialId` (`0..4`) |
  | `17` | `uint8` | `phaseAndBody` | `(bodyId << 4) \| (phaseId & 0x0F)` |
  | `18` | `uint8` | `damageU8` | $\mathrm{round}(\mathrm{clamp}(D, 0, 1) \cdot 255)$ |
  | `19` | `uint8` | `radiusLogU8` | $\mathrm{round}(255 \cdot \mathrm{clamp}(\frac{\ln(r_{km}/1.0)}{\ln(2000.0)}, 0, 1))$ for $r_{km} \in [1, 2000]\ \mathrm{km}$ |

- **Crust Tracer Block (`40 bytes/tracer`, starting at `40 + 20 * N`):**
  | Byte Offset | Type | Field | Description |
  | :--- | :--- | :--- | :--- |
  | `0..1` | `uint16` | `anchorIndex` | `face * tracerGridRes^2 + gy * tracerGridRes + gx` |
  | `2..3` | `uint16` | `flags` | Bit 0: `isOcean`, Bit 1: `isDisrupted` |
  | `4..7` | `float32` | `xKm` | Current position in authoritative system COM frame (km; already encodes planet rotation and tidal/impact deformation) |
  | `8..11` | `float32` | `yKm` | Current position in authoritative system COM frame (km; already encodes planet rotation and tidal/impact deformation) |
  | `12..15` | `float32` | `zKm` | Current position in authoritative system COM frame (km; already encodes planet rotation and tidal/impact deformation) |
  | `16..19` | `float32` | `dispKm` | Radial displacement $r - r_{rest}$ (km) |
  | `20..23` | `float32` | `normalDelta` | Tilt angle relative to rest normal (rad) |
  | `24..27` | `float32` | `tempK` | Local temperature (K) |
  | `28..31` | `float32` | `damage` | Local fracture damage $D \in [0, 1]$ |
  | `32..35` | `float32` | `overpressureKpa` | Local shock overpressure (kPa) |
  | `36..39` | `float32` | `tsunamiHeightM` | Local ocean displacement / wave height (m) |

- **Optional Velocity Speed Block (`1 byte/particle`, starting at `40 + 20*N + 40*T` when `flags & 0x08`):**
  - `uint8 speedLogU8`: $\mathrm{round}(255 \cdot \mathrm{clamp}(\frac{\ln(\max(v_{km/s}, 0.05) / 0.05)}{\ln(2000.0)}, 0, 1))$ covering $[0.05, 100.0]\ \mathrm{km/s}$.

#### C++ Server Contracts (`server/`)

```cpp
namespace astraea::server {

struct ServerConfig {
    std::string bindHost  = "127.0.0.1";
    uint16_t    port      = 8080;
    std::string webRoot   = "./web/dist";
    uint32_t    targetFps = 30;
    bool        compressBinaryZlib = false;
};

std::string sha1Base64AcceptKey(const std::string& clientWebSocketKey); // RFC 6455 exact accept key

class HttpWsServer {
public:
    HttpWsServer(Simulation& sim, const ServerConfig& cfg);
    ~HttpWsServer();

    bool start(); // binds socket; returns false if bind/listen fails
    void stop();
    void runEventLoop(); // blocks until stop() called
    uint16_t boundPort() const;
    size_t connectedClientCount() const;
};

} // namespace astraea::server
```

#### WebSocket JSON Control & Telemetry Schemas

- **Client $\to$ Server Commands (UTF-8 Text Frames, `opcode = 0x1`):**
  - `{"type":"set_preset","preset":"LivingGaiaAsteroidCataclysm","particles":100000}`
  - `{"type":"launch_custom","config":{"targetArchetype":"LivingTerra","impactorArchetype":"ProtoTheia","targetMassKg":5.97219e24,"impactorMassKg":6.417e23,"impactVelocityRatioVesc":1.15,"impactParamB":0.45,"initialSepRadii":3.2,"targetSpinRadS":7.2921e-5,"particles":100000}}`
  - `{"type":"play"}` / `{"type":"pause"}` / `{"type":"step","count":1}`
  - `{"type":"set_timescale","scale":1.0}` (clamped `[0.1, 100.0]`)
  - `{"type":"set_slice_plane","enabled":true,"normal":[0,1,0],"offsetKm":0.0}`
  - `{"type":"set_vis_mode","mode":"NaturalBlackbody"|"TemperatureK"|"PressureGPa"|"MaterialLayer"|"DamageFracture"|"Velocity"}` (enables `flags & 0x08` speed byte stream when `"Velocity"`)
  - `{"type":"request_export","format":"csv"|"json"}`
  - `{"type":"ping","ts":1700000000}`

- **Server $\to$ Client Messages (UTF-8 Text Frames, `opcode = 0x1`):**
  - `{"type":"telemetry", ...TelemetrySnapshotDto}`
  - `{"type":"preset_loaded","preset":string,"particleCount":number,"tracerCount":number,"tracerGridRes":number,"livingBodyPresent":boolean}`
  - `{"type":"export_data","format":"csv"|"json","filename":string,"content":string}`
  - `{"type":"pong","ts":number}`
  - `{"type":"error","code":string,"message":string}`

#### TypeScript Client Contracts (`web/src/`)

```typescript
// web/src/net/SimClient.ts
export interface TelemetrySnapshotDto {
  type: 'telemetry';
  frameSeq: number;
  simTimeSec: number;
  dtSec: number;
  state: 'paused' | 'playing';
  preset: string;
  outcome: 'Approaching' | 'Merging' | 'GrazeAndMerge' | 'HitAndRun' | 'CatastrophicDisruption';
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

export type ClientCommand =
  | { type: 'set_preset'; preset: string; particles?: number }
  | { type: 'launch_custom'; config: CustomScenarioDto }
  | { type: 'play' }
  | { type: 'pause' }
  | { type: 'step'; count: number }
  | { type: 'set_timescale'; scale: number }
  | { type: 'set_slice_plane'; enabled: boolean; normal: [number, number, number]; offsetKm: number }
  | { type: 'set_vis_mode'; mode: string }
  | { type: 'request_export'; format: 'csv' | 'json' }
  | { type: 'ping'; ts: number };

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

export class SimClient {
  constructor(url: string, particlePool: ParticleBufferPool, tracerPool: TracerBufferPool);
  connect(): void;
  disconnect(): void;
  sendCommand(cmd: ClientCommand): void;
  onFrame(cb: (header: DecodedFrameHeader) => void): void;
  onTelemetry(cb: (dto: TelemetrySnapshotDto) => void): void;
  onPresetLoaded(cb: (msg: { preset: string; particleCount: number; tracerCount: number; tracerGridRes: number; livingBodyPresent: boolean }) => void): void;
  onExportData(cb: (msg: { format: 'csv' | 'json'; filename: string; content: string }) => void): void;
}
```

```typescript
// web/src/net/BinaryProtocol.ts
export const PCLS_MAGIC = 0x50434c53;
export const HEADER_BYTES = 40;
export const PARTICLE_RECORD_BYTES = 20;
export const TRACER_RECORD_BYTES = 40;

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
  positionsKm: Float32Array;  // length 3 * capacity
  props: Float32Array;        // length 4 * capacity: [tempK, pressureGPa, damage01, radiusKm]
  meta: Float32Array;         // length 4 * capacity: [materialId, phaseId, bodyId, speedKmS]
}

export interface TracerBufferPool {
  capacity: number;
  count: number;
  gridRes: number;
  anchorIndex: Uint16Array;
  flags: Uint16Array;
  positionsKm: Float32Array;  // 3 * capacity
  dispKm: Float32Array;
  normalDelta: Float32Array;
  tempK: Float32Array;
  damage: Float32Array;
  overpressureKpa: Float32Array;
  tsunamiHeightM: Float32Array;
}

export function decodeBinaryFrame(
  buffer: ArrayBuffer,
  particles: ParticleBufferPool,
  tracers: TracerBufferPool
): DecodedFrameHeader;
```

```typescript
// web/src/render/OrbitalSurfaceCamera.ts
export type CameraMode = 'OrbitalFree' | 'SurfaceLock';

export interface SurfaceAnchorCoord {
  face: number;      // 0..5
  u: number;         // 0..1
  v: number;         // 0..1
}

export class OrbitalSurfaceCamera {
  readonly camera: THREE.PerspectiveCamera;
  mode: CameraMode;
  altitudeKm: number;           // 0.001 km (1 m) to 1.0e5 km (10^8 m)
  worldPositionKm: [number, number, number]; // Float64 precision!
  surfaceAnchor: SurfaceAnchorCoord;

  setMode(mode: CameraMode, anchor?: SurfaceAnchorCoord): void;
  zoomByWheel(deltaY: number): void;
  orbitByDelta(dx: number, dy: number): void;
  // Resolves camera worldPositionKm in the authoritative system COM frame (matching CubeSphereLod):
  // SurfaceLock anchors to the deformed/rotated COM-frame surface point when tracers are active,
  // falling back to livingComKm + R * rotate(unitNormal, spinAngleRad) only when tracerCount == 0.
  update(
    dtSec: number,
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    sampleSurfaceDisplacementKm: (face: number, u: number, v: number) => number
  ): void;
  toCameraRelativePos(worldKm: [number, number, number], outVec3: THREE.Vector3): THREE.Vector3;
}
```

```typescript
// web/src/render/SphParticleRenderer.ts
export type VisMode =
  | 'NaturalBlackbody'
  | 'TemperatureK'
  | 'PressureGPa'
  | 'MaterialLayer'
  | 'DamageFracture'
  | 'Velocity';

export class SphParticleRenderer {
  readonly mesh: THREE.Points;
  constructor(maxParticles: number);
  updateFromPool(pool: ParticleBufferPool, cameraPosKm: [number, number, number]): void;
  setVisMode(mode: VisMode): void;
  setSlicePlane(enabled: boolean, normal: THREE.Vector3, offsetKm: number, comKm: [number, number, number]): void;
  dispose(): void;
}
```

```typescript
// web/src/render/CubeSphereLod.ts
export class CubeSphereLod {
  readonly group: THREE.Group;
  readonly maxDepth = 10;
  constructor(biomeSystem: BiomeSystem);
  // Computes each vertex position in system COM coordinates directly by bilinearly interpolating
  // surrounding CrustTracerRecord (xKm, yKm, zKm) positions plus sub-grid radial fBm relief +
  // tsunamiHeightM along the interpolated surface normal; falls back to
  // livingComKm + R * rotate(unitNormal, spinAngleRad) only when tracers.count === 0.
  update(
    camera: OrbitalSurfaceCamera,
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number,
    tracers: TracerBufferPool,
    coupler: ImpactBiosphereCoupler
  ): void;
  sampleTracerField(face: number, u: number, v: number, tracers: TracerBufferPool): {
    dispKm: number; tempK: number; damage: number; overpressureKpa: number; tsunamiHeightM: number; isOcean: boolean;
  };
  getActivePatchCount(): number;
  getCulledDamagedPatchCount(): number;
  dispose(): void;
}
```

```typescript
// web/src/biosphere/BiomeSystem.ts
export type BiomeId =
  | 'TropicalRainforest'
  | 'Savanna'
  | 'TemperateForest'
  | 'BorealTaiga'
  | 'AlpineTundra'
  | 'AridSteppe'
  | 'OceanReef';

export interface SurfaceSample {
  elevationM: number;
  latitudeRad: number;
  moisture01: number;
  climateTempK: number;
  biome: BiomeId;
  albedoRgb: [number, number, number];
}

export class BiomeSystem {
  constructor(seed?: number);
  sampleSurface(unitNormal: THREE.Vector3): SurfaceSample;
  getElevationM(unitNormal: THREE.Vector3): number;
  static classifyBiome(elevationM: number, latitudeRad: number, moisture01: number, climateTempK: number): BiomeId;
}
```

```typescript
// web/src/biosphere/ProceduralFlora.ts
export type FloraSpeciesId =
  | 'ButtressCanopyTree'
  | 'CoastalPalm'
  | 'AcaciaSavannaTree'
  | 'BroadleafOak'
  | 'BorealConifer'
  | 'GiantFern'
  | 'DesertSaguaroCactus'
  | 'CoralColony'
  | 'GiantKelp';

export function buildFloraSpeciesGeometry(species: FloraSpeciesId, seed: number): THREE.BufferGeometry;

export class ProceduralFlora {
  readonly group: THREE.Group;
  constructor(biomeSystem: BiomeSystem, maxInstances?: number);
  populateChunk(
    chunkCenterUnit: THREE.Vector3,
    planetRadiusKm: number,
    cameraPosKm: [number, number, number],
    livingComKm: [number, number, number],
    spinAngleRad: number
  ): void;
  updateImpactResponse(
    timeSec: number,
    pgaG: number,
    overpressureKpa: number,
    blastDirLocal: THREE.Vector3,
    localTempK: number,
    radFluxKwM2: number
  ): void;
  getSpeciesCount(): number; // >= 8 (returns 9)
  dispose(): void;
}
```

```typescript
// web/src/biosphere/ProceduralFauna.ts
export type FaunaGuild = 'HerbivoreHerd' | 'PursuitPredator' | 'AvianFlock' | 'PelagicMarine';
export type FaunaState = 'Grazing' | 'Alert' | 'StampedeOrMassFlight' | 'ThermalEscapeOrBurrow' | 'Lethal';

export interface FaunaStimulus {
  tImpactElapsedSec: number;
  impactorDistRadii: number;
  pgaG: number;
  overpressureKpa: number;
  localTempK: number;
  radFluxKwM2: number;
  ejectaFluxKgM2: number;
  tsunamiHeightM: number;
  crustDamage: number;
  impactAzimuthLocal: THREE.Vector3;
}

export function transitionFaunaState(current: FaunaState, guild: FaunaGuild, stim: FaunaStimulus, isLand: boolean): FaunaState;

export class ProceduralFauna {
  readonly group: THREE.Group;
  constructor(biomeSystem: BiomeSystem, maxAgents?: number);
  spawnChunkAgents(chunkCenterUnit: THREE.Vector3, planetRadiusKm: number, seed: number): void;
  step(
    dtSec: number,
    timeSec: number,
    stim: FaunaStimulus,
    cameraPosKm: [number, number, number],
    livingComKm: [number, number, number],
    livingRadiusKm: number,
    spinAngleRad: number
  ): void;
  getStateCounts(): Record<FaunaState, number>;
  dispose(): void;
}
```

```typescript
// web/src/biosphere/ImpactBiosphereCoupler.ts
export type ExtinctionTier = 'Nominal' | 'RegionalCrisis' | 'MassExtinction' | 'Sterilization';

export interface SitePhysicalExposure {
  siteNormal: THREE.Vector3;
  arcDistanceM: number;
  chordDistanceM: number;
  tsunamiHeightM: number;
  pWaveArrived: boolean;
  sWaveArrived: boolean;
  pgaG: number;
  radFluxKwM2: number;
  isWildfireIgnited: boolean;
  ejectaArrived: boolean;
  ejectaFluxKgM2: number;
  blastArrived: boolean;
  overpressureKpa: number;
  effectiveTempK: number;
  crustDamage: number;
}

export interface BiosphereMetrics {
  totalBiomassGtC: number;
  baselineBiomassGtC: number;
  shannonDiversityH: number;
  overallSurvivalPct: number;
  globalOpticalDepthTau: number;
  globalDeltaTempK: number;
  survivalByBiome: Record<BiomeId, number>;
  survivalByGuild: Record<FaunaGuild | 'Flora', number>;
  extinctionTier: ExtinctionTier;
}

export class ImpactBiosphereCoupler {
  static readonly BASELINE_BIOMASS_GTC = 550.0;
  constructor(biomeSystem: BiomeSystem, sampleTileCount?: number);
  resetBaseline(): void;
  ingestSimulationState(
    telemetry: TelemetrySnapshotDto,
    tracers: TracerBufferPool,
    particles: ParticleBufferPool
  ): void;
  evaluateSiteExposure(unitNormal: THREE.Vector3, simTimeSec: number): SitePhysicalExposure;
  getGlobalMetrics(): BiosphereMetrics;
}

// web/src/render/AtmosphereShader.ts
export class AtmosphereShader {
  readonly mesh: THREE.Mesh;
  constructor(segments?: number);
  update(
    cameraPosKm: [number, number, number],
    planetComKm: [number, number, number],
    planetRadiusKm: number,
    sunDir: THREE.Vector3,
    opticalDepthTau: number,
    plumeGlowTempK: number
  ): void;
  dispose(): void;
}

// web/src/ui/TelemetryCharts.ts
export class TelemetryCharts {
  constructor(container: HTMLElement);
  pushSample(t: TelemetrySnapshotDto, bio: BiosphereMetrics): void;
  clear(): void;
}

// web/src/ui/ControlPanel.ts
export class ControlPanel {
  constructor(container: HTMLElement, callbacks: {
    onPreset: (preset: string, particles: number) => void;
    onLaunchCustom: (cfg: CustomScenarioDto) => void;
    onPlayPause: (playing: boolean) => void;
    onStep: () => void;
    onTimescale: (scale: number) => void;
    onVisMode: (mode: VisMode) => void;
    onSliceChange: (enabled: boolean, normal: [number, number, number], offsetKm: number) => void;
    onCameraMode: (mode: CameraMode, anchor?: SurfaceAnchorCoord) => void;
    onExport: (format: 'csv' | 'json') => void;
  });
  updateState(telemetry: TelemetrySnapshotDto, bio: BiosphereMetrics): void;
}

// web/src/ui/SimulationApp.ts
export class SimulationApp {
  constructor(canvasContainer: HTMLElement, uiContainer: HTMLElement, wsUrl?: string);
  start(): void;
  dispose(): void;
}
```

#### Deterministic Biome Classification & Flora Species Mapping

- **`BiomeSystem.classifyBiome(elevationM, latitudeRad, moisture01, climateTempK)`:**
  - `elevationM < 0.0` $\implies$ `OceanReef`
  - `elevationM >= 3200.0` or `climateTempK < 268.0` $\implies$ `AlpineTundra`
  - `absLatDeg > 55.0` (i.e. $|\mathrm{lat}| > 0.96\ \mathrm{rad}$) $\implies$ `BorealTaiga` (if `moisture01 >= 0.30`) else `AlpineTundra`
  - `climateTempK >= 293.0`: `moisture01 >= 0.62` $\implies$ `TropicalRainforest`, `moisture01 >= 0.28` $\implies$ `Savanna`, else `AridSteppe`
  - Otherwise (`268.0 <= climateTempK < 293.0`): `moisture01 >= 0.40` $\implies$ `TemperateForest`, else `AridSteppe`
- **Flora Biome Placement (`ProceduralFlora`):**
  - `TropicalRainforest`: `ButtressCanopyTree`, `CoastalPalm`, `GiantFern`
  - `Savanna`: `AcaciaSavannaTree`, `CoastalPalm`
  - `TemperateForest`: `BroadleafOak`, `GiantFern`
  - `BorealTaiga`: `BorealConifer`
  - `AridSteppe` / `AlpineTundra`: `DesertSaguaroCactus` (in warm arid) / stunted `BorealConifer` shrub
  - `OceanReef` (shallow `-150 m < elevationM < -1 m`): `CoralColony`, `GiantKelp`

---

## 5. Behavior & Exact Algorithms

### 5.1 Self-Gravity Architecture: Options & Justification

1. **Direct All-Pairs $N^2$ Gravity:** Exact, $O(N^2)$ tiled shared-memory kernel. At $N = 100{,}000$, $10^{10}$ pairwise evaluations per step require $\sim 280\ \mathrm{ms}$ on RTX 4050 — fails real-time interactive budgets and becomes $1.75\ \mathrm{s/step}$ at $N = 250{,}000$.
2. **Uniform Grid Particle-Mesh FFT (cuFFT):** $O(N + M\log M)$. Fails planetary collision dynamics because colliding planets sit inside huge vacuum domains ($10^8\ \mathrm{m}$ arena) while requiring sub-$50\ \mathrm{km}$ core resolution; uniform grids either blow past 6 GiB VRAM or smear cores into single grid cells.
3. **3D Morton-Sorted Karras Linear BVH + Barnes-Hut ($\theta = 0.5$) [CHOSEN]:**
   - Compute scene AABB via `cub::DeviceReduce::Min`/`Max`.
   - Map normalized particle coordinates $u,v,w \in [0, 1]$ to **64-bit Morton codes** (21 bits per axis interleaved) via `morton3D_64`, preserving spatial locality down to $\sim 1 / 2^{21}$ of domain width.
   - Sort `(mortonCodes, indices)` using `cub::DeviceRadixSort::SortPairs`, then permute all SoA particle arrays into Morton Z-order.
   - Construct binary radix tree of $N-1$ internal nodes and $N$ leaves via Karras (2012) parallel prefix XOR (`__clzll(code[i] ^ code[j])`, breaking identical-code ties with `i ^ j`).
   - Bottom-up atomic accumulation (`atomicAdd` arrival flag per internal node) aggregates `aabbMin`, `aabbMax`, `nodeMass`, `comX/Y/Z`, and `nodeMaxH` across internal nodes.
   - Per-particle stack-based traversal (`int stack[64]`): accept internal node $u$ as monopole if $d_u^2 / \|\mathbf{r}_{com,u} - \mathbf{r}_i\|^2 < \theta^2$ ($\theta = 0.5 \implies \theta^2 = 0.25$) and $\mathbf{r}_i$ lies outside node $u$'s AABB; otherwise descend into left/right children.
   - **Conservative Particle-Constant Plummer Softening:**
     $$\epsilon_i = 0.8\, h_{0,i}, \quad \epsilon_{ij} = \frac{1}{2}(\epsilon_i + \epsilon_j)$$
     For leaf interaction $(i, j)$:
     $$\mathbf{a}_{ij} = -G_{PSU} m_j \frac{\mathbf{r}_i - \mathbf{r}_j}{\left(\|\mathbf{r}_i - \mathbf{r}_j\|^2 + \epsilon_{ij}^2\right)^{3/2}}, \quad \Phi_{ij} = -\frac{G_{PSU} m_j}{\sqrt{\|\mathbf{r}_i - \mathbf{r}_j\|^2 + \epsilon_{ij}^2}}$$
     For accepted internal node $u$, substitute $\epsilon_{i,u} = \epsilon_i$. Because $\epsilon_i$ depends only on fixed initial spacing $h_{0,i}$, gravitational forces are strictly conservative ($\mathbf{F} = -\nabla \Phi$) with zero varying-$h$ energy drift.
   - **Bonus Reuse:** The exact same Karras BVH and `nodeMaxH` enable $O(N\log N)$ stack-based SPH neighbor queries (`r_ij <= max(H_i, H_j)`).

### 5.2 3D Hydrodynamics (SPH): Multi-Material Density Handling & Equations

- **Multi-Material Density Trade-off & Choice:**
  1. *Standard Density Summation ($\rho_i = \sum_j m_j W_{ij}$):* Blurs sharp core-mantle ($\rho = 11{,}000 \to 4{,}500\ \mathrm{kg/m^3}$) and ocean-mantle interfaces, inducing artificial tensile surface-tension gaps that suppress Kelvin-Helmholtz mixing.
  2. *Pressure-Energy DISPH (Saitoh & Makino 2013):* Assumes ideal-gas pressure linearity $P = (\gamma - 1)\rho u$; fails for Tillotson solid/liquid EOS where tension ($P < 0$) and quadratic $B\mu^2$ compression dominate.
  3. **$\delta$-SPH Continuity Density + Same-Material Acoustic Diffusion [CHOSEN]:**
     $$\frac{d\rho_i}{dt} = \rho_i \sum_j \frac{m_j}{\rho_j} (\mathbf{v}_i - \mathbf{v}_j) \cdot \nabla_i \bar{W}_{ij} + \delta_{SPH} H_i c_{s,i} \sum_{j:\,\mathrm{mat}_j = \mathrm{mat}_i} \frac{m_j}{\rho_j} \left[2(\rho_j - \rho_i) \frac{\mathbf{r}_j - \mathbf{r}_i}{r_{ij}^2 + 0.01 H_i^2}\right] \cdot \nabla_i \bar{W}_{ij}$$
     with $\delta_{SPH} = 0.05$. Because uniform translation yields $\frac{d\rho_i}{dt} = 0$ identically and diffusion acts strictly within identical `MaterialId` domains, sharp density discontinuities across Core-Mantle-Ocean-Gas boundaries are preserved without spurious tension gaps, while acoustic noise inside single-material regions is suppressed.

- **3D Wendland $C^2$ Kernel (Support Radius $H = 2h$, $q = r/H \in [0, 1]$):**
  $$W(r, H) = \frac{21}{2\pi H^3} (1 - q)^4 (1 + 4q), \qquad \nabla_i W(r_{ij}, H) = -\frac{210}{\pi H^5} (1 - q)^3 (\mathbf{r}_i - \mathbf{r}_j)$$
  Symmetric pairwise kernel gradient: $\nabla_i \bar{W}_{ij} = \frac{1}{2}\left(\nabla_i W(r_{ij}, H_i) + \nabla_i W(r_{ij}, H_j)\right)$.
  Adaptive smoothing length update: compute local number density $n_i = \sum_j W(r_{ij}, H_i)$ (and neighbor count $N_{ngb,i}$), then update $h_i \leftarrow h_i \cdot \mathrm{clamp}\left((56.0 / \max(N_{ngb,i}, 1))^{1/3},\ 0.92,\ 1.08\right)$ clamped to $[0.35 h_{0,i},\ 6.0 h_{0,i}]$.

- **Momentum, Monaghan Artificial Viscosity ($\alpha=1.0, \beta=2.0$), Balsara Switch, Artificial Stress & Energy:**
  - Pass 1 computes Shepard normalization $C_i = \sum_j \frac{m_j}{\rho_j} W_{ij}$, surface centroid offset $\mathbf{n}_{surf,i} = \sum_j \frac{m_j}{\rho_j}(\mathbf{r}_j - \mathbf{r}_i) W_{ij}$, and velocity gradient $\nabla \mathbf{v}_i = \sum_j \frac{m_j}{\rho_j}(\mathbf{v}_j - \mathbf{v}_i) \otimes \nabla_i W(r_{ij}, H_i)$ from which $\nabla \cdot \mathbf{v}_i$, $\nabla \times \mathbf{v}_i$, strain rate $\dot{\epsilon}_i^{\alpha\beta}$, spin $R_i^{\alpha\beta}$, and Balsara switch
    $$f_i = \frac{|\nabla \cdot \mathbf{v}_i|}{|\nabla \cdot \mathbf{v}_i| + |\nabla \times \mathbf{v}_i| + 10^{-4} c_{s,i}/h_i}$$
    are evaluated.
  - Artificial viscosity (for approaching pairs $\mathbf{v}_{ij} \cdot \mathbf{r}_{ij} < 0$):
    $$\mu_{ij} = \frac{\bar{h}_{ij} (\mathbf{v}_i - \mathbf{v}_j)\cdot(\mathbf{r}_i - \mathbf{r}_j)}{r_{ij}^2 + 0.01 \bar{h}_{ij}^2}, \qquad \Pi_{ij} = \frac{-\alpha_{AV}\bar{c}_{ij}\mu_{ij} + \beta_{AV}\mu_{ij}^2}{\bar{\rho}_{ij}} \cdot \frac{f_i + f_j}{2}$$
  - Tensile instability mitigation (Monaghan 2000): when $P_i < 0$, $R_i = 0.3 \frac{|P_i|}{\rho_i^2}$ (else $0$), multiplied by $f_{ij}^4 = \left[W(r_{ij}, \bar{H}_{ij}) / W(0.5 \bar{h}_{ij}, \bar{H}_{ij})\right]^4$.
  - Momentum and specific internal energy equations:
    $$\frac{d v_i^\alpha}{dt} = a_{grav,i}^\alpha - \sum_j m_j \left(\frac{P_i}{\rho_i^2} + \frac{P_j}{\rho_j^2} + \Pi_{ij} + (R_i + R_j) f_{ij}^4\right) \frac{\partial \bar{W}_{ij}}{\partial x_i^\alpha} + \sum_j m_j \left(\frac{S_i^{\alpha\beta}}{\rho_i^2} + \frac{S_j^{\alpha\beta}}{\rho_j^2}\right) \frac{\partial \bar{W}_{ij}}{\partial x_i^\beta}$$
    $$\frac{d u_i}{dt} = \frac{P_i}{\rho_i^2} \sum_j m_j (\mathbf{v}_i - \mathbf{v}_j) \cdot \nabla_i \bar{W}_{ij} + \frac{1}{2}\sum_j m_j \Pi_{ij} (\mathbf{v}_i - \mathbf{v}_j)\cdot \nabla_i \bar{W}_{ij} + \frac{1}{\rho_i} S_i^{\alpha\beta} \dot{\epsilon}_i^{\alpha\beta} - \frac{\dot{E}_{rad,i}}{m_i}$$
  - XSPH drift velocity (applied strictly to position drift $\frac{d\mathbf{r}_i}{dt}$, preserving linear/angular momentum):
    $$\mathbf{v}_{XSPH,i} = \mathbf{v}_i + 0.25 \sum_j \frac{m_j}{\bar{\rho}_{ij}}(\mathbf{v}_j - \mathbf{v}_i)\bar{W}_{ij}$$

### 5.3 Equations of State, Strength, Brittle Damage & Radiative Cooling

- **Tillotson EOS ($\eta = \rho / \rho_0$, $\mu = \eta - 1$):**
  Published Tillotson parameters anchor $\rho_0$ at STP reference temperature $T_{ref}$ (`tRefK`: $293.0\ \mathrm{K}$ for `Iron`/`Granite`/`Regolith`, $273.15\ \mathrm{K}$ for `WaterIce`, $0.0\ \mathrm{K}$ for `HydrogenHelium`). Total specific internal energy $u$ is stored from $0\ \mathrm{K}$ so $T = \max(3.0, (u - u_{cold})/c_v)$ and energy conservation remains exact, whereas `evaluateEosPsu` and `evaluateFromRhoTempSI` evaluate Tillotson regime thresholds, pressures $P_c, P_e$, and sound speed $c_s$ using the reference-shifted thermal excess:
  $$u_{EOS} = u_{cold}(\rho) + \max\!\left(0.0,\ u - u_{cold}(\rho) - c_v T_{ref}\right)$$
  (Note: for `WaterIce` at $T = 288\ \mathrm{K}$ near $\rho_0 = 1000\ \mathrm{kg/m^3}$, $u - c_v T_{ref} = c_v(288 - 273.15) \approx 6.21\times 10^4\ \mathrm{J/kg} \ll E_{iv} = 4.19\times 10^5\ \mathrm{J/kg}$, so a valid zero-pressure liquid water root exists near $\rho_0 = 1000\ \mathrm{kg/m^3}$.)
  1. **Compressed or Cold ($\eta \ge 1$ or $u_{EOS} < E_{iv}$):**
     With $B_{eff} = (\mu \ge 0) ? B : 0$ (prevents spurious re-compression in deep expansion):
     $$P_c(\rho, u_{EOS}) = \left[a + \frac{b}{\frac{u_{EOS}}{E_0 \eta^2} + 1}\right] \rho u_{EOS} + A\mu + B_{eff}\mu^2$$
     For damaged tensile states ($P_c < 0$), attenuate tension by intact fraction: $P_c \leftarrow (1 - D) P_c$.
  2. **Hot Expanded ($\eta < 1$ and $u_{EOS} \ge E_{cv}$):**
     With $x = \frac{\rho_0}{\rho} - 1$:
     $$P_e(\rho, u_{EOS}) = a\rho u_{EOS} + \left[\frac{b\rho u_{EOS}}{\frac{u_{EOS}}{E_0\eta^2} + 1} + A\mu e^{-\beta x}\right] e^{-\alpha x^2}$$
  3. **Two-Phase Transition ($\eta < 1$ and $E_{iv} \le u_{EOS} < E_{cv}$):**
     $$P(\rho, u_{EOS}) = \frac{(u_{EOS} - E_{iv}) P_e(\rho, u_{EOS}) + (E_{cv} - u_{EOS}) P_c(\rho, u_{EOS})}{E_{cv} - E_{iv}}$$
  4. **Analytical Sound Speed:**
     Compute $c_s^2 = \left(\frac{\partial P}{\partial \rho}\right)_{u_{EOS}} + \frac{P}{\rho^2}\left(\frac{\partial P}{\partial u_{EOS}}\right)_\rho$, clamped below at $c_{min} = 0.05\ \mathrm{km/s}$ (or $\sqrt{A/\rho_0}$ for condensed states).
     Explicit derivatives in compressed regime ($w = \frac{u_{EOS}}{E_0 \eta^2} + 1$):
     $$\left(\frac{\partial P_c}{\partial \rho}\right)_{u_{EOS}} = \left(a + \frac{b}{w}\right)u_{EOS} + \frac{2 b u_{EOS} (w - 1)}{w^2} + \frac{A + 2 B_{eff}\mu}{\rho_0}, \qquad \left(\frac{\partial P_c}{\partial u_{EOS}}\right)_\rho = \left(a + \frac{b}{w^2}\right)\rho$$
     and in expanded regime ($e_\alpha = e^{-\alpha x^2}$, $e_\beta = e^{-\beta x}$, $\frac{\partial x}{\partial \rho} = -\frac{\rho_0}{\rho^2}$):
     $$\left(\frac{\partial P_e}{\partial \rho}\right)_{u_{EOS}} = a u_{EOS} + e_\alpha \left[ \frac{b u_{EOS}}{w} + \frac{2 b u_{EOS} (w-1)}{w^2} + \left(\frac{A}{\rho_0} + A\mu \beta \frac{\rho_0}{\rho^2}\right)e_\beta \right] + 2\alpha x \frac{\rho_0}{\rho^2} (P_e - a\rho u_{EOS})$$
     $$\left(\frac{\partial P_e}{\partial u_{EOS}}\right)_\rho = \left(a + \frac{b e_\alpha}{w^2}\right)\rho$$
  5. **`HydrogenHelium` EOS:**
     $$P(\rho, u) = K_{H_2}\rho^2 + (\gamma - 1)\rho u_{th}, \qquad u_{cold}(\rho) = K_{H_2}\rho, \qquad u_{th} = \max(0, u - u_{cold}(\rho))$$
     $$c_s^2 = 2 K_{H_2}\rho + \gamma(\gamma - 1)u_{th}$$
     (exact thermodynamic identity since $P = (2-\gamma)K_{H_2}\rho^2 + (\gamma-1)\rho u$ for $u \ge u_{cold}$).
  6. **Cold Compression Energy & Temperature:**
     For Tillotson solids ($\eta > 1$):
     $$u_{cold}(\rho) = \frac{1}{\rho_0}\left[A\left(\ln\eta + \frac{1}{\eta} - 1\right) + B\left(\eta - \frac{1}{\eta} - 2\ln\eta\right)\right]$$
     ($0$ for $\eta \le 1$). Temperature (from total stored $u$): $T(\rho, u) = \max\left(3.0,\ \frac{u - u_{cold}(\rho)}{c_v}\right)$.

- **Strength, Brittle Damage & Radiative Cooling:**
  - Jaumann stress update: $\dot{S}^{\alpha\beta} = 2\mu_{eff}\dot{\epsilon}^{\alpha\beta} + S^{\alpha\gamma}R^{\beta\gamma} + S^{\gamma\beta}R^{\alpha\gamma}$ where $\mu_{eff} = (1-D)\mu_{shear}\max(0, 1 - (T/T_{melt})^2)$.
  - Von Mises + granular friction yield: $Y = \max\left((1-D)Y_0 \max(0, 1 - T/T_{melt}),\ D\min(Y_0, 0.6\max(P, 0))\right)$. Radial return scales $S^{\alpha\beta} \leftarrow S^{\alpha\beta} \cdot \min(1, Y/\sqrt{3 J_2 + 10^{-30}})$.
  - Damage evolution: compute max principal tensile stress $\sigma_{max}$ of $\sigma^{\alpha\beta} = -P\delta^{\alpha\beta} + S^{\alpha\beta}$; $\epsilon_{tens} = \frac{\max(0, \sigma_{max})}{A + \frac{4}{3}\mu_{shear}}$. When $\epsilon_{tens} > \epsilon_{fail,i}$ and solid ($T < T_{melt}$), $\frac{d D^{1/3}}{dt} = \frac{0.4 c_s}{h_i}$, $D = \mathrm{clamp}(D^{1/3}, 0, 1)^3$.
  - Optical surface exposure factor:
    $$f_{surf,i} = \mathrm{clamp}\left(1.6 \max\left(1 - C_i,\ \frac{\|\mathbf{n}_{surf,i}\|}{h_i}\right) - 0.15,\ 0.0,\ 1.0\right)$$
    $$\dot{E}_{rad,i} = f_{surf,i} \cdot 0.9\, \sigma_{SB} \left(\pi \left(\frac{3 m_i}{4\pi \rho_i}\right)^{2/3}\right) \left(T_i^4 - T_{amb}^4\right)$$
    where $T_{amb} = 255.0\ \mathrm{K}$ if `LivingTerra` is intact ($D < 0.8$), else $3.0\ \mathrm{K}$.
  - Phase classification (`PhaseId`):
    - `HydrogenHelium`: $T > 10{,}000\ \mathrm{K} \to$ `Plasma`, else `Gas`.
    - `WaterIce`: $T > 15{,}000\ \mathrm{K} \to$ `Plasma`; $u \ge E_{cv} \lor (T > 647\ \mathrm{K} \land \rho < 322) \to$ `Vapor`; $T \ge 273.15\ \mathrm{K} \to$ `LiquidWater`; else `Solid`.
    - `Iron`, `Granite`, `Regolith`: $T > 12{,}000\ \mathrm{K} \to$ `Plasma`; $u \ge E_{cv} \land \rho < 0.8\rho_0 \to$ `Vapor`; $T \ge T_{melt,0} \to$ `Molten`; else `Solid`.

### 5.4 Adaptive Symplectic KDK Leapfrog & 1D Planet Relaxation

- Global $\Delta t$ criterion:
  $$\Delta t_{new} = \mathrm{clamp}\left(\min_i \left[0.25\frac{h_i}{c_{s,i} + \|\mathbf{v}_i\| + 1.2 c_{s,i} + 10^{-6}},\ 0.25\sqrt{\frac{h_i}{\|\mathbf{a}_i\| + 10^{-8}}},\ 0.20\frac{u_i + 0.01 u_{0,i}}{|\dot{u}_i| + 10^{-8}}\right],\ 10^{-5},\ \min(\Delta t_{prev}\cdot 1.15, \Delta t_{max})\right)$$
- **KDK Step:** Half-kick $(\mathbf{v}, u, \rho, S, D)$ by $\Delta t/2$ $\to$ Drift $\mathbf{r}$ by $\Delta t \, \mathbf{v}_{XSPH}$ (plus predictor $(\rho, u)$ to $t+\Delta t$) $\to$ Morton sort & Karras BVH $\to$ SPH Pass 1 $\to$ EOS $\to$ Gravity + SPH Pass 2 $\to$ Half-kick by $\Delta t/2$ + radial return.
- **Planet Builder & Relaxation:**
  - `solveHydrostatic1D` solves $\frac{dP}{dr} = -\frac{G M(r) \rho(r)}{r^2},\ \frac{dM}{dr} = 4\pi r^2\rho(r)$ by RK4 outward from $r=0$ using bisection on $\log P_c$ until $M(R) = M_{target}$ within $10^{-5}$. The outward RK4 integration stops at the outer surface boundary $P \le P_{surf}$, where $P_{surf} = 10^5\ \mathrm{Pa}$ ($1\ \mathrm{bar}$) for `GasGiant` (clamping gas density $\rho \ge 10^{-3}\ \mathrm{kg/m^3}$) and $P_{surf} = 0\ \mathrm{Pa}$ (zero-pressure equilibrium root $\rho_{eq}(T_{surf})$) for rocky/oceanic planets.
  - `buildPlanet` partitions mass into radial spherical shells with isotropic Fibonacci golden-angle sampling ($\Delta r \approx \sqrt{4\pi r^2 / n_k}$).
  - `relaxBody` steps the single planet with $\mathbf{v}_i \leftarrow 0.85 \mathbf{v}_i$ each step (holding initial temperature $T_i$ anchored) until $v_{rms} < 0.01 v_{esc}$ (default 80 steps).
  - For `LivingTerra`, the outermost crust + ocean shell selects $6 \times G^2$ tracers ($G = \mathrm{tracerGridRes}$, default `16` = 1536 tracers) by assigning each cube-sphere grid bin `(face, gx, gy)` its closest unique surface particle, setting `tracerIndex[p]`, `tracerRestRadiusPsu[p]`, `tracerRestNormal[p]`, and `tracerIsOcean[p]`.

### 5.5 Frontend Cube-Sphere Quadtree LOD & Impact Biosphere

- **LOD-to-SPH Transition Options & Justification:**
  1. *Global Altitude Switch:* Swaps the entire planet mesh for SPH points below a threshold; fails when half the planet is a crater/plume while the antipode remains an intact biosphere.
  2. *Real-Time Volumetric Marching Cubes over SPH:* Cannot resolve 1 m terrain/flora and breaks deterministic surface biomes.
  3. **Anchored Cube-Sphere Quadtree (Depth $0 \dots 10$) with Local Fracture Dissolve ($D \in [0.55, 0.85]$) [CHOSEN]:**
     - Subdivide quadtree nodes by projected screen-space solid angle up to depth $10$ ($\sim 8\ \mathrm{km}$ patch with $33\times 33$ vertices $\approx 250\ \mathrm{m}$ mesh + analytical fBm detail down to $1\ \mathrm{m}$).
     - **Tracer-to-LOD Coordinate Frame Alignment:** `CrustTracerRecord` streams `(xKm, yKm, zKm)` in the authoritative system COM frame (already encoding planet rotation and tidal/impact deformation). When `tracerCount > 0`, `CubeSphereLod` computes each mesh vertex `(face, u, v)` position directly in system COM coordinates by bilinearly interpolating the surrounding `CrustTracerRecord` `(xKm, yKm, zKm)` positions plus sub-grid radial fBm relief + `tsunamiHeightM` ($\times 10^{-3}\ \mathrm{km}$) along the interpolated surface normal, falling back to `livingComKm + R * rotate(unitNormal, spinAngleRad)` only when `tracerCount == 0`; simultaneously, it bilinearly samples `dispKm`, `tempK`, `damage`, `overpressureKpa`, and `tsunamiHeightM` from `TracerBufferPool`. `OrbitalSurfaceCamera` operates in the same authoritative system COM frame so `SurfaceLock` matches `CubeSphereLod` without double-rotating tracer positions by `spinAngleRad`.
     - **Dissolve Threshold:**
       - $D < 0.55$: solid biome terrain + animated ocean surface (`waterHeight = 0 + tsunamiHeightM`). `SphParticleRenderer` suppresses intact `LivingTerra` solid crust spheres behind the surface shell (`alpha = 0` when `bodyId == 0 && damage < 0.55 && phase == Solid/LiquidWater` if `LivingTerra` LOD is active), while shocked/molten/vapor/impactor particles always render.
       - $D \in [0.55, 0.85]$: terrain vertex shader renders glowing incandescent Voronoi crust fissures, uplifts fragmented tiles, chars biome albedo, and cross-fades mesh opacity $1 \to 0$ as underlying SPH particles fade $0 \to 1$.
       - $D \ge 0.85$: quadtree leaf culls completely (`culledDamagedPatchCount++`), revealing the raw SPH debris cloud.

- **Fauna State Machine (`ProceduralFauna`):**
  ```text
  [Grazing] --(impactorDistRadii < 4.0 OR pgaG > 0.01)--> [Alert]
  [Alert]   --(pgaG > 0.05 OR overpressureKpa > 2.0 OR radFluxKwM2 > 5.0)--> [StampedeOrMassFlight]
  [Grazing | Alert | StampedeOrMassFlight] --(radFluxKwM2 > 12.0 OR localTempK > 340.0)--> [ThermalEscapeOrBurrow]
  [Any] --(localTempK > 420.0 OR overpressureKpa > 80.0 OR pgaG > 2.5 OR ejectaFluxKgM2 > 50.0 OR (isLand && tsunamiHeightM > 20.0) OR crustDamage > 0.7)--> [Lethal]
  ```

- **Impact Biosphere Delay Equations (`ImpactBiosphereCoupler`):**
  For any surface normal $\hat{\mathbf{n}}_{site}$, arc angle $\theta = \arccos(\mathrm{clamp}(\hat{\mathbf{n}}_{site} \cdot \hat{\mathbf{n}}_{imp}, -1, 1))$, $d_{surf} = R_p \theta$, $d_{chord} = 2 R_p \sin(\theta/2)$:
  - **Tidal & Tsunami:** Pre-impact Roche tidal amplitude $h_{tide} = 0.6 \frac{M_{imp}}{M_{target}} \frac{R_p^4}{r_{sep}^3} \frac{3\cos^2\psi - 1}{2}$. Post-impact tsunami wave arrives at $t_{tsu} = t_{imp} + d_{surf} / 200.0\ \mathrm{m/s}$, amplitude $H_0 / \sqrt{\max(0.05, \sin\theta)}$ attenuated by land elevation.
  - **Seismic P/S Waves:** $t_P = t_{imp} + d_{chord} / 6500.0\ \mathrm{m/s}$, $t_S = t_{imp} + d_{surf} / 3700.0\ \mathrm{m/s}$, PGA in $g$ combining wave packet propagation with local tracer acceleration/damage.
  - **Thermal Radiation Flash:** Horizon check $\hat{\mathbf{n}}_{site} \cdot (\mathbf{r}_{plume} - \mathbf{r}_{site}) > 0$; if visible, $q_{rad} = \epsilon \sigma_{SB} T_{plume}^4 \frac{R_{plume}^2}{\|\mathbf{r}_{plume} - \mathbf{r}_{site}\|^2} e^{-\tau}$. Ignites local wildfire if $q_{rad} > 35\ \mathrm{kW/m^2}$ or tracer $T > 573\ \mathrm{K}$.
  - **Ballistic Ejecta:** Minimum-energy suborbital flight time $t_{ej}(\theta)$ ($0\ \mathrm{s} \to 2400\ \mathrm{s}$) + re-entry heat flux and debris loading $\Sigma_{ej}(\theta)$.
  - **Atmospheric Blast:** $t_{blast} = t_{imp} + d_{surf} / 340.0\ \mathrm{m/s}$, merging Sach blast-wave $\Delta P$ with tracer `overpressureKpa`.
  - **Extinction Tracker:** Evaluates $256$ Fibonacci global tiles across all 7 biomes to track `totalBiomassGtC`, Shannon diversity $H' = -\sum p_s \ln p_s$, survival % per biome and per guild, and `ExtinctionTier` (`Nominal`, `RegionalCrisis` $<85\%$, `MassExtinction` $<25\%$, `Sterilization` $<0.5\%$).

---

## 6. Interaction and Trust Boundaries

- **Authoritative Server Simulation:** C++20/CUDA `Simulation` holds all authoritative physics state (`pos`, `vel`, `rho`, `u`, `T`, `P`, `D`, `TelemetrySnapshot`, `CollisionOutcome`).
- **Client Role:** TypeScript client renders SPH particles, Cube-Sphere LOD terrain, and client-side procedural flora/fauna driven deterministically by the streamed `TracerBufferPool` and `TelemetrySnapshot`. Client sends strictly validated JSON RPC commands; unknown commands or malformed fields return `{"type":"error","code":"BAD_REQUEST",...}` without mutating simulation state.
- **WebSocket Lifecycle & Backpressure:**
  - `SimClient` maintains exponential backoff reconnect (`Math.min(1000 * 2 ** attempt, 30000)`), sends JSON heartbeat `ping` every `20s`, and queues outbound commands while disconnected.
  - `HttpWsServer` runs physics on a dedicated worker thread and stages encoded binary frames in a double buffer. Slow WebSocket clients drop intermediate stale frames (never blocking physics step loop).
  - HTTP static file handler resolves `weakly_canonical(webRoot / relPath)` and rejects any path escaping `webRoot` with `403 Forbidden`.

---

## 7. Performance and Budgets

- **VRAM Budget (RTX 4050, 6141 MiB limit):**
  - Max capacity `250,000` particles: double-buffered SoA (`~68 MiB`) + Karras BVH arrays (`~36 MiB`) + CUB radix/reduce temp (`~20 MiB`) + packed frame staging (`~6 MiB`) = **`< 150 MiB` total** (`< 2.5%` of available VRAM).
- **GPU Physics Step Budget:**
  - At $N = 100{,}000$ particles, full KDK step (Morton sort + BVH + 2 SPH passes + Barnes-Hut $\theta=0.5$ + EOS + reduction) targets **`<= 33.3 ms`** ($\ge 30\ \mathrm{steps/s}$).
- **Network Bandwidth Budget:**
  - At $N = 100{,}000$ and $T = 1{,}536$: frame size $= 40 + 2{,}000{,}000 + 61{,}440 = 2{,}061{,}480\ \mathrm{bytes}$ ($\approx 1.97\ \mathrm{MiB}$). At $20\text{--}30\ \mathrm{Hz}$ local loopback, throughput is $40\text{--}60\ \mathrm{MiB/s}$ with optional `zlib` flag for remote connections.
- **Browser Frame & GC Budget (60 FPS Target):**
  - Zero per-frame heap allocations in `BinaryProtocol.decodeFrame` (reuses pre-allocated `ParticleBufferPool` & `TracerBufferPool`).
  - Active `InstancedMesh` caps: `<= 4,096` flora instances and `<= 512` fauna agents in local LOD radius; `CubeSphereLod` active leaf patches `<= 192`.

---

## 8. Non-Functional Requirements

- **Determinism:** Given identical `CollisionScenarioConfig.randomSeed` and `particles`, `PlanetBuilder` and `sim_headless` produce bit-reproducible initial states and deterministic single-GPU trajectories within floating-point atomic reduction tolerance ($< 10^{-5}$).
- **Headless Isolation:** `engine/` compiles and executes without X11, Wayland, OpenGL, or Web browser dependencies.
- **Strict Type Safety:** `web/tsconfig.json` enforces `"strict": true` and `"noImplicitAny": true`.
- **CLI Flag Parity (`sim_headless`):** Supports `--preset`, `--particles`, `--steps`, `--dt-max`, `--relax-steps`, `--dump-telemetry`, `--output-csv`, `--verify-conservation`, `--list-presets`, `--help`.
- **CLI Flag Parity (`sim_server`):** Supports `--port`, `--bind`, `--web-root`, `--particles`, `--preset`, `--target-fps`, `--help`.

---

## 9. Acceptance Criteria

- **AC-1 (Tillotson & $H_2/He$ EOS Exactness):** Calling `eos::evaluateEosPsu` across all 5 `MaterialId` values in compressed ($\eta=1.5$), two-phase transition ($E_{iv} \le u < E_{cv}$), hot expanded ($\eta=0.2, u=2E_{cv}$), and damaged tensile ($\eta=0.9, D=1.0$) states returns finite positive sound speed $c_s > 0$, non-negative temperature $T \ge 3\ \mathrm{K}$, $P(\text{damaged tension}) = 0$, and exact ideal+polytropic pressure $P = K_{H_2}\rho^2 + (\gamma-1)\rho u_{th}$ for `HydrogenHelium`.
- **AC-2 (Hydrostatic Equilibrium & Relaxation Stability):** Building all 6 `PlanetArchetype` values via `PlanetBuilder::buildPlanet` and calling `Simulation::relaxBody(0, 80, vrmsOverVesc)` converges to $v_{rms} / v_{esc} < 0.01$ with monotonic radial density decreasing outward within each layer.
- **AC-3 (Barnes-Hut Gravity Accuracy):** On a two-body system of $N = 4{,}096$ particles, `launchGravityBarnesHut(..., theta=0.5f)` matches an exact $N^2$ double-precision reference acceleration within $< 1.5\%$ median relative error and $< 5.0\%$ 99th-percentile error.
- **AC-4 (SPH Wendland $C^2$ & Multi-Material CMB Stability):** Integrating a hydrostatic two-layer `Iron` core ($32\%$) + `Granite` mantle ($68\%$) sphere for 200 steps preserves the sharp density discontinuity across the core-mantle boundary without interface void formation and maintains radial velocity jitter $v_{rms} / v_{esc} < 0.015$.
- **AC-5 (Conservation Audit on Giant Impact):** Executing `sim_headless --preset EarthTheiaMoonForming --particles 20000 --steps 150 --verify-conservation` exits with code `0`, proving $|(E_{tot}(t) + E_{rad}(t) - E_0) / E_0| < 0.01$, $|\Delta \mathbf{P}| / (\sum m_i v_{esc}) < 10^{-4}$, and $|\Delta \mathbf{L}| / |\mathbf{L}_0| < 0.005$.
- **AC-6 (Shock Heating, Plasticity & Brittle Spallation):** Running `MercuryMantleStrip` ($v_{imp}/v_{esc}=2.8$) for 100 steps produces peak temperatures exceeding `Granite` $T_{melt,0} = 1600\ \mathrm{K}$ (`PhaseId::Molten`/`Vapor`) and damage $D > 0.8$ near the impact crater while intact far-field particles remain $D < 0.2$.
- **AC-7 (Radiative Cooling & Surface Exposure):** Isolated hot particles ($T=4000\ \mathrm{K}, f_{surf} \approx 1$) decrease in internal energy according to $\sigma_{SB} T^4$, whereas deep interior particles ($f_{surf} = 0$) suffer zero direct radiative energy loss.
- **AC-8 (RTX 4050 Real-Time Throughput):** Running `Simulation::step()` on `LivingGaiaAsteroidCataclysm` with `totalParticles = 100000` achieves median `gpuStepMs <= 33.3` ms on RTX 4050 and peak VRAM allocation $< 512\ \mathrm{MiB}$.
- **AC-9 (Surface Tracer Continuity Across Morton Re-Sorts):** Stepping `LivingGaiaAsteroidCataclysm` for 100 steps maintains $100\%$ unique `anchorIndex` coverage ($0..T-1$, $T = 1536$) with no NaN positions, displacements, or out-of-range temperatures.
- **AC-10 (Headless CLI Export):** Executing `sim_headless --preset LivingGaiaAsteroidCataclysm --particles 10000 --steps 40 --output-csv /tmp/test.csv --dump-telemetry /tmp/test.json --verify-conservation` exits `0` and generates valid non-empty CSV and JSON files containing all `TelemetrySnapshot` fields.
- **AC-11 (RFC 6455 Handshake & HTTP Path Traversal Guard):** `sha1Base64AcceptKey("dGhlIHNhbXBsZSBub25jZQ==")` returns `"s3pPLMBiTxaQ9kYGzzhZRbK+xOo="`; HTTP `GET /../etc/passwd` returns status `403`.
- **AC-12 (Binary Protocol Pack/Unpack Roundtrip):** Packing $N = 100{,}000$ particles and $T = 1{,}536$ tracers via `Simulation::extractRenderFrame` and decoding via `BinaryProtocol.decodeBinaryFrame` verifies magic `0x50434C53`, exact byte size `40 + 20*N + 40*T`, $< 0.1\%$ relative log quantization error on $T$ and $P$, and exact `materialId`/`phaseId`/`bodyId`/`damage` match.
- **AC-13 (WebSocket JSON Control RPC):** Sending `set_preset`, `launch_custom`, `play`, `pause`, `step`, `set_timescale`, `set_slice_plane`, and `request_export` over WebSocket produces valid `preset_loaded`, `telemetry`, and `export_data` responses matching requested state.
- **AC-14 (Orbital-to-Surface Camera Precision):** `OrbitalSurfaceCamera` interpolates smoothly from `altitudeKm = 10000` ($10^7\ \mathrm{m}$) to `altitudeKm = 0.001` ($1\ \mathrm{m}$) in `SurfaceLock` mode while `toCameraRelativePos` resolves local surface coordinates within $< 10^{-3}\ \mathrm{m}$ precision without `float32` jitter.
- **AC-15 (SPH Shader Modes & Cross-Section Slice):** `Vitest` verifies `SphParticleRenderer` GLSL shader source generation, all 6 `VisMode` uniform switches, slice plane math (`setSlicePlane(true, ...)`), and `ParticleBufferPool` $\to$ attribute packing in Node, while actual WebGL2 shader compilation and draw pass across `VisMode` and slice states are verified in `AC-22` (Headless Chromium 156).
- **AC-16 (Cube-Sphere Quadtree LOD & Fracture Dissolve):** `CubeSphereLod.update(...)` subdivides down to depth `10` near camera, applies `CrustTracerRecord.dispKm`, enters crack-dissolve state for $D \in [0.55, 0.85]$, and culls patches (`getCulledDamagedPatchCount() > 0`) when $D \ge 0.85$ while intact patches remain active.
- **AC-17 (Deterministic 7-Biome Climate Classification):** `BiomeSystem` deterministically produces all 7 `BiomeId` categories across planet samples (`Vitest` asserts determinism for fixed seed and non-zero coverage of all 7 biomes).
- **AC-18 (Procedural L-System Flora >= 8 Species):** `buildFloraSpeciesGeometry` produces valid `BufferGeometry` with `position`, `normal`, `aBranchWeight`, `aPartType` attributes for all 9 `FloraSpeciesId` species (`getSpeciesCount() >= 8`), and `ProceduralFlora.updateImpactResponse` updates blast/thermal shader uniforms.
- **AC-19 (Procedural Fauna Guilds & Impact State Machine):** `transitionFaunaState` transitions deterministically across `Grazing -> Alert -> StampedeOrMassFlight -> ThermalEscapeOrBurrow -> Lethal` as seismic PGA, thermal radiation, blast overpressure, and damage cross defined physical thresholds for all 4 `FaunaGuild` types.
- **AC-20 (Physical Impact Delay Solver & Horizon Occultation):** `ImpactBiosphereCoupler.evaluateSiteExposure` enforces $t_P < t_S < t_{blast}$ arrival ordering, tsunami propagation at $\approx 200\ \mathrm{m/s}$, and returns `radFluxKwM2 === 0` at antipodal coordinates ($\hat{\mathbf{n}}_{site} \cdot (\mathbf{r}_{plume} - \mathbf{r}_{site}) \le 0$).
- **AC-21 (Mass Extinction Tracker & Outcome Classification):** For `LivingGaiaAsteroidCataclysm`, `ImpactBiosphereCoupler.getGlobalMetrics()` reports decreasing `totalBiomassGtC` and `overallSurvivalPct` after global wave/winter onset, valid Shannon $H' \ge 0$, and `Simulation::computeTelemetry()` classifies distinct `Merging`, `HitAndRun`, and `CatastrophicDisruption` outcomes across corresponding presets.
- **AC-22 (End-to-End Headless Chromium Smoke Test):** Starting `sim_server --particles 100000` and navigating `chromium 156` to `http://127.0.0.1:<port>` renders live streamed SPH particles (verifying WebGL2 shader compilation & draw pass across all 6 `VisMode` switches), passes interactive preset switch, slice plane toggle, zoom down to `LivingTerra` surface biome with flora/fauna, and reports `0` uncaught JS/WebGL errors.

---

## 10. Assumptions and Design Decisions

1. **Planetary Simulation Units (PSU) internally, SI at boundaries:** Using $L_u = 10^6\ \mathrm{m}$, $M_u = 10^{24}\ \mathrm{kg}$, $T_u = 10^3\ \mathrm{s}$ eliminates `float32` overflow on $m_i^2$ ($> 3.4 \times 10^{38}$ in raw SI) while keeping RTX 4050 FP32 throughput intact; all telemetry and energy totals accumulate in `double` SI.
2. **Optional Velocity Speed Block (`flags & 0x08`):** Preserves the exact 20-byte `ParticleRenderRecord` layout while supporting accurate GPU `Velocity` visualization mode when requested (`N` extra `uint8_t` bytes appended after `CrustTracerRecord` block).
3. **Particle-Constant Gravitational Softening ($\epsilon_i = 0.8 h_{0,i}$):** Avoids non-conservative $\nabla h$ correction terms while maintaining smooth multi-resolution core/mantle softening.
4. **Tillotson Granite + Iron + Allen Water + Polytropic $H_2/He$ + Regolith:** Uses exact published Melosh (1989) / Benz & Asphaug (1999) / Allen (1967) constants with analytical $u_{cold}(\rho)$ subtraction so adiabatic core compression does not spuriously inflate temperature.
5. **Equal-Mass Particles Across Colliding Bodies (where $M_{imp} \ge 0.02 M_{target}$):** Ensures numerical shock-front stability at impact interfaces without high-mass-ratio SPH penetration artifacts; `LivingGaiaAsteroidCataclysm` sets impactor mass to $0.025 M_\oplus$ ($\approx 2{,}439$ particles at $N = 100{,}000$), resolving a hemispherical cataclysm while preserving an intact antipodal biosphere for close-up inspection.
6. **Frontend Stack (`Vite + TypeScript + Three.js + Vitest`):** Selected per approved plan `A3` for `logarithmicDepthBuffer`, `InstancedMesh`, and typed WebGL2 shaders.
