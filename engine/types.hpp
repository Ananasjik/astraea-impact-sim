#pragma once

#include <cstdint>
#include <string>

#include "units.hpp"

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
    PlanetArchetype archetype     = PlanetArchetype::LivingTerra;
    double          massKg        = units::M_EARTH_SI;
    double          surfaceTempK  = 288.0;
    double          coreTempK     = 5500.0;
    double          spinOmegaRadS = 7.292115e-5;
    uint32_t        relaxSteps    = 80;
};

struct CollisionScenarioConfig {
    std::string  presetName              = "LivingGaiaAsteroidCataclysm";
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
