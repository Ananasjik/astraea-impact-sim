#include "simulation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "eos.hpp"
#include "planet_builder.hpp"
#include "presets.hpp"
#include "sph_gravity.cuh"
#include "units.hpp"

namespace astraea {

namespace {

constexpr double kPi = 3.14159265358979323846;

const char* outcomeToString(CollisionOutcome outcome) {
    switch (outcome) {
        case CollisionOutcome::Approaching:            return "Approaching";
        case CollisionOutcome::Merging:                return "Merging";
        case CollisionOutcome::GrazeAndMerge:          return "GrazeAndMerge";
        case CollisionOutcome::HitAndRun:              return "HitAndRun";
        case CollisionOutcome::CatastrophicDisruption: return "CatastrophicDisruption";
    }
    return "Approaching";
}

double finiteOrZeroD(double v) {
    return std::isfinite(v) ? v : 0.0;
}

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

void writeSnapshotJsonFields(std::ostringstream& os, const TelemetrySnapshot& s) {
    os << std::setprecision(15);
    os << "\"frameSeq\":" << s.frameSeq << ","
       << "\"simTimeSec\":" << finiteOrZeroD(s.simTimeSec) << ","
       << "\"dtSec\":" << finiteOrZeroD(static_cast<double>(s.dtSec)) << ","
       << "\"outcome\":\"" << outcomeToString(s.outcome) << "\","
       << "\"kineticEnergyJ\":" << finiteOrZeroD(s.kineticEnergyJ) << ","
       << "\"potentialEnergyJ\":" << finiteOrZeroD(s.potentialEnergyJ) << ","
       << "\"internalEnergyJ\":" << finiteOrZeroD(s.internalEnergyJ) << ","
       << "\"radiatedEnergyJ\":" << finiteOrZeroD(s.radiatedEnergyJ) << ","
       << "\"totalEnergyJ\":" << finiteOrZeroD(s.totalEnergyJ) << ","
       << "\"initialTotalEnergyJ\":" << finiteOrZeroD(s.initialTotalEnergyJ) << ","
       << "\"energyDriftRel\":" << finiteOrZeroD(s.energyDriftRel) << ","
       << "\"momentumKgMS\":[" << finiteOrZeroD(s.momentumKgMS[0]) << ","
       << finiteOrZeroD(s.momentumKgMS[1]) << ","
       << finiteOrZeroD(s.momentumKgMS[2]) << "],"
       << "\"angularMomentumKgM2S\":[" << finiteOrZeroD(s.angularMomentumKgM2S[0]) << ","
       << finiteOrZeroD(s.angularMomentumKgM2S[1]) << ","
       << finiteOrZeroD(s.angularMomentumKgM2S[2]) << "],"
       << "\"virialRatio\":" << finiteOrZeroD(s.virialRatio) << ","
       << "\"maxTempK\":" << finiteOrZeroD(s.maxTempK) << ","
       << "\"maxPressureGPa\":" << finiteOrZeroD(s.maxPressureGPa) << ","
       << "\"boundMassFrac\":" << finiteOrZeroD(s.boundMassFrac) << ","
       << "\"diskMassFrac\":" << finiteOrZeroD(s.diskMassFrac) << ","
       << "\"ejectaMassFrac\":" << finiteOrZeroD(s.ejectaMassFrac) << ","
       << "\"qRatioQrd\":" << finiteOrZeroD(s.qRatioQrd) << ","
       << "\"gpuStepMs\":" << finiteOrZeroD(s.gpuStepMs) << ","
       << "\"livingBodyComKm\":[" << finiteOrZeroD(s.livingBodyComKm[0]) << ","
       << finiteOrZeroD(s.livingBodyComKm[1]) << ","
       << finiteOrZeroD(s.livingBodyComKm[2]) << "],"
       << "\"livingBodyRadiusKm\":" << finiteOrZeroD(s.livingBodyRadiusKm) << ","
       << "\"livingBodySpinRadS\":" << finiteOrZeroD(s.livingBodySpinRadS) << ","
       << "\"impactorComKm\":[" << finiteOrZeroD(s.impactorComKm[0]) << ","
       << finiteOrZeroD(s.impactorComKm[1]) << ","
       << finiteOrZeroD(s.impactorComKm[2]) << "],"
       << "\"particleCount\":" << s.particleCount << ","
       << "\"tracerCount\":" << s.tracerCount << ","
       << "\"tracerGridRes\":" << s.tracerGridRes << ","
       << "\"livingBodyPresent\":" << (s.livingBodyPresent ? "true" : "false");
}

void writeSnapshotCsvRow(std::ostringstream& os, const TelemetrySnapshot& s) {
    os << std::setprecision(15)
       << s.frameSeq << ','
       << finiteOrZeroD(s.simTimeSec) << ','
       << finiteOrZeroD(static_cast<double>(s.dtSec)) << ','
       << outcomeToString(s.outcome) << ','
       << finiteOrZeroD(s.kineticEnergyJ) << ','
       << finiteOrZeroD(s.potentialEnergyJ) << ','
       << finiteOrZeroD(s.internalEnergyJ) << ','
       << finiteOrZeroD(s.radiatedEnergyJ) << ','
       << finiteOrZeroD(s.totalEnergyJ) << ','
       << finiteOrZeroD(s.initialTotalEnergyJ) << ','
       << finiteOrZeroD(s.energyDriftRel) << ','
       << finiteOrZeroD(s.momentumKgMS[0]) << ','
       << finiteOrZeroD(s.momentumKgMS[1]) << ','
       << finiteOrZeroD(s.momentumKgMS[2]) << ','
       << finiteOrZeroD(s.angularMomentumKgM2S[0]) << ','
       << finiteOrZeroD(s.angularMomentumKgM2S[1]) << ','
       << finiteOrZeroD(s.angularMomentumKgM2S[2]) << ','
       << finiteOrZeroD(s.virialRatio) << ','
       << finiteOrZeroD(s.maxTempK) << ','
       << finiteOrZeroD(s.maxPressureGPa) << ','
       << finiteOrZeroD(s.boundMassFrac) << ','
       << finiteOrZeroD(s.diskMassFrac) << ','
       << finiteOrZeroD(s.ejectaMassFrac) << ','
       << finiteOrZeroD(s.qRatioQrd) << ','
       << finiteOrZeroD(s.gpuStepMs) << ','
       << finiteOrZeroD(s.livingBodyComKm[0]) << ','
       << finiteOrZeroD(s.livingBodyComKm[1]) << ','
       << finiteOrZeroD(s.livingBodyComKm[2]) << ','
       << finiteOrZeroD(s.livingBodyRadiusKm) << ','
       << finiteOrZeroD(s.livingBodySpinRadS) << ','
       << finiteOrZeroD(s.impactorComKm[0]) << ','
       << finiteOrZeroD(s.impactorComKm[1]) << ','
       << finiteOrZeroD(s.impactorComKm[2]) << ','
       << s.particleCount << ','
       << s.tracerCount << ','
       << s.tracerGridRes << ','
       << (s.livingBodyPresent ? 1 : 0) << '\n';
}

struct CombinedHostState {
    uint32_t count = 0;
    std::vector<float>    posX, posY, posZ;
    std::vector<float>    velX, velY, velZ;
    std::vector<float>    mass, h, h0, rho, u, tempK, epsFail;
    std::vector<float>    sXX, sYY, sZZ, sXY, sXZ, sYZ;
    std::vector<float>    damage, damageCbrt;
    std::vector<uint8_t>  materialId, bodyId;
    std::vector<uint32_t> originalId;
    std::vector<int32_t>  tracerIndex;
    std::vector<float>    tracerRestRadiusPsu;
    std::vector<float>    tracerRestNormalX, tracerRestNormalY, tracerRestNormalZ;
    std::vector<uint8_t>  tracerIsOcean;
};

template <typename T>
void allocPinned(T*& ptr, size_t count) {
    if (count == 0) {
        ptr = nullptr;
        return;
    }
    ASTRAEA_CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
}

template <typename T>
void freePinned(T*& ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
        ptr = nullptr;
    }
}

struct HostPinnedStaging {
    uint32_t  capacity = 0;
    float*    posX = nullptr;      float* posY = nullptr;      float* posZ = nullptr;
    float*    velX = nullptr;      float* velY = nullptr;      float* velZ = nullptr;
    float*    mass = nullptr;
    float*    h = nullptr;
    float*    rho = nullptr;
    float*    u = nullptr;
    float*    phiGrav = nullptr;
    float*    tempK = nullptr;
    float*    pressure = nullptr;
    float*    radPowerW = nullptr;
    float*    damage = nullptr;
    uint8_t*  materialId = nullptr;
    uint8_t*  bodyId = nullptr;
    uint32_t* originalId = nullptr;

    void allocate(uint32_t cap) {
        free();
        if (cap == 0) {
            return;
        }
        allocPinned(posX, cap);      allocPinned(posY, cap);      allocPinned(posZ, cap);
        allocPinned(velX, cap);      allocPinned(velY, cap);      allocPinned(velZ, cap);
        allocPinned(mass, cap);
        allocPinned(h, cap);
        allocPinned(rho, cap);
        allocPinned(u, cap);
        allocPinned(phiGrav, cap);
        allocPinned(tempK, cap);
        allocPinned(pressure, cap);
        allocPinned(radPowerW, cap);
        allocPinned(damage, cap);
        allocPinned(materialId, cap);
        allocPinned(bodyId, cap);
        allocPinned(originalId, cap);
        capacity = cap;
    }

    void free() {
        freePinned(posX);      freePinned(posY);      freePinned(posZ);
        freePinned(velX);      freePinned(velY);      freePinned(velZ);
        freePinned(mass);
        freePinned(h);
        freePinned(rho);
        freePinned(u);
        freePinned(phiGrav);
        freePinned(tempK);
        freePinned(pressure);
        freePinned(radPowerW);
        freePinned(damage);
        freePinned(materialId);
        freePinned(bodyId);
        freePinned(originalId);
        capacity = 0;
    }

    ~HostPinnedStaging() {
        free();
    }
};

double computeQRatioQrd(
    double targetMassKg,
    double impactorMassKg,
    double targetRadiusM,
    double impactorRadiusM,
    double vContactMs,
    double impactParamB) {
    if (impactorMassKg <= 0.0 || targetMassKg <= 0.0) {
        return 0.0;
    }
    const double totalMassKg = targetMassKg + impactorMassKg;
    const double muKg = (targetMassKg * impactorMassKg) / totalMassKg;
    const double qImpact = 0.5 * (muKg / totalMassKg) * vContactMs * vContactMs;

    // Leinhardt & Stewart (2012) gravity-regime catastrophic disruption threshold Q*_RD
    constexpr double kRho1KgM3 = 1000.0;
    constexpr double kCStar = 1.9;
    constexpr double kMuBar = 0.36;

    const double rc1M = std::cbrt((3.0 * totalMassKg) / (4.0 * kPi * kRho1KgM3));
    const double qRdGamma1 = kCStar * 0.8 * kPi * kRho1KgM3 * units::G_SI * rc1M * rc1M;

    const double mSmall = std::min(targetMassKg, impactorMassKg);
    const double mLarge = std::max(targetMassKg, impactorMassKg);
    const double gamma = std::max(mSmall / mLarge, 1.0e-9);

    const double massFactorExp = (2.0 / (3.0 * kMuBar)) - 1.0; // ~0.85185
    const double gammaTerm = ((gamma + 1.0) * (gamma + 1.0)) / (4.0 * gamma);
    const double qRdHeadOn = qRdGamma1 * std::pow(gammaTerm, massFactorExp);

    // Oblique interacting mass correction
    const double b = std::clamp(std::abs(impactParamB), 0.0, 1.0);
    const double rT = std::max(targetRadiusM, 1.0);
    const double rI = std::max(impactorRadiusM, 1.0);
    const double l = (rT + rI) * (1.0 - b);
    double alpha = 1.0;
    if (l <= 0.0) {
        alpha = 0.02;
    } else if (l < 2.0 * rI) {
        alpha = std::clamp((l * l * (3.0 * rI - l)) / (4.0 * rI * rI * rI), 0.02, 1.0);
    }

    const double muAlphaKg = (alpha * impactorMassKg * targetMassKg) / (targetMassKg + alpha * impactorMassKg);
    const double oblExp = 2.0 - 1.5 * kMuBar; // 1.46
    const double qRdOblique = qRdHeadOn * std::pow(std::max(muKg / std::max(muAlphaKg, 1.0e-12 * totalMassKg), 1.0), oblExp);

    return (qRdOblique > 0.0) ? (qImpact / qRdOblique) : 0.0;
}

void centerAndZeroBulkVelocity(BuiltPlanetHost& body) {
    const size_t n = body.posX.size();
    if (n == 0) {
        return;
    }
    double mTot = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double m = static_cast<double>(body.mass[i]);
        mTot += m;
        cx += m * static_cast<double>(body.posX[i]);
        cy += m * static_cast<double>(body.posY[i]);
        cz += m * static_cast<double>(body.posZ[i]);
        vx += m * static_cast<double>(body.velX[i]);
        vy += m * static_cast<double>(body.velY[i]);
        vz += m * static_cast<double>(body.velZ[i]);
    }
    if (mTot <= 0.0) {
        return;
    }
    cx /= mTot; cy /= mTot; cz /= mTot;
    vx /= mTot; vy /= mTot; vz /= mTot;
    for (size_t i = 0; i < n; ++i) {
        body.posX[i] = static_cast<float>(static_cast<double>(body.posX[i]) - cx);
        body.posY[i] = static_cast<float>(static_cast<double>(body.posY[i]) - cy);
        body.posZ[i] = static_cast<float>(static_cast<double>(body.posZ[i]) - cz);
        body.velX[i] = static_cast<float>(static_cast<double>(body.velX[i]) - vx);
        body.velY[i] = static_cast<float>(static_cast<double>(body.velY[i]) - vy);
        body.velZ[i] = static_cast<float>(static_cast<double>(body.velZ[i]) - vz);
    }
}

void applyRigidSpinAroundY(BuiltPlanetHost& body, double spinOmegaRadS) {
    const double omegaPsu = spinOmegaRadS * units::T_UNIT_S;
    const size_t n = body.posX.size();
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(body.posX[i]);
        const double z = static_cast<double>(body.posZ[i]);
        body.velX[i] = static_cast<float>(omegaPsu * z);
        body.velY[i] = 0.0f;
        body.velZ[i] = static_cast<float>(-omegaPsu * x);
    }
}

} // namespace

struct Simulation::Impl {
    uint32_t maxParticles = 0;
    uint32_t allocatedCapacity = 0;
    uint32_t tracerCapacity = 0;

    ParticleBuffersDevice p{};
    ParticleBuffersDevice alt{};
    cuda::RadixBvhDevice bvh{};
    HostPinnedStaging host{};

    uint8_t* d_renderParticles = nullptr;
    uint8_t* d_renderSpeed = nullptr;
    uint8_t* d_renderTracers = nullptr;

    cudaEvent_t evStart = nullptr;
    cudaEvent_t evStop = nullptr;
    bool eventsCreated = false;

    CollisionScenarioConfig config{};
    SlicePlaneConfig slicePlane{};
    double timeScale = 1.0;

    uint64_t frameSeq = 0;
    double simTimeSec = 0.0;
    float lastDtSec = 0.0f;
    float prevDtPsu = 0.0f;
    double radiatedEnergyJ = 0.0;
    double prevTotalRadPowerW = 0.0;
    double initialTotalEnergyJ = 0.0;
    double lastGpuStepMs = 0.0;

    double targetRadiusPsu = 1.0;
    double impactorRadiusPsu = 0.0;
    double qRatioQrd = 0.0;
    double initialPeakTempK = 0.0;
    bool hasImpacted = false;

    uint32_t tracerCount = 0;
    uint32_t tracerGridRes = 0;
    bool livingBodyPresent = false;
    double sysComPsu[3] = {0.0, 0.0, 0.0};
    double livingComPsu[3] = {0.0, 0.0, 0.0};

    TelemetrySnapshot lastSnapshot{};
    std::vector<TelemetrySnapshot> history;

    explicit Impl(uint32_t maxP) : maxParticles(std::max(maxP, 1u)) {
        ASTRAEA_CUDA_CHECK(cudaEventCreate(&evStart));
        ASTRAEA_CUDA_CHECK(cudaEventCreate(&evStop));
        eventsCreated = true;
        ensureCapacity(maxParticles);
    }

    ~Impl() {
        freeGpuResources();
        if (eventsCreated) {
            cudaEventDestroy(evStart);
            cudaEventDestroy(evStop);
            eventsCreated = false;
        }
    }

    void freeGpuResources() {
        cuda::freeParticleBuffers(p);
        cuda::freeParticleBuffers(alt);
        cuda::freeRadixBvh(bvh);
        host.free();
        if (d_renderParticles != nullptr) {
            cudaFree(d_renderParticles);
            d_renderParticles = nullptr;
        }
        if (d_renderSpeed != nullptr) {
            cudaFree(d_renderSpeed);
            d_renderSpeed = nullptr;
        }
        if (d_renderTracers != nullptr) {
            cudaFree(d_renderTracers);
            d_renderTracers = nullptr;
        }
        allocatedCapacity = 0;
        tracerCapacity = 0;
    }

    void ensureCapacity(uint32_t needed) {
        if (needed <= allocatedCapacity) {
            return;
        }
        const uint32_t cap = std::max(needed, maxParticles);
        cuda::freeParticleBuffers(p);
        cuda::freeParticleBuffers(alt);
        cuda::freeRadixBvh(bvh);
        if (d_renderParticles != nullptr) {
            cudaFree(d_renderParticles);
            d_renderParticles = nullptr;
        }
        if (d_renderSpeed != nullptr) {
            cudaFree(d_renderSpeed);
            d_renderSpeed = nullptr;
        }

        cuda::allocateParticleBuffers(p, cap);
        cuda::allocateParticleBuffers(alt, cap);
        cuda::allocateRadixBvh(bvh, cap);
        host.allocate(cap);
        ASTRAEA_CUDA_CHECK(cudaMalloc(&d_renderParticles, static_cast<size_t>(cap) * 20u));
        ASTRAEA_CUDA_CHECK(cudaMalloc(&d_renderSpeed, static_cast<size_t>(cap)));

        allocatedCapacity = cap;
        maxParticles = cap;
    }

    void ensureTracerCapacity(uint32_t neededTracers) {
        if (neededTracers == 0u || neededTracers <= tracerCapacity) {
            return;
        }
        if (d_renderTracers != nullptr) {
            cudaFree(d_renderTracers);
            d_renderTracers = nullptr;
        }
        ASTRAEA_CUDA_CHECK(cudaMalloc(&d_renderTracers, static_cast<size_t>(neededTracers) * 40u));
        tracerCapacity = neededTracers;
    }

    void uploadHostStateToDevice(const CombinedHostState& state) {
        const uint32_t n = state.count;
        ensureCapacity(std::max(n, 1u));
        p.count = n;
        alt.count = n;
        if (n == 0u) {
            return;
        }

        const size_t fBytes = static_cast<size_t>(n) * sizeof(float);
        const size_t u8Bytes = static_cast<size_t>(n) * sizeof(uint8_t);
        const size_t u32Bytes = static_cast<size_t>(n) * sizeof(uint32_t);
        const size_t i32Bytes = static_cast<size_t>(n) * sizeof(int32_t);

        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.posX, state.posX.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.posY, state.posY.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.posZ, state.posZ.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.velX, state.velX.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.velY, state.velY.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.velZ, state.velZ.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.velHalfX, state.velX.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.velHalfY, state.velY.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.velHalfZ, state.velZ.data(), fBytes, cudaMemcpyHostToDevice));

        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.mass, state.mass.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.h, state.h.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.h0, state.h0.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.rho, state.rho.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.u, state.u.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.tempK, state.tempK.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.epsFail, state.epsFail.data(), fBytes, cudaMemcpyHostToDevice));

        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.sXX, state.sXX.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.sYY, state.sYY.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.sZZ, state.sZZ.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.sXY, state.sXY.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.sXZ, state.sXZ.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.sYZ, state.sYZ.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.damage, state.damage.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.damageCbrt, state.damageCbrt.data(), fBytes, cudaMemcpyHostToDevice));

        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.materialId, state.materialId.data(), u8Bytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.bodyId, state.bodyId.data(), u8Bytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.originalId, state.originalId.data(), u32Bytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.tracerIndex, state.tracerIndex.data(), i32Bytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.tracerRestRadiusPsu, state.tracerRestRadiusPsu.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.tracerRestNormalX, state.tracerRestNormalX.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.tracerRestNormalY, state.tracerRestNormalY.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.tracerRestNormalZ, state.tracerRestNormalZ.data(), fBytes, cudaMemcpyHostToDevice));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(p.tracerIsOcean, state.tracerIsOcean.data(), u8Bytes, cudaMemcpyHostToDevice));

        ASTRAEA_CUDA_CHECK(cudaMemset(p.pressure, 0, fBytes));
        ASTRAEA_CUDA_CHECK(cudaMemset(p.soundSpeed, 0, fBytes));
        ASTRAEA_CUDA_CHECK(cudaMemset(p.phaseId, 0, u8Bytes));
        cuda::launchClearStats(p);
    }

    void evaluateInitialForcesOnDevice() {
        if (p.count == 0u) {
            return;
        }
        cuda::launchClearStats(p);
        cuda::launchSortAndBuildBvh(p, alt, bvh);
        cuda::launchSphPass1Neighbors(p, bvh);
        cuda::launchEvaluateEos(p);
        cuda::launchGravityBarnesHut(p, bvh, 0.5f);
        cuda::launchSphPass2Forces(p, bvh);
        ASTRAEA_CUDA_CHECK(cudaDeviceSynchronize());
    }

    void downloadTelemetryFieldsFromDevice() {
        const uint32_t n = p.count;
        if (n == 0u) {
            return;
        }
        const size_t fBytes = static_cast<size_t>(n) * sizeof(float);
        const size_t u8Bytes = static_cast<size_t>(n) * sizeof(uint8_t);
        const size_t u32Bytes = static_cast<size_t>(n) * sizeof(uint32_t);

        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posX, p.posX, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posY, p.posY, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posZ, p.posZ, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velX, p.velX, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velY, p.velY, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velZ, p.velZ, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.mass, p.mass, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.rho, p.rho, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.u, p.u, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.phiGrav, p.phiGrav, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.tempK, p.tempK, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.pressure, p.pressure, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.radPowerW, p.radPowerW, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.damage, p.damage, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.bodyId, p.bodyId, u8Bytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.materialId, p.materialId, u8Bytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.originalId, p.originalId, u32Bytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaDeviceSynchronize());
    }

    double sumDeviceRadPowerW() {
        const uint32_t n = p.count;
        if (n == 0u) {
            return 0.0;
        }
        ASTRAEA_CUDA_CHECK(cudaMemcpy(
            host.radPowerW, p.radPowerW, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
        double sum = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            const double w = static_cast<double>(host.radPowerW[i]);
            if (std::isfinite(w) && w > 0.0) {
                sum += w;
            }
        }
        return sum;
    }

    // Relaxes the single body currently uploaded in `p` (assumed centered at origin with zero spin/bulk velocity).
    void relaxLoadedSingleBody(
        const std::vector<double>& anchorTempByOrigId,
        double nominalRadiusPsu,
        uint32_t steps,
        double& outFinalVrmsOverVesc) {
        const uint32_t n = p.count;
        if (n == 0u) {
            outFinalVrmsOverVesc = 0.0;
            return;
        }

        evaluateInitialForcesOnDevice();

        float localPrevDt = 0.0f;
        const size_t fBytes = static_cast<size_t>(n) * sizeof(float);
        const size_t u8Bytes = static_cast<size_t>(n) * sizeof(uint8_t);
        const size_t u32Bytes = static_cast<size_t>(n) * sizeof(uint32_t);

        for (uint32_t s = 0; s < steps; ++s) {
            const float dtPsu = cuda::launchReduceDtPsu(p, localPrevDt, 0.02f);
            localPrevDt = dtPsu;

            cuda::launchLeapfrogKickHalf(p, 0.5f * dtPsu, true, true);
            cuda::launchLeapfrogDrift(p, dtPsu, bvh);
            cuda::launchSortAndBuildBvh(p, alt, bvh);
            cuda::launchSphPass1Neighbors(p, bvh);
            cuda::launchEvaluateEos(p);
            cuda::launchGravityBarnesHut(p, bvh, 0.5f);
            cuda::launchSphPass2Forces(p, bvh);
            cuda::launchLeapfrogKickHalf(p, 0.5f * dtPsu, false, true);

            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posX, p.posX, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posY, p.posY, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posZ, p.posZ, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velX, p.velX, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velY, p.velY, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velZ, p.velZ, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.mass, p.mass, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.rho, p.rho, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.materialId, p.materialId, u8Bytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.originalId, p.originalId, u32Bytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaDeviceSynchronize());

            double mTot = 0.0;
            double comX = 0.0, comY = 0.0, comZ = 0.0;
            double vComX = 0.0, vComY = 0.0, vComZ = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                const double m = static_cast<double>(host.mass[i]);
                mTot += m;
                comX += m * static_cast<double>(host.posX[i]);
                comY += m * static_cast<double>(host.posY[i]);
                comZ += m * static_cast<double>(host.posZ[i]);
                vComX += m * static_cast<double>(host.velX[i]);
                vComY += m * static_cast<double>(host.velY[i]);
                vComZ += m * static_cast<double>(host.velZ[i]);
            }
            if (mTot > 0.0) {
                comX /= mTot; comY /= mTot; comZ /= mTot;
                vComX /= mTot; vComY /= mTot; vComZ /= mTot;
            }

            for (uint32_t i = 0; i < n; ++i) {
                host.posX[i] = static_cast<float>(static_cast<double>(host.posX[i]) - comX);
                host.posY[i] = static_cast<float>(static_cast<double>(host.posY[i]) - comY);
                host.posZ[i] = static_cast<float>(static_cast<double>(host.posZ[i]) - comZ);
                host.velX[i] = static_cast<float>(0.85 * (static_cast<double>(host.velX[i]) - vComX));
                host.velY[i] = static_cast<float>(0.85 * (static_cast<double>(host.velY[i]) - vComY));
                host.velZ[i] = static_cast<float>(0.85 * (static_cast<double>(host.velZ[i]) - vComZ));

                const uint32_t oid = host.originalId[i];
                const double tAnchor = (oid < anchorTempByOrigId.size()) ? anchorTempByOrigId[oid] : 288.0;
                const MaterialId mat = static_cast<MaterialId>(host.materialId[i] <= 4u ? host.materialId[i] : 1u);
                const double cvPsu = eos::getParams(mat).cv_JKgK / units::U_UNIT_JKG;
                const double uCold = eos::computeColdEnergyPsu(mat, static_cast<double>(host.rho[i]));
                host.u[i] = static_cast<float>(uCold + cvPsu * tAnchor);
            }

            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.posX, host.posX, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.posY, host.posY, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.posZ, host.posZ, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.velX, host.velX, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.velY, host.velY, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.velZ, host.velZ, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.velHalfX, host.velX, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.velHalfY, host.velY, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.velHalfZ, host.velZ, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.u, host.u, fBytes, cudaMemcpyHostToDevice));

            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sXX, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sYY, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sZZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sXY, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sXZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sYZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.damage, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.damageCbrt, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSXX, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSYY, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSZZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSXY, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSXZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSYZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dDamageCbrtDt, 0, fBytes));

            if (s + 1u < steps) {
                cuda::launchEvaluateEos(p);
                cuda::launchGravityBarnesHut(p, bvh, 0.5f);
                cuda::launchSphPass2Forces(p, bvh);
            }
        }

        // Run 4 undamped diagnostic steps if relaxation ran, to measure true residual velocity jitter
        if (steps > 0u) {
            cuda::launchSortAndBuildBvh(p, alt, bvh);
            cuda::launchSphPass1Neighbors(p, bvh);
            cuda::launchEvaluateEos(p);
            cuda::launchGravityBarnesHut(p, bvh, 0.5f);
            cuda::launchSphPass2Forces(p, bvh);
            for (int c = 0; c < 4; ++c) {
                const float dtPsu = cuda::launchReduceDtPsu(p, localPrevDt, 0.02f);
                localPrevDt = dtPsu;
                cuda::launchLeapfrogKickHalf(p, 0.5f * dtPsu, true, true);
                cuda::launchLeapfrogDrift(p, dtPsu, bvh);
                cuda::launchSortAndBuildBvh(p, alt, bvh);
                cuda::launchSphPass1Neighbors(p, bvh);
                cuda::launchEvaluateEos(p);
                cuda::launchGravityBarnesHut(p, bvh, 0.5f);
                cuda::launchSphPass2Forces(p, bvh);
                cuda::launchLeapfrogKickHalf(p, 0.5f * dtPsu, false, true);
            }
            cuda::launchEvaluateEos(p);
        }

        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posX, p.posX, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posY, p.posY, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.posZ, p.posZ, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velX, p.velX, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velY, p.velY, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.velZ, p.velZ, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.mass, p.mass, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaDeviceSynchronize());

        double mTot = 0.0;
        double sumMVel2 = 0.0;
        double rMax = nominalRadiusPsu;
        for (uint32_t i = 0; i < n; ++i) {
            const double m = static_cast<double>(host.mass[i]);
            const double vx = static_cast<double>(host.velX[i]);
            const double vy = static_cast<double>(host.velY[i]);
            const double vz = static_cast<double>(host.velZ[i]);
            const double rx = static_cast<double>(host.posX[i]);
            const double ry = static_cast<double>(host.posY[i]);
            const double rz = static_cast<double>(host.posZ[i]);
            mTot += m;
            sumMVel2 += m * (vx * vx + vy * vy + vz * vz);
            rMax = std::max(rMax, std::sqrt(rx * rx + ry * ry + rz * rz));
        }

        const double vRms = (mTot > 0.0) ? std::sqrt(sumMVel2 / mTot) : 0.0;
        const double rScale = std::max(rMax, 1.0e-3);
        const double vEsc = std::sqrt((2.0 * units::G_PSU * mTot) / rScale);
        outFinalVrmsOverVesc = (vEsc > 0.0) ? (vRms / vEsc) : 0.0;

        // Re-anchor thermal profile and clear any diagnostic shear/damage so t=0 starts cleanly at rest geotherm
        if (steps > 0u) {
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.rho, p.rho, fBytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.materialId, p.materialId, u8Bytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(host.originalId, p.originalId, u32Bytes, cudaMemcpyDeviceToHost));
            ASTRAEA_CUDA_CHECK(cudaDeviceSynchronize());
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t oid = host.originalId[i];
                const double tAnchor = (oid < anchorTempByOrigId.size()) ? anchorTempByOrigId[oid] : 288.0;
                const MaterialId mat = static_cast<MaterialId>(host.materialId[i] <= 4u ? host.materialId[i] : 1u);
                const double cvPsu = eos::getParams(mat).cv_JKgK / units::U_UNIT_JKG;
                const double uCold = eos::computeColdEnergyPsu(mat, static_cast<double>(host.rho[i]));
                host.u[i] = static_cast<float>(uCold + cvPsu * tAnchor);
            }
            ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(p.u, host.u, fBytes, cudaMemcpyHostToDevice));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sXX, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sYY, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sZZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sXY, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sXZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.sYZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.damage, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.damageCbrt, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSXX, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSYY, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSZZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSXY, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSXZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dSYZ, 0, fBytes));
            ASTRAEA_CUDA_CHECK(cudaMemsetAsync(p.dDamageCbrtDt, 0, fBytes));
            cuda::launchEvaluateEos(p);
            ASTRAEA_CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    void copyRelaxedBodyToBuiltHost(BuiltPlanetHost& body, uint32_t n) {
        const size_t fBytes = static_cast<size_t>(n) * sizeof(float);
        const size_t u32Bytes = static_cast<size_t>(n) * sizeof(uint32_t);
        const size_t i32Bytes = static_cast<size_t>(n) * sizeof(int32_t);
        const size_t u8Bytes = static_cast<size_t>(n) * sizeof(uint8_t);

        std::vector<float> hBuf(n);
        std::vector<float> tempBuf(n);
        std::vector<int32_t> trIdx(n);
        std::vector<uint8_t> trOcean(n);
        std::vector<uint32_t> origIds(n);

        ASTRAEA_CUDA_CHECK(cudaMemcpy(host.posX, p.posX, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(host.posY, p.posY, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(host.posZ, p.posZ, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(host.rho, p.rho, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(host.u, p.u, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(host.mass, p.mass, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(hBuf.data(), p.h0, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(tempBuf.data(), p.tempK, fBytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(host.materialId, p.materialId, u8Bytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(origIds.data(), p.originalId, u32Bytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(trIdx.data(), p.tracerIndex, i32Bytes, cudaMemcpyDeviceToHost));
        ASTRAEA_CUDA_CHECK(cudaMemcpy(trOcean.data(), p.tracerIsOcean, u8Bytes, cudaMemcpyDeviceToHost));
        std::vector<float> epsFailBuf(n);
        ASTRAEA_CUDA_CHECK(cudaMemcpy(epsFailBuf.data(), p.epsFail, fBytes, cudaMemcpyDeviceToHost));

        // Reorder back to originalId order to preserve deterministic ordering
        double rMaxPsu = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t dst = (origIds[i] < n) ? origIds[i] : i;
            body.posX[dst] = host.posX[i];
            body.posY[dst] = host.posY[i];
            body.posZ[dst] = host.posZ[i];
            body.velX[dst] = 0.0f;
            body.velY[dst] = 0.0f;
            body.velZ[dst] = 0.0f;
            body.mass[dst] = host.mass[i];
            body.h0[dst] = hBuf[i];
            body.rho[dst] = host.rho[i];
            body.u[dst] = host.u[i];
            body.tempK[dst] = tempBuf[i];
            body.epsFail[dst] = epsFailBuf[i];
            body.materialId[dst] = host.materialId[i];
            body.tracerIndex[dst] = trIdx[i];
            body.tracerIsOcean[dst] = trOcean[i];

            const double r = std::sqrt(
                static_cast<double>(host.posX[i]) * host.posX[i] +
                static_cast<double>(host.posY[i]) * host.posY[i] +
                static_cast<double>(host.posZ[i]) * host.posZ[i]);
            rMaxPsu = std::max(rMaxPsu, r);

            if (trIdx[i] >= 0) {
                body.tracerRestRadiusPsu[dst] = static_cast<float>(r);
                if (r > 1.0e-9) {
                    const double invR = 1.0 / r;
                    body.tracerRestNormalX[dst] = static_cast<float>(host.posX[i] * invR);
                    body.tracerRestNormalY[dst] = static_cast<float>(host.posY[i] * invR);
                    body.tracerRestNormalZ[dst] = static_cast<float>(host.posZ[i] * invR);
                }
            }
        }
        if (rMaxPsu > 0.0) {
            body.equatorialRadiusM = rMaxPsu * units::L_UNIT_M;
        }
        centerAndZeroBulkVelocity(body);
    }

    static CombinedHostState makeCombinedFromSingleBody(const BuiltPlanetHost& body) {
        CombinedHostState s;
        const uint32_t n = static_cast<uint32_t>(body.posX.size());
        s.count = n;
        s.posX = body.posX; s.posY = body.posY; s.posZ = body.posZ;
        s.velX = body.velX; s.velY = body.velY; s.velZ = body.velZ;
        s.mass = body.mass;
        s.h = body.h0;
        s.h0 = body.h0;
        s.rho = body.rho;
        s.u = body.u;
        s.tempK = body.tempK;
        s.epsFail = body.epsFail;
        s.sXX.assign(n, 0.0f); s.sYY.assign(n, 0.0f); s.sZZ.assign(n, 0.0f);
        s.sXY.assign(n, 0.0f); s.sXZ.assign(n, 0.0f); s.sYZ.assign(n, 0.0f);
        s.damage.assign(n, 0.0f);
        s.damageCbrt.assign(n, 0.0f);
        s.materialId = body.materialId;
        s.bodyId = body.bodyId;
        s.originalId.resize(n);
        std::iota(s.originalId.begin(), s.originalId.end(), 0u);
        s.tracerIndex = body.tracerIndex;
        s.tracerRestRadiusPsu = body.tracerRestRadiusPsu;
        s.tracerRestNormalX = body.tracerRestNormalX;
        s.tracerRestNormalY = body.tracerRestNormalY;
        s.tracerRestNormalZ = body.tracerRestNormalZ;
        s.tracerIsOcean = body.tracerIsOcean;
        return s;
    }

    static CombinedHostState mergeTwoBodies(const BuiltPlanetHost& b0, const BuiltPlanetHost& b1) {
        CombinedHostState s = makeCombinedFromSingleBody(b0);
        const uint32_t n0 = s.count;
        const uint32_t n1 = static_cast<uint32_t>(b1.posX.size());
        const uint32_t n = n0 + n1;
        s.count = n;

        auto appendF = [](std::vector<float>& dst, const std::vector<float>& src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };
        appendF(s.posX, b1.posX); appendF(s.posY, b1.posY); appendF(s.posZ, b1.posZ);
        appendF(s.velX, b1.velX); appendF(s.velY, b1.velY); appendF(s.velZ, b1.velZ);
        appendF(s.mass, b1.mass);
        appendF(s.h, b1.h0);
        appendF(s.h0, b1.h0);
        appendF(s.rho, b1.rho);
        appendF(s.u, b1.u);
        appendF(s.tempK, b1.tempK);
        appendF(s.epsFail, b1.epsFail);

        s.sXX.assign(n, 0.0f); s.sYY.assign(n, 0.0f); s.sZZ.assign(n, 0.0f);
        s.sXY.assign(n, 0.0f); s.sXZ.assign(n, 0.0f); s.sYZ.assign(n, 0.0f);
        s.damage.assign(n, 0.0f);
        s.damageCbrt.assign(n, 0.0f);

        s.materialId.insert(s.materialId.end(), b1.materialId.begin(), b1.materialId.end());
        s.bodyId.insert(s.bodyId.end(), b1.bodyId.begin(), b1.bodyId.end());
        s.originalId.resize(n);
        std::iota(s.originalId.begin(), s.originalId.end(), 0u);

        s.tracerIndex.insert(s.tracerIndex.end(), b1.tracerIndex.begin(), b1.tracerIndex.end());
        appendF(s.tracerRestRadiusPsu, b1.tracerRestRadiusPsu);
        appendF(s.tracerRestNormalX, b1.tracerRestNormalX);
        appendF(s.tracerRestNormalY, b1.tracerRestNormalY);
        appendF(s.tracerRestNormalZ, b1.tracerRestNormalZ);
        s.tracerIsOcean.insert(s.tracerIsOcean.end(), b1.tracerIsOcean.begin(), b1.tracerIsOcean.end());

        // Enforce exact system COM = 0 and net momentum = 0
        double mTot = 0.0;
        double cx = 0.0, cy = 0.0, cz = 0.0;
        double vx = 0.0, vy = 0.0, vz = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            const double m = static_cast<double>(s.mass[i]);
            mTot += m;
            cx += m * static_cast<double>(s.posX[i]);
            cy += m * static_cast<double>(s.posY[i]);
            cz += m * static_cast<double>(s.posZ[i]);
            vx += m * static_cast<double>(s.velX[i]);
            vy += m * static_cast<double>(s.velY[i]);
            vz += m * static_cast<double>(s.velZ[i]);
        }
        if (mTot > 0.0) {
            cx /= mTot; cy /= mTot; cz /= mTot;
            vx /= mTot; vy /= mTot; vz /= mTot;
            for (uint32_t i = 0; i < n; ++i) {
                s.posX[i] = static_cast<float>(static_cast<double>(s.posX[i]) - cx);
                s.posY[i] = static_cast<float>(static_cast<double>(s.posY[i]) - cy);
                s.posZ[i] = static_cast<float>(static_cast<double>(s.posZ[i]) - cz);
                s.velX[i] = static_cast<float>(static_cast<double>(s.velX[i]) - vx);
                s.velY[i] = static_cast<float>(static_cast<double>(s.velY[i]) - vy);
                s.velZ[i] = static_cast<float>(static_cast<double>(s.velZ[i]) - vz);
            }
        }
        return s;
    }

    TelemetrySnapshot computeTelemetryFromDevice() {
        TelemetrySnapshot snap;
        snap.frameSeq = frameSeq;
        snap.simTimeSec = simTimeSec;
        snap.dtSec = lastDtSec;
        snap.gpuStepMs = lastGpuStepMs;
        snap.particleCount = p.count;
        snap.tracerCount = tracerCount;
        snap.tracerGridRes = tracerGridRes;
        snap.livingBodyPresent = livingBodyPresent;
        snap.radiatedEnergyJ = radiatedEnergyJ;
        snap.initialTotalEnergyJ = initialTotalEnergyJ;
        snap.qRatioQrd = qRatioQrd;

        const uint32_t n = p.count;
        if (n == 0u) {
            return snap;
        }

        downloadTelemetryFieldsFromDevice();

        double totalMassPsu = 0.0;
        double sysX = 0.0, sysY = 0.0, sysZ = 0.0;
        double m0 = 0.0, x0 = 0.0, y0 = 0.0, z0 = 0.0, vx0 = 0.0, vy0 = 0.0, vz0 = 0.0;
        double m1 = 0.0, x1 = 0.0, y1 = 0.0, z1 = 0.0, vx1 = 0.0, vy1 = 0.0, vz1 = 0.0;

        double kePsu = 0.0;
        double pePsu = 0.0;
        double iePsu = 0.0;
        double pxPsu = 0.0, pyPsu = 0.0, pzPsu = 0.0;
        double lxPsu = 0.0, lyPsu = 0.0, lzPsu = 0.0;
        double maxTemp = 0.0;
        double maxPresGPa = 0.0;

        uint32_t seedIdx = 0;
        double minPhi = std::numeric_limits<double>::infinity();

        for (uint32_t i = 0; i < n; ++i) {
            const double m = static_cast<double>(host.mass[i]);
            const double x = static_cast<double>(host.posX[i]);
            const double y = static_cast<double>(host.posY[i]);
            const double z = static_cast<double>(host.posZ[i]);
            const double vx = static_cast<double>(host.velX[i]);
            const double vy = static_cast<double>(host.velY[i]);
            const double vz = static_cast<double>(host.velZ[i]);
            const double phi = static_cast<double>(host.phiGrav[i]);
            const double uVal = static_cast<double>(host.u[i]);
            const double temp = static_cast<double>(host.tempK[i]);
            const double presGPa = static_cast<double>(host.pressure[i]) * 1000.0; // P_UNIT = 1 TPa = 1000 GPa

            totalMassPsu += m;
            sysX += m * x; sysY += m * y; sysZ += m * z;

            if (host.bodyId[i] == 0u) {
                m0 += m;
                x0 += m * x; y0 += m * y; z0 += m * z;
                vx0 += m * vx; vy0 += m * vy; vz0 += m * vz;
            } else {
                m1 += m;
                x1 += m * x; y1 += m * y; z1 += m * z;
                vx1 += m * vx; vy1 += m * vy; vz1 += m * vz;
            }

            kePsu += 0.5 * m * (vx * vx + vy * vy + vz * vz);
            pePsu += 0.5 * m * phi;
            iePsu += m * uVal;

            pxPsu += m * vx;
            pyPsu += m * vy;
            pzPsu += m * vz;

            lxPsu += m * (y * vz - z * vy);
            lyPsu += m * (z * vx - x * vz);
            lzPsu += m * (x * vy - y * vx);

            if (std::isfinite(temp)) {
                maxTemp = std::max(maxTemp, temp);
            }
            if (std::isfinite(presGPa)) {
                maxPresGPa = std::max(maxPresGPa, presGPa);
            }
            if (std::isfinite(phi) && phi < minPhi) {
                minPhi = phi;
                seedIdx = i;
            }
        }

        if (totalMassPsu > 0.0) {
            sysComPsu[0] = sysX / totalMassPsu;
            sysComPsu[1] = sysY / totalMassPsu;
            sysComPsu[2] = sysZ / totalMassPsu;
        } else {
            sysComPsu[0] = sysComPsu[1] = sysComPsu[2] = 0.0;
        }

        if (m0 > 0.0) {
            x0 /= m0; y0 /= m0; z0 /= m0;
            vx0 /= m0; vy0 /= m0; vz0 /= m0;
            livingComPsu[0] = x0;
            livingComPsu[1] = y0;
            livingComPsu[2] = z0;
        } else {
            livingComPsu[0] = sysComPsu[0];
            livingComPsu[1] = sysComPsu[1];
            livingComPsu[2] = sysComPsu[2];
        }

        if (m1 > 0.0) {
            x1 /= m1; y1 /= m1; z1 /= m1;
            vx1 /= m1; vy1 /= m1; vz1 /= m1;
        }

        snap.kineticEnergyJ = kePsu * units::E_UNIT_J;
        snap.potentialEnergyJ = pePsu * units::E_UNIT_J;
        snap.internalEnergyJ = iePsu * units::E_UNIT_J;
        snap.totalEnergyJ = snap.kineticEnergyJ + snap.potentialEnergyJ + snap.internalEnergyJ;
        if (frameSeq == 0u && simTimeSec == 0.0) {
            initialTotalEnergyJ = snap.totalEnergyJ;
            snap.initialTotalEnergyJ = initialTotalEnergyJ;
            snap.energyDriftRel = 0.0;
        } else {
            const double denom = std::max(std::abs(initialTotalEnergyJ), 1.0);
            snap.energyDriftRel = (snap.totalEnergyJ + radiatedEnergyJ - initialTotalEnergyJ) / denom;
        }

        constexpr double kMomUnit = units::M_UNIT_KG * units::V_UNIT_MS;
        snap.momentumKgMS[0] = pxPsu * kMomUnit;
        snap.momentumKgMS[1] = pyPsu * kMomUnit;
        snap.momentumKgMS[2] = pzPsu * kMomUnit;

        snap.angularMomentumKgM2S[0] = lxPsu * units::L_ANG_UNIT;
        snap.angularMomentumKgM2S[1] = lyPsu * units::L_ANG_UNIT;
        snap.angularMomentumKgM2S[2] = lzPsu * units::L_ANG_UNIT;

        snap.virialRatio = (std::abs(snap.potentialEnergyJ) > 1.0)
                               ? (-2.0 * snap.kineticEnergyJ / snap.potentialEnergyJ)
                               : 0.0;
        snap.maxTempK = maxTemp;
        snap.maxPressureGPa = maxPresGPa;

        if (livingBodyPresent) {
            snap.livingBodyComKm[0] = (livingComPsu[0] - sysComPsu[0]) * 1000.0;
            snap.livingBodyComKm[1] = (livingComPsu[1] - sysComPsu[1]) * 1000.0;
            snap.livingBodyComKm[2] = (livingComPsu[2] - sysComPsu[2]) * 1000.0;
            snap.livingBodyRadiusKm = targetRadiusPsu * 1000.0;

            double lSpinY = 0.0;
            double iYY = 0.0;
            const double rLimSq = (1.5 * targetRadiusPsu) * (1.5 * targetRadiusPsu);
            for (uint32_t i = 0; i < n; ++i) {
                if (host.bodyId[i] != 0u) {
                    continue;
                }
                const double dx = static_cast<double>(host.posX[i]) - x0;
                const double dy = static_cast<double>(host.posY[i]) - y0;
                const double dz = static_cast<double>(host.posZ[i]) - z0;
                if (dx * dx + dy * dy + dz * dz <= rLimSq) {
                    const double m = static_cast<double>(host.mass[i]);
                    const double dvx = static_cast<double>(host.velX[i]) - vx0;
                    const double dvz = static_cast<double>(host.velZ[i]) - vz0;
                    lSpinY += m * (dz * dvx - dx * dvz);
                    iYY += m * (dx * dx + dz * dz);
                }
            }
            snap.livingBodySpinRadS = (iYY > 0.0) ? ((lSpinY / iYY) / units::T_UNIT_S) : config.target.spinOmegaRadS;
        }

        if (m1 > 0.0) {
            snap.impactorComKm[0] = (x1 - sysComPsu[0]) * 1000.0;
            snap.impactorComKm[1] = (y1 - sysComPsu[1]) * 1000.0;
            snap.impactorComKm[2] = (z1 - sysComPsu[2]) * 1000.0;
        }

        // Iterative largest-remnant gravitational boundness classification (planet vs disk vs ejecta)
        double remX = static_cast<double>(host.posX[seedIdx]);
        double remY = static_cast<double>(host.posY[seedIdx]);
        double remZ = static_cast<double>(host.posZ[seedIdx]);
        double remVx = static_cast<double>(host.velX[seedIdx]);
        double remVy = static_cast<double>(host.velY[seedIdx]);
        double remVz = static_cast<double>(host.velZ[seedIdx]);
        double remMass = std::max(m0, totalMassPsu * 0.5);
        double remRadius = std::max(targetRadiusPsu, 0.25);

        for (int iter = 0; iter < 5; ++iter) {
            double nextM = 0.0;
            double nextX = 0.0, nextY = 0.0, nextZ = 0.0;
            double nextVx = 0.0, nextVy = 0.0, nextVz = 0.0;
            double sumMR2 = 0.0;
            const double rSoft = std::max(0.5 * remRadius, 0.1);

            for (uint32_t i = 0; i < n; ++i) {
                const double dx = static_cast<double>(host.posX[i]) - remX;
                const double dy = static_cast<double>(host.posY[i]) - remY;
                const double dz = static_cast<double>(host.posZ[i]) - remZ;
                const double r2 = dx * dx + dy * dy + dz * dz;
                const double r = std::sqrt(r2);

                const double dvx = static_cast<double>(host.velX[i]) - remVx;
                const double dvy = static_cast<double>(host.velY[i]) - remVy;
                const double dvz = static_cast<double>(host.velZ[i]) - remVz;
                const double v2 = dvx * dvx + dvy * dvy + dvz * dvz;

                const double phiRem = -units::G_PSU * remMass / std::max(r, rSoft);
                const double eSpec = 0.5 * v2 + phiRem;
                if (eSpec < 0.0) {
                    const double m = static_cast<double>(host.mass[i]);
                    nextM += m;
                    nextX += m * static_cast<double>(host.posX[i]);
                    nextY += m * static_cast<double>(host.posY[i]);
                    nextZ += m * static_cast<double>(host.posZ[i]);
                    nextVx += m * static_cast<double>(host.velX[i]);
                    nextVy += m * static_cast<double>(host.velY[i]);
                    nextVz += m * static_cast<double>(host.velZ[i]);
                    sumMR2 += m * r2;
                }
            }
            if (nextM > 0.0) {
                remMass = nextM;
                remX = nextX / nextM;
                remY = nextY / nextM;
                remZ = nextZ / nextM;
                remVx = nextVx / nextM;
                remVy = nextVy / nextM;
                remVz = nextVz / nextM;
                const double rmsR = std::sqrt(sumMR2 / nextM);
                remRadius = std::max(0.5 * targetRadiusPsu, std::sqrt(5.0 / 3.0) * rmsR);
            } else {
                break;
            }
        }

        double boundPlanetMass = 0.0;
        double diskMass = 0.0;
        double ejectaMass = 0.0;
        double impactorBoundMass = 0.0;
        double maxDamage = 0.0;

        const double rSoftFinal = std::max(0.5 * remRadius, 0.1);
        const double muRem = units::G_PSU * remMass;

        for (uint32_t i = 0; i < n; ++i) {
            const double m = static_cast<double>(host.mass[i]);
            maxDamage = std::max(maxDamage, static_cast<double>(host.damage[i]));

            const double dx = static_cast<double>(host.posX[i]) - remX;
            const double dy = static_cast<double>(host.posY[i]) - remY;
            const double dz = static_cast<double>(host.posZ[i]) - remZ;
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

            const double dvx = static_cast<double>(host.velX[i]) - remVx;
            const double dvy = static_cast<double>(host.velY[i]) - remVy;
            const double dvz = static_cast<double>(host.velZ[i]) - remVz;
            const double v2 = dvx * dvx + dvy * dvy + dvz * dvz;

            const double phiRem = -muRem / std::max(r, rSoftFinal);
            const double eSpec = 0.5 * v2 + phiRem;

            if (eSpec >= 0.0) {
                ejectaMass += m;
            } else {
                if (host.bodyId[i] != 0u) {
                    impactorBoundMass += m;
                }
                bool isDisk = false;
                if (r > remRadius && muRem > 0.0) {
                    const double jx = dy * dvz - dz * dvy;
                    const double jy = dz * dvx - dx * dvz;
                    const double jz = dx * dvy - dy * dvx;
                    const double j2 = jx * jx + jy * jy + jz * jz;
                    const double ecc2 = std::max(0.0, 1.0 + (2.0 * eSpec * j2) / (muRem * muRem));
                    const double ecc = std::sqrt(ecc2);
                    const double rPeri = j2 / (muRem * (1.0 + ecc));
                    if (rPeri > remRadius) {
                        isDisk = true;
                    }
                }
                if (isDisk) {
                    diskMass += m;
                } else {
                    boundPlanetMass += m;
                }
            }
        }

        if (totalMassPsu > 0.0) {
            snap.boundMassFrac = boundPlanetMass / totalMassPsu;
            snap.diskMassFrac = diskMass / totalMassPsu;
            snap.ejectaMassFrac = ejectaMass / totalMassPsu;
        }

        // Collision outcome classification
        if (m1 <= 0.0) {
            snap.outcome = CollisionOutcome::Merging;
        } else {
            const double rx = x1 - x0;
            const double ry = y1 - y0;
            const double rz = z1 - z0;
            const double dSep = std::sqrt(rx * rx + ry * ry + rz * rz);
            const double rvx = vx1 - vx0;
            const double rvy = vy1 - vy0;
            const double rvz = vz1 - vz0;
            const double rDotV = rx * rvx + ry * rvy + rz * rvz;
            const double contactDist = targetRadiusPsu + impactorRadiusPsu;

            if (dSep <= 1.05 * contactDist ||
                maxDamage > 0.25 ||
                rDotV >= 0.0) {
                hasImpacted = true;
            }

            if (!hasImpacted && dSep > 1.05 * contactDist && rDotV < 0.0) {
                snap.outcome = CollisionOutcome::Approaching;
            } else {
                const double impactorBoundFrac = (m1 > 0.0) ? (impactorBoundMass / m1) : 1.0;
                const double bAbs = std::abs(config.impactParamB);
                const double twoBodyEnergySpec =
                    0.5 * (rvx * rvx + rvy * rvy + rvz * rvz) -
                    units::G_PSU * (m0 + m1) / std::max(dSep, 1.0e-3);

                if (snap.boundMassFrac < 0.50 ||
                    (qRatioQrd >= 1.0 && (snap.ejectaMassFrac >= 0.15 || snap.boundMassFrac < 0.72))) {
                    snap.outcome = CollisionOutcome::CatastrophicDisruption;
                } else if (rDotV > 0.0 && dSep > 1.05 * contactDist &&
                           (impactorBoundFrac < 0.50 || twoBodyEnergySpec > 0.0)) {
                    snap.outcome = CollisionOutcome::HitAndRun;
                } else if (bAbs >= 0.50 && config.impactVelocityRatioVesc > 1.25 && impactorBoundFrac < 0.45) {
                    snap.outcome = CollisionOutcome::HitAndRun;
                } else if (bAbs >= 0.45) {
                    snap.outcome = CollisionOutcome::GrazeAndMerge;
                } else {
                    snap.outcome = CollisionOutcome::Merging;
                }
            }
        }

        lastSnapshot = snap;
        return snap;
    }
};

Simulation::Simulation(uint32_t maxParticles)
    : impl_(std::make_unique<Impl>(maxParticles)) {}

Simulation::~Simulation() = default;
Simulation::Simulation(Simulation&&) noexcept = default;
Simulation& Simulation::operator=(Simulation&&) noexcept = default;

void Simulation::loadScenario(const CollisionScenarioConfig& config) {
    if (config.totalParticles == 0u) {
        throw std::invalid_argument("Simulation::loadScenario requires totalParticles > 0");
    }

    impl_->ensureCapacity(config.totalParticles);
    impl_->config = config;
    impl_->frameSeq = 0;
    impl_->simTimeSec = 0.0;
    impl_->lastDtSec = 0.0f;
    impl_->prevDtPsu = 0.0f;
    impl_->radiatedEnergyJ = 0.0;
    impl_->prevTotalRadPowerW = 0.0;
    impl_->initialTotalEnergyJ = 0.0;
    impl_->lastGpuStepMs = 0.0;
    impl_->hasImpacted = false;
    impl_->history.clear();

    const bool hasImpactor = (config.impactor.massKg > 0.0) && (config.totalParticles >= 2u);
    uint32_t targetN = config.totalParticles;
    uint32_t impactorN = 0;
    if (hasImpactor) {
        const double mTot = config.target.massKg + config.impactor.massKg;
        uint32_t impN = static_cast<uint32_t>(
            std::llround(static_cast<double>(config.totalParticles) * (config.impactor.massKg / mTot)));
        const uint32_t minImpN = std::min(config.totalParticles / 2u, 16u);
        impN = std::clamp(impN, minImpN, config.totalParticles - 1u);
        impactorN = impN;
        targetN = config.totalParticles - impN;
    }

    BuiltPlanetHost targetHost = PlanetBuilder::buildPlanet(
        config.target, targetN, 0, config.tracerGridRes, config.randomSeed);

    if (config.target.relaxSteps > 0u) {
        centerAndZeroBulkVelocity(targetHost);
        // Zero spin during relaxation so spherical hydrostatic equilibrium settles cleanly
        std::fill(targetHost.velX.begin(), targetHost.velX.end(), 0.0f);
        std::fill(targetHost.velY.begin(), targetHost.velY.end(), 0.0f);
        std::fill(targetHost.velZ.begin(), targetHost.velZ.end(), 0.0f);

        std::vector<double> anchorTemps(targetN);
        for (uint32_t i = 0; i < targetN; ++i) {
            anchorTemps[i] = static_cast<double>(targetHost.tempK[i]);
        }

        CombinedHostState tmp = Impl::makeCombinedFromSingleBody(targetHost);
        impl_->uploadHostStateToDevice(tmp);
        double dummyVrms = 0.0;
        impl_->relaxLoadedSingleBody(
            anchorTemps,
            targetHost.equatorialRadiusM / units::L_UNIT_M,
            config.target.relaxSteps,
            dummyVrms);
        impl_->copyRelaxedBodyToBuiltHost(targetHost, targetN);
        applyRigidSpinAroundY(targetHost, config.target.spinOmegaRadS);
    }

    impl_->targetRadiusPsu = targetHost.equatorialRadiusM / units::L_UNIT_M;
    impl_->tracerCount = targetHost.tracerCount;
    impl_->tracerGridRes = (targetHost.tracerCount > 0u) ? config.tracerGridRes : 0u;
    impl_->livingBodyPresent = (config.target.archetype == PlanetArchetype::LivingTerra);
    impl_->ensureTracerCapacity(impl_->tracerCount);

    CombinedHostState combined;
    if (hasImpactor) {
        BuiltPlanetHost impactorHost = PlanetBuilder::buildPlanet(
            config.impactor, impactorN, 1, 0, config.randomSeed + 1ULL);

        if (config.impactor.relaxSteps > 0u) {
            centerAndZeroBulkVelocity(impactorHost);
            std::fill(impactorHost.velX.begin(), impactorHost.velX.end(), 0.0f);
            std::fill(impactorHost.velY.begin(), impactorHost.velY.end(), 0.0f);
            std::fill(impactorHost.velZ.begin(), impactorHost.velZ.end(), 0.0f);

            std::vector<double> anchorTemps(impactorN);
            for (uint32_t i = 0; i < impactorN; ++i) {
                anchorTemps[i] = static_cast<double>(impactorHost.tempK[i]);
            }

            CombinedHostState tmp = Impl::makeCombinedFromSingleBody(impactorHost);
            impl_->uploadHostStateToDevice(tmp);
            double dummyVrms = 0.0;
            impl_->relaxLoadedSingleBody(
                anchorTemps,
                impactorHost.equatorialRadiusM / units::L_UNIT_M,
                config.impactor.relaxSteps,
                dummyVrms);
            impl_->copyRelaxedBodyToBuiltHost(impactorHost, impactorN);
            applyRigidSpinAroundY(impactorHost, config.impactor.spinOmegaRadS);
        }

        impl_->impactorRadiusPsu = impactorHost.equatorialRadiusM / units::L_UNIT_M;

        // Place target and impactor on exact two-body Keplerian approach orbit in total COM frame
        double mTPsu = 0.0;
        for (float m : targetHost.mass) {
            mTPsu += static_cast<double>(m);
        }
        double mIPsu = 0.0;
        for (float m : impactorHost.mass) {
            mIPsu += static_cast<double>(m);
        }
        const double mTotPsu = mTPsu + mIPsu;

        const double rContact = std::max(impl_->targetRadiusPsu + impl_->impactorRadiusPsu, 1.0e-3);
        const double vEscPsu = std::sqrt((2.0 * units::G_PSU * mTotPsu) / rContact);
        const double vRatio = std::max(0.0, config.impactVelocityRatioVesc);
        const double vContactPsu = vRatio * vEscPsu;

        double d0 = std::max(config.initialSepRadii, 1.0) * rContact;
        double v0Sq = vContactPsu * vContactPsu -
                      2.0 * units::G_PSU * mTotPsu * (1.0 / rContact - 1.0 / d0);
        if (v0Sq < 0.0) {
            // Sub-escape orbit: requested contact speed exceeds free-fall from d0; clamp d0 within apoapsis
            const double rApo = rContact / std::max(1.0 - vRatio * vRatio, 1.0e-6);
            d0 = std::clamp(0.98 * rApo, 1.01 * rContact, d0);
            v0Sq = std::max(0.0, vContactPsu * vContactPsu -
                                 2.0 * units::G_PSU * mTotPsu * (1.0 / rContact - 1.0 / d0));
        }
        const double v0 = std::sqrt(v0Sq);
        const double b = std::clamp(config.impactParamB, -1.2, 1.2);
        double vPerp0 = (rContact * vContactPsu * b) / d0;
        if (std::abs(vPerp0) > 0.995 * v0) {
            vPerp0 = (vPerp0 >= 0.0 ? 1.0 : -1.0) * 0.995 * v0;
        }
        const double vRad0 = -std::sqrt(std::max(0.0, v0 * v0 - vPerp0 * vPerp0));

        // Orbit lies in equatorial XZ plane (matching +Y spin axis and default Y=0 slice plane)
        const double fT = -mIPsu / mTotPsu;
        const double fI =  mTPsu / mTotPsu;

        for (uint32_t i = 0; i < targetN; ++i) {
            targetHost.posX[i] = static_cast<float>(static_cast<double>(targetHost.posX[i]) + fT * d0);
            targetHost.velX[i] = static_cast<float>(static_cast<double>(targetHost.velX[i]) + fT * vRad0);
            targetHost.velZ[i] = static_cast<float>(static_cast<double>(targetHost.velZ[i]) + fT * (-vPerp0));
        }
        for (uint32_t i = 0; i < impactorN; ++i) {
            impactorHost.posX[i] = static_cast<float>(static_cast<double>(impactorHost.posX[i]) + fI * d0);
            impactorHost.velX[i] = static_cast<float>(static_cast<double>(impactorHost.velX[i]) + fI * vRad0);
            impactorHost.velZ[i] = static_cast<float>(static_cast<double>(impactorHost.velZ[i]) + fI * (-vPerp0));
        }

        impl_->qRatioQrd = computeQRatioQrd(
            targetHost.actualMassKg,
            impactorHost.actualMassKg,
            targetHost.equatorialRadiusM,
            impactorHost.equatorialRadiusM,
            vContactPsu * units::V_UNIT_MS,
            b);

        combined = Impl::mergeTwoBodies(targetHost, impactorHost);
    } else {
        impl_->impactorRadiusPsu = 0.0;
        impl_->qRatioQrd = 0.0;
        combined = Impl::makeCombinedFromSingleBody(targetHost);
    }

    impl_->initialPeakTempK = 0.0;
    for (float t : combined.tempK) {
        impl_->initialPeakTempK = std::max(impl_->initialPeakTempK, static_cast<double>(t));
    }

    impl_->uploadHostStateToDevice(combined);
    impl_->evaluateInitialForcesOnDevice();
    impl_->prevTotalRadPowerW = impl_->sumDeviceRadPowerW();
    impl_->computeTelemetryFromDevice();
}

void Simulation::relaxBody(uint8_t bodyId, uint32_t steps, double& outFinalVrmsOverVesc) {
    const uint32_t nAll = impl_->p.count;
    if (nAll == 0u) {
        throw std::runtime_error("Simulation::relaxBody called before loadScenario");
    }

    impl_->downloadTelemetryFieldsFromDevice();

    // Extract current full state from device so single or multi-body setups are handled cleanly
    const size_t fBytes = static_cast<size_t>(nAll) * sizeof(float);
    const size_t i32Bytes = static_cast<size_t>(nAll) * sizeof(int32_t);
    const size_t u8Bytes = static_cast<size_t>(nAll) * sizeof(uint8_t);

    std::vector<float> h0All(nAll), epsAll(nAll);
    std::vector<int32_t> trIdxAll(nAll);
    std::vector<float> trRadAll(nAll), trNxAll(nAll), trNyAll(nAll), trNzAll(nAll);
    std::vector<uint8_t> trOceanAll(nAll);

    ASTRAEA_CUDA_CHECK(cudaMemcpy(h0All.data(), impl_->p.h0, fBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(epsAll.data(), impl_->p.epsFail, fBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(trIdxAll.data(), impl_->p.tracerIndex, i32Bytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(trRadAll.data(), impl_->p.tracerRestRadiusPsu, fBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(trNxAll.data(), impl_->p.tracerRestNormalX, fBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(trNyAll.data(), impl_->p.tracerRestNormalY, fBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(trNzAll.data(), impl_->p.tracerRestNormalZ, fBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(trOceanAll.data(), impl_->p.tracerIsOcean, u8Bytes, cudaMemcpyDeviceToHost));

    std::vector<uint32_t> targetIndices;
    std::vector<uint32_t> otherIndices;
    targetIndices.reserve(nAll);
    for (uint32_t i = 0; i < nAll; ++i) {
        if (impl_->host.bodyId[i] == bodyId) {
            targetIndices.push_back(i);
        } else {
            otherIndices.push_back(i);
        }
    }
    if (targetIndices.empty()) {
        throw std::invalid_argument("Simulation::relaxBody: no particles for requested bodyId");
    }

    auto extractSubset = [&](const std::vector<uint32_t>& idxs) {
        CombinedHostState s;
        const uint32_t m = static_cast<uint32_t>(idxs.size());
        s.count = m;
        s.posX.resize(m); s.posY.resize(m); s.posZ.resize(m);
        s.velX.resize(m); s.velY.resize(m); s.velZ.resize(m);
        s.mass.resize(m); s.h.resize(m); s.h0.resize(m);
        s.rho.resize(m); s.u.resize(m); s.tempK.resize(m); s.epsFail.resize(m);
        s.sXX.assign(m, 0.0f); s.sYY.assign(m, 0.0f); s.sZZ.assign(m, 0.0f);
        s.sXY.assign(m, 0.0f); s.sXZ.assign(m, 0.0f); s.sYZ.assign(m, 0.0f);
        s.damage.assign(m, 0.0f); s.damageCbrt.assign(m, 0.0f);
        s.materialId.resize(m); s.bodyId.resize(m); s.originalId.resize(m);
        s.tracerIndex.resize(m); s.tracerRestRadiusPsu.resize(m);
        s.tracerRestNormalX.resize(m); s.tracerRestNormalY.resize(m); s.tracerRestNormalZ.resize(m);
        s.tracerIsOcean.resize(m);

        for (uint32_t k = 0; k < m; ++k) {
            const uint32_t i = idxs[k];
            s.posX[k] = impl_->host.posX[i];
            s.posY[k] = impl_->host.posY[i];
            s.posZ[k] = impl_->host.posZ[i];
            s.velX[k] = impl_->host.velX[i];
            s.velY[k] = impl_->host.velY[i];
            s.velZ[k] = impl_->host.velZ[i];
            s.mass[k] = impl_->host.mass[i];
            s.h[k] = h0All[i];
            s.h0[k] = h0All[i];
            s.rho[k] = impl_->host.rho[i];
            s.u[k] = impl_->host.u[i];
            s.tempK[k] = impl_->host.tempK[i];
            s.epsFail[k] = epsAll[i];
            s.materialId[k] = impl_->host.materialId[i];
            s.bodyId[k] = impl_->host.bodyId[i];
            s.originalId[k] = k;
            s.tracerIndex[k] = trIdxAll[i];
            s.tracerRestRadiusPsu[k] = trRadAll[i];
            s.tracerRestNormalX[k] = trNxAll[i];
            s.tracerRestNormalY[k] = trNyAll[i];
            s.tracerRestNormalZ[k] = trNzAll[i];
            s.tracerIsOcean[k] = trOceanAll[i];
        }
        return s;
    };

    CombinedHostState bodyState = extractSubset(targetIndices);
    const uint32_t mTarget = bodyState.count;

    // Record original COM & bulk velocity to restore if other body exists
    double mSum = 0.0, origCx = 0.0, origCy = 0.0, origCz = 0.0;
    double origVx = 0.0, origVy = 0.0, origVz = 0.0;
    for (uint32_t k = 0; k < mTarget; ++k) {
        const double m = static_cast<double>(bodyState.mass[k]);
        mSum += m;
        origCx += m * static_cast<double>(bodyState.posX[k]);
        origCy += m * static_cast<double>(bodyState.posY[k]);
        origCz += m * static_cast<double>(bodyState.posZ[k]);
        origVx += m * static_cast<double>(bodyState.velX[k]);
        origVy += m * static_cast<double>(bodyState.velY[k]);
        origVz += m * static_cast<double>(bodyState.velZ[k]);
    }
    if (mSum > 0.0) {
        origCx /= mSum; origCy /= mSum; origCz /= mSum;
        origVx /= mSum; origVy /= mSum; origVz /= mSum;
    }

    std::vector<double> anchorTemps(mTarget);
    for (uint32_t k = 0; k < mTarget; ++k) {
        bodyState.posX[k] = static_cast<float>(static_cast<double>(bodyState.posX[k]) - origCx);
        bodyState.posY[k] = static_cast<float>(static_cast<double>(bodyState.posY[k]) - origCy);
        bodyState.posZ[k] = static_cast<float>(static_cast<double>(bodyState.posZ[k]) - origCz);
        if (steps > 0u) {
            bodyState.velX[k] = 0.0f;
            bodyState.velY[k] = 0.0f;
            bodyState.velZ[k] = 0.0f;
        }
        anchorTemps[k] = static_cast<double>(bodyState.tempK[k]);
    }

    const double nomRadiusPsu = (bodyId == 0u) ? impl_->targetRadiusPsu : impl_->impactorRadiusPsu;
    impl_->uploadHostStateToDevice(bodyState);
    impl_->relaxLoadedSingleBody(anchorTemps, nomRadiusPsu, steps, outFinalVrmsOverVesc);

    // Refresh tracers to match relaxed equilibrium geometry
    const size_t mfBytes = static_cast<size_t>(mTarget) * sizeof(float);
    const size_t mu32Bytes = static_cast<size_t>(mTarget) * sizeof(uint32_t);
    const size_t mi32Bytes = static_cast<size_t>(mTarget) * sizeof(int32_t);
    const size_t mu8Bytes = static_cast<size_t>(mTarget) * sizeof(uint8_t);

    std::vector<float> rPosX(mTarget), rPosY(mTarget), rPosZ(mTarget);
    std::vector<float> rVelX(mTarget), rVelY(mTarget), rVelZ(mTarget);
    std::vector<float> rMass(mTarget), rH(mTarget), rH0(mTarget), rRho(mTarget), rU(mTarget), rTemp(mTarget), rEps(mTarget);
    std::vector<uint8_t> rMat(mTarget), rBody(mTarget), rTrOcean(mTarget);
    std::vector<uint32_t> rOrig(mTarget);
    std::vector<int32_t> rTrIdx(mTarget);

    ASTRAEA_CUDA_CHECK(cudaMemcpy(rPosX.data(), impl_->p.posX, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rPosY.data(), impl_->p.posY, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rPosZ.data(), impl_->p.posZ, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rVelX.data(), impl_->p.velX, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rVelY.data(), impl_->p.velY, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rVelZ.data(), impl_->p.velZ, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rMass.data(), impl_->p.mass, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rH.data(), impl_->p.h, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rH0.data(), impl_->p.h0, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rRho.data(), impl_->p.rho, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rU.data(), impl_->p.u, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rTemp.data(), impl_->p.tempK, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rEps.data(), impl_->p.epsFail, mfBytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rMat.data(), impl_->p.materialId, mu8Bytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rBody.data(), impl_->p.bodyId, mu8Bytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rOrig.data(), impl_->p.originalId, mu32Bytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rTrIdx.data(), impl_->p.tracerIndex, mi32Bytes, cudaMemcpyDeviceToHost));
    ASTRAEA_CUDA_CHECK(cudaMemcpy(rTrOcean.data(), impl_->p.tracerIsOcean, mu8Bytes, cudaMemcpyDeviceToHost));

    double rMaxPsu = 0.0;
    bodyState.tracerRestRadiusPsu.assign(mTarget, 0.0f);
    bodyState.tracerRestNormalX.assign(mTarget, 0.0f);
    bodyState.tracerRestNormalY.assign(mTarget, 0.0f);
    bodyState.tracerRestNormalZ.assign(mTarget, 0.0f);

    for (uint32_t i = 0; i < mTarget; ++i) {
        const double r = std::sqrt(
            static_cast<double>(rPosX[i]) * rPosX[i] +
            static_cast<double>(rPosY[i]) * rPosY[i] +
            static_cast<double>(rPosZ[i]) * rPosZ[i]);
        rMaxPsu = std::max(rMaxPsu, r);

        bodyState.posX[i] = static_cast<float>(static_cast<double>(rPosX[i]) + (otherIndices.empty() ? 0.0 : origCx));
        bodyState.posY[i] = static_cast<float>(static_cast<double>(rPosY[i]) + (otherIndices.empty() ? 0.0 : origCy));
        bodyState.posZ[i] = static_cast<float>(static_cast<double>(rPosZ[i]) + (otherIndices.empty() ? 0.0 : origCz));
        bodyState.velX[i] = static_cast<float>(static_cast<double>(rVelX[i]) + (otherIndices.empty() ? 0.0 : origVx));
        bodyState.velY[i] = static_cast<float>(static_cast<double>(rVelY[i]) + (otherIndices.empty() ? 0.0 : origVy));
        bodyState.velZ[i] = static_cast<float>(static_cast<double>(rVelZ[i]) + (otherIndices.empty() ? 0.0 : origVz));
        bodyState.mass[i] = rMass[i];
        bodyState.h[i] = rH[i];
        bodyState.h0[i] = rH0[i];
        bodyState.rho[i] = rRho[i];
        bodyState.u[i] = rU[i];
        bodyState.tempK[i] = rTemp[i];
        bodyState.epsFail[i] = rEps[i];
        bodyState.materialId[i] = rMat[i];
        bodyState.bodyId[i] = rBody[i];
        bodyState.originalId[i] = i;
        bodyState.tracerIndex[i] = rTrIdx[i];
        bodyState.tracerIsOcean[i] = rTrOcean[i];
        if (rTrIdx[i] >= 0) {
            bodyState.tracerRestRadiusPsu[i] = static_cast<float>(r);
            if (r > 1.0e-9) {
                const double invR = 1.0 / r;
                bodyState.tracerRestNormalX[i] = static_cast<float>(rPosX[i] * invR);
                bodyState.tracerRestNormalY[i] = static_cast<float>(rPosY[i] * invR);
                bodyState.tracerRestNormalZ[i] = static_cast<float>(rPosZ[i] * invR);
            }
        }
    }

    if (rMaxPsu > 0.0) {
        if (bodyId == 0u) {
            impl_->targetRadiusPsu = rMaxPsu;
        } else {
            impl_->impactorRadiusPsu = rMaxPsu;
        }
    }

    if (!otherIndices.empty()) {
        CombinedHostState otherState = extractSubset(otherIndices);
        CombinedHostState merged = bodyState;
        merged.count = mTarget + otherState.count;
        auto appendF = [](std::vector<float>& dst, const std::vector<float>& src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };
        appendF(merged.posX, otherState.posX); appendF(merged.posY, otherState.posY); appendF(merged.posZ, otherState.posZ);
        appendF(merged.velX, otherState.velX); appendF(merged.velY, otherState.velY); appendF(merged.velZ, otherState.velZ);
        appendF(merged.mass, otherState.mass); appendF(merged.h, otherState.h); appendF(merged.h0, otherState.h0);
        appendF(merged.rho, otherState.rho); appendF(merged.u, otherState.u); appendF(merged.tempK, otherState.tempK);
        appendF(merged.epsFail, otherState.epsFail);
        merged.sXX.assign(merged.count, 0.0f); merged.sYY.assign(merged.count, 0.0f); merged.sZZ.assign(merged.count, 0.0f);
        merged.sXY.assign(merged.count, 0.0f); merged.sXZ.assign(merged.count, 0.0f); merged.sYZ.assign(merged.count, 0.0f);
        merged.damage.assign(merged.count, 0.0f); merged.damageCbrt.assign(merged.count, 0.0f);
        merged.materialId.insert(merged.materialId.end(), otherState.materialId.begin(), otherState.materialId.end());
        merged.bodyId.insert(merged.bodyId.end(), otherState.bodyId.begin(), otherState.bodyId.end());
        merged.originalId.resize(merged.count);
        std::iota(merged.originalId.begin(), merged.originalId.end(), 0u);
        merged.tracerIndex.insert(merged.tracerIndex.end(), otherState.tracerIndex.begin(), otherState.tracerIndex.end());
        appendF(merged.tracerRestRadiusPsu, otherState.tracerRestRadiusPsu);
        appendF(merged.tracerRestNormalX, otherState.tracerRestNormalX);
        appendF(merged.tracerRestNormalY, otherState.tracerRestNormalY);
        appendF(merged.tracerRestNormalZ, otherState.tracerRestNormalZ);
        merged.tracerIsOcean.insert(merged.tracerIsOcean.end(), otherState.tracerIsOcean.begin(), otherState.tracerIsOcean.end());
        impl_->uploadHostStateToDevice(merged);
    } else {
        impl_->uploadHostStateToDevice(bodyState);
    }

    impl_->evaluateInitialForcesOnDevice();
    impl_->frameSeq = 0;
    impl_->simTimeSec = 0.0;
    impl_->lastDtSec = 0.0f;
    impl_->prevDtPsu = 0.0f;
    impl_->radiatedEnergyJ = 0.0;
    impl_->prevTotalRadPowerW = impl_->sumDeviceRadPowerW();
    impl_->hasImpacted = false;
    impl_->history.clear();
    impl_->computeTelemetryFromDevice();
}

TelemetrySnapshot Simulation::step(double maxDtSec) {
    if (impl_->p.count == 0u) {
        throw std::runtime_error("Simulation::step called before loadScenario");
    }

    const double effMaxDtSec = std::max(1.0e-5, maxDtSec * impl_->timeScale);
    const float maxDtPsu = static_cast<float>(effMaxDtSec / units::T_UNIT_S);

    ASTRAEA_CUDA_CHECK(cudaEventRecord(impl_->evStart));

    const float dtPsu = cuda::launchReduceDtPsu(impl_->p, impl_->prevDtPsu, maxDtPsu);
    impl_->prevDtPsu = dtPsu;

    cuda::launchLeapfrogKickHalf(impl_->p, 0.5f * dtPsu, true, false);
    cuda::launchLeapfrogDrift(impl_->p, dtPsu, impl_->bvh);
    cuda::launchSortAndBuildBvh(impl_->p, impl_->alt, impl_->bvh);
    cuda::launchSphPass1Neighbors(impl_->p, impl_->bvh);
    cuda::launchEvaluateEos(impl_->p);
    cuda::launchGravityBarnesHut(impl_->p, impl_->bvh, 0.5f);
    cuda::launchSphPass2Forces(impl_->p, impl_->bvh);
    cuda::launchLeapfrogKickHalf(impl_->p, 0.5f * dtPsu, false, false);
    cuda::launchEvaluateEos(impl_->p);

    ASTRAEA_CUDA_CHECK(cudaEventRecord(impl_->evStop));
    ASTRAEA_CUDA_CHECK(cudaEventSynchronize(impl_->evStop));

    float gpuMs = 0.0f;
    ASTRAEA_CUDA_CHECK(cudaEventElapsedTime(&gpuMs, impl_->evStart, impl_->evStop));
    impl_->lastGpuStepMs = static_cast<double>(gpuMs);

    const double dtSec = static_cast<double>(dtPsu) * units::T_UNIT_S;
    const double currRadPowerW = impl_->sumDeviceRadPowerW();
    impl_->radiatedEnergyJ += 0.5 * (impl_->prevTotalRadPowerW + currRadPowerW) * dtSec;
    impl_->prevTotalRadPowerW = currRadPowerW;

    impl_->simTimeSec += dtSec;
    impl_->lastDtSec = static_cast<float>(dtSec);
    impl_->frameSeq += 1u;

    TelemetrySnapshot snap = impl_->computeTelemetryFromDevice();
    impl_->history.push_back(snap);
    return snap;
}

TelemetrySnapshot Simulation::stepForDuration(double durationSec, double maxDtSec) {
    if (impl_->p.count == 0u) {
        throw std::runtime_error("Simulation::stepForDuration called before loadScenario");
    }
    if (durationSec <= 0.0) {
        return computeTelemetry();
    }

    const double targetEndTime = impl_->simTimeSec + durationSec;
    TelemetrySnapshot snap = impl_->lastSnapshot;
    while (impl_->simTimeSec < targetEndTime - 1.0e-9) {
        const double remaining = targetEndTime - impl_->simTimeSec;
        const double stepMaxDt = std::min(maxDtSec, std::max(1.0e-5, remaining / impl_->timeScale));
        snap = step(stepMaxDt);
    }
    return snap;
}

void Simulation::setTimeScale(double scale) {
    if (!std::isfinite(scale)) {
        return;
    }
    impl_->timeScale = std::clamp(scale, 0.1, 100.0);
}

void Simulation::setSlicePlane(const SlicePlaneConfig& slice) {
    SlicePlaneConfig normSlice = slice;
    const double len = std::sqrt(
        static_cast<double>(slice.normalX) * slice.normalX +
        static_cast<double>(slice.normalY) * slice.normalY +
        static_cast<double>(slice.normalZ) * slice.normalZ);
    if (len > 1.0e-9) {
        normSlice.normalX = static_cast<float>(slice.normalX / len);
        normSlice.normalY = static_cast<float>(slice.normalY / len);
        normSlice.normalZ = static_cast<float>(slice.normalZ / len);
    } else {
        normSlice.normalX = 0.0f;
        normSlice.normalY = 1.0f;
        normSlice.normalZ = 0.0f;
    }
    impl_->slicePlane = normSlice;
}

TelemetrySnapshot Simulation::computeTelemetry() {
    if (impl_->p.count == 0u) {
        return impl_->lastSnapshot;
    }
    return impl_->computeTelemetryFromDevice();
}

void Simulation::extractRenderFrame(
    std::vector<uint8_t>& outParticleBytes,
    std::vector<uint8_t>& outTracerBytes,
    std::vector<uint8_t>* outSpeedBytes) {
    const uint32_t n = impl_->p.count;
    const uint32_t t = impl_->tracerCount;

    if (n == 0u) {
        outParticleBytes.clear();
        outTracerBytes.clear();
        if (outSpeedBytes != nullptr) {
            outSpeedBytes->clear();
        }
        return;
    }

    impl_->ensureCapacity(n);
    outParticleBytes.resize(static_cast<size_t>(n) * 20u);

    uint8_t* dSpeed = nullptr;
    if (outSpeedBytes != nullptr) {
        outSpeedBytes->resize(n);
        dSpeed = impl_->d_renderSpeed;
    }

    cuda::launchPackRenderRecords(
        impl_->p,
        impl_->d_renderParticles,
        dSpeed,
        static_cast<float>(impl_->sysComPsu[0]),
        static_cast<float>(impl_->sysComPsu[1]),
        static_cast<float>(impl_->sysComPsu[2]));

    if (t > 0u) {
        impl_->ensureTracerCapacity(t);
        outTracerBytes.resize(static_cast<size_t>(t) * 40u);
        cuda::launchExtractTracers(
            impl_->p,
            impl_->d_renderTracers,
            t,
            impl_->tracerGridRes,
            static_cast<float>(impl_->livingComPsu[0]),
            static_cast<float>(impl_->livingComPsu[1]),
            static_cast<float>(impl_->livingComPsu[2]),
            static_cast<float>(impl_->sysComPsu[0]),
            static_cast<float>(impl_->sysComPsu[1]),
            static_cast<float>(impl_->sysComPsu[2]));
    } else {
        outTracerBytes.clear();
    }

    ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(
        outParticleBytes.data(), impl_->d_renderParticles, outParticleBytes.size(), cudaMemcpyDeviceToHost));
    if (t > 0u) {
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(
            outTracerBytes.data(), impl_->d_renderTracers, outTracerBytes.size(), cudaMemcpyDeviceToHost));
    }
    if (outSpeedBytes != nullptr) {
        ASTRAEA_CUDA_CHECK(cudaMemcpyAsync(
            outSpeedBytes->data(), impl_->d_renderSpeed, outSpeedBytes->size(), cudaMemcpyDeviceToHost));
    }
    ASTRAEA_CUDA_CHECK(cudaDeviceSynchronize());
}

std::string Simulation::exportTelemetryCsv() const {
    std::ostringstream os;
    os << "frameSeq,simTimeSec,dtSec,outcome,"
       << "kineticEnergyJ,potentialEnergyJ,internalEnergyJ,radiatedEnergyJ,totalEnergyJ,"
       << "initialTotalEnergyJ,energyDriftRel,"
       << "momentumKgMS_x,momentumKgMS_y,momentumKgMS_z,"
       << "angularMomentumKgM2S_x,angularMomentumKgM2S_y,angularMomentumKgM2S_z,"
       << "virialRatio,maxTempK,maxPressureGPa,"
       << "boundMassFrac,diskMassFrac,ejectaMassFrac,qRatioQrd,gpuStepMs,"
       << "livingBodyComKm_x,livingBodyComKm_y,livingBodyComKm_z,"
       << "livingBodyRadiusKm,livingBodySpinRadS,"
       << "impactorComKm_x,impactorComKm_y,impactorComKm_z,"
       << "particleCount,tracerCount,tracerGridRes,livingBodyPresent\n";

    if (impl_->history.empty()) {
        writeSnapshotCsvRow(os, impl_->lastSnapshot);
    } else {
        for (const TelemetrySnapshot& s : impl_->history) {
            writeSnapshotCsvRow(os, s);
        }
    }
    return os.str();
}

std::string Simulation::exportTelemetryJson() const {
    std::ostringstream os;
    const TelemetrySnapshot& latest = impl_->history.empty() ? impl_->lastSnapshot : impl_->history.back();

    os << "{";
    os << "\"preset\":\"" << jsonEscape(impl_->config.presetName) << "\",";
    writeSnapshotJsonFields(os, latest);
    os << ",\"history\":[";
    if (impl_->history.empty()) {
        os << "{";
        writeSnapshotJsonFields(os, impl_->lastSnapshot);
        os << "}";
    } else {
        for (size_t i = 0; i < impl_->history.size(); ++i) {
            if (i > 0) {
                os << ",";
            }
            os << "{";
            writeSnapshotJsonFields(os, impl_->history[i]);
            os << "}";
        }
    }
    os << "]}";
    return os.str();
}

const std::vector<TelemetrySnapshot>& Simulation::telemetryHistory() const {
    return impl_->history;
}

const CollisionScenarioConfig& Simulation::activeConfig() const {
    return impl_->config;
}

const SlicePlaneConfig& Simulation::slicePlane() const {
    return impl_->slicePlane;
}

double Simulation::timeScale() const {
    return impl_->timeScale;
}

} // namespace astraea
