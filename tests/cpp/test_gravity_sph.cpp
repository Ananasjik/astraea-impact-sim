#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <vector>

#include "eos.hpp"
#include "planet_builder.hpp"
#include "presets.hpp"
#include "simulation.hpp"
#include "sph_gravity.cuh"
#include "types.hpp"
#include "units.hpp"

namespace {

using namespace astraea;

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err__ = (call);                                                                \
        ASSERT_EQ(err__, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err__);              \
    } while (0)

template <typename T>
T* cudaAllocZero(size_t count) {
    T* ptr = nullptr;
    if (cudaMalloc(&ptr, count * sizeof(T)) != cudaSuccess) {
        return nullptr;
    }
    if (cudaMemset(ptr, 0, count * sizeof(T)) != cudaSuccess) {
        cudaFree(ptr);
        return nullptr;
    }
    return ptr;
}

class DeviceParticlesGuard {
public:
    ParticleBuffersDevice dev{};

    explicit DeviceParticlesGuard(uint32_t capacity) {
        dev.capacity = capacity;
        dev.count = 0;
        const size_t n = capacity;
        dev.posX = cudaAllocZero<float>(n);
        dev.posY = cudaAllocZero<float>(n);
        dev.posZ = cudaAllocZero<float>(n);
        dev.velX = cudaAllocZero<float>(n);
        dev.velY = cudaAllocZero<float>(n);
        dev.velZ = cudaAllocZero<float>(n);
        dev.velHalfX = cudaAllocZero<float>(n);
        dev.velHalfY = cudaAllocZero<float>(n);
        dev.velHalfZ = cudaAllocZero<float>(n);
        dev.accX = cudaAllocZero<float>(n);
        dev.accY = cudaAllocZero<float>(n);
        dev.accZ = cudaAllocZero<float>(n);
        dev.mass = cudaAllocZero<float>(n);
        dev.h = cudaAllocZero<float>(n);
        dev.h0 = cudaAllocZero<float>(n);
        dev.rho = cudaAllocZero<float>(n);
        dev.drhoDt = cudaAllocZero<float>(n);
        dev.pressure = cudaAllocZero<float>(n);
        dev.soundSpeed = cudaAllocZero<float>(n);
        dev.u = cudaAllocZero<float>(n);
        dev.duDt = cudaAllocZero<float>(n);
        dev.tempK = cudaAllocZero<float>(n);
        dev.radPowerW = cudaAllocZero<float>(n);

        dev.sXX = cudaAllocZero<float>(n);
        dev.sYY = cudaAllocZero<float>(n);
        dev.sZZ = cudaAllocZero<float>(n);
        dev.sXY = cudaAllocZero<float>(n);
        dev.sXZ = cudaAllocZero<float>(n);
        dev.sYZ = cudaAllocZero<float>(n);
        dev.dSXX = cudaAllocZero<float>(n);
        dev.dSYY = cudaAllocZero<float>(n);
        dev.dSZZ = cudaAllocZero<float>(n);
        dev.dSXY = cudaAllocZero<float>(n);
        dev.dSXZ = cudaAllocZero<float>(n);
        dev.dSYZ = cudaAllocZero<float>(n);

        dev.damage = cudaAllocZero<float>(n);
        dev.damageCbrt = cudaAllocZero<float>(n);
        dev.dDamageCbrtDt = cudaAllocZero<float>(n);
        dev.epsFail = cudaAllocZero<float>(n);
        dev.divV = cudaAllocZero<float>(n);
        dev.curlVX = cudaAllocZero<float>(n);
        dev.curlVY = cudaAllocZero<float>(n);
        dev.curlVZ = cudaAllocZero<float>(n);
        dev.shepardC = cudaAllocZero<float>(n);
        dev.surfFactor = cudaAllocZero<float>(n);
        dev.phiGrav = cudaAllocZero<float>(n);

        dev.materialId = cudaAllocZero<uint8_t>(n);
        dev.phaseId = cudaAllocZero<uint8_t>(n);
        dev.bodyId = cudaAllocZero<uint8_t>(n);
        dev.originalId = cudaAllocZero<uint32_t>(n);
        dev.tracerIndex = cudaAllocZero<int32_t>(n);
        if (dev.tracerIndex) {
            cudaMemset(dev.tracerIndex, 0xFF, n * sizeof(int32_t));
        }
        dev.tracerRestRadiusPsu = cudaAllocZero<float>(n);
        dev.tracerRestNormalX = cudaAllocZero<float>(n);
        dev.tracerRestNormalY = cudaAllocZero<float>(n);
        dev.tracerRestNormalZ = cudaAllocZero<float>(n);
        dev.tracerIsOcean = cudaAllocZero<uint8_t>(n);
    }

    ~DeviceParticlesGuard() {
        cudaFree(dev.posX);
        cudaFree(dev.posY);
        cudaFree(dev.posZ);
        cudaFree(dev.velX);
        cudaFree(dev.velY);
        cudaFree(dev.velZ);
        cudaFree(dev.velHalfX);
        cudaFree(dev.velHalfY);
        cudaFree(dev.velHalfZ);
        cudaFree(dev.accX);
        cudaFree(dev.accY);
        cudaFree(dev.accZ);
        cudaFree(dev.mass);
        cudaFree(dev.h);
        cudaFree(dev.h0);
        cudaFree(dev.rho);
        cudaFree(dev.drhoDt);
        cudaFree(dev.pressure);
        cudaFree(dev.soundSpeed);
        cudaFree(dev.u);
        cudaFree(dev.duDt);
        cudaFree(dev.tempK);
        cudaFree(dev.radPowerW);
        cudaFree(dev.sXX);
        cudaFree(dev.sYY);
        cudaFree(dev.sZZ);
        cudaFree(dev.sXY);
        cudaFree(dev.sXZ);
        cudaFree(dev.sYZ);
        cudaFree(dev.dSXX);
        cudaFree(dev.dSYY);
        cudaFree(dev.dSZZ);
        cudaFree(dev.dSXY);
        cudaFree(dev.dSXZ);
        cudaFree(dev.dSYZ);
        cudaFree(dev.damage);
        cudaFree(dev.damageCbrt);
        cudaFree(dev.dDamageCbrtDt);
        cudaFree(dev.epsFail);
        cudaFree(dev.divV);
        cudaFree(dev.curlVX);
        cudaFree(dev.curlVY);
        cudaFree(dev.curlVZ);
        cudaFree(dev.shepardC);
        cudaFree(dev.surfFactor);
        cudaFree(dev.phiGrav);
        cudaFree(dev.materialId);
        cudaFree(dev.phaseId);
        cudaFree(dev.bodyId);
        cudaFree(dev.originalId);
        cudaFree(dev.tracerIndex);
        cudaFree(dev.tracerRestRadiusPsu);
        cudaFree(dev.tracerRestNormalX);
        cudaFree(dev.tracerRestNormalY);
        cudaFree(dev.tracerRestNormalZ);
        cudaFree(dev.tracerIsOcean);
    }

    DeviceParticlesGuard(const DeviceParticlesGuard&) = delete;
    DeviceParticlesGuard& operator=(const DeviceParticlesGuard&) = delete;
};

class DeviceBvhGuard {
public:
    cuda::RadixBvhDevice bvh{};

    explicit DeviceBvhGuard(uint32_t capacity) {
        bvh.capacity = capacity;
        const size_t n = capacity;
        const size_t nodes = 2 * n;
        bvh.mortonCodes = cudaAllocZero<uint64_t>(n);
        bvh.mortonCodesSorted = cudaAllocZero<uint64_t>(n);
        bvh.indicesIn = cudaAllocZero<uint32_t>(n);
        bvh.indicesSorted = cudaAllocZero<uint32_t>(n);
        bvh.leftChild = cudaAllocZero<int32_t>(n);
        bvh.rightChild = cudaAllocZero<int32_t>(n);
        bvh.parent = cudaAllocZero<int32_t>(nodes);
        bvh.atomicVisited = cudaAllocZero<uint32_t>(n);
        bvh.aabbMinX = cudaAllocZero<float>(nodes);
        bvh.aabbMinY = cudaAllocZero<float>(nodes);
        bvh.aabbMinZ = cudaAllocZero<float>(nodes);
        bvh.aabbMaxX = cudaAllocZero<float>(nodes);
        bvh.aabbMaxY = cudaAllocZero<float>(nodes);
        bvh.aabbMaxZ = cudaAllocZero<float>(nodes);
        bvh.nodeMass = cudaAllocZero<float>(nodes);
        bvh.comX = cudaAllocZero<float>(nodes);
        bvh.comY = cudaAllocZero<float>(nodes);
        bvh.comZ = cudaAllocZero<float>(nodes);
        bvh.nodeMaxH = cudaAllocZero<float>(nodes);
        bvh.cubTempStorage = nullptr;
        bvh.cubTempBytes = 0;
    }

    ~DeviceBvhGuard() {
        cudaFree(bvh.mortonCodes);
        cudaFree(bvh.mortonCodesSorted);
        cudaFree(bvh.indicesIn);
        cudaFree(bvh.indicesSorted);
        cudaFree(bvh.leftChild);
        cudaFree(bvh.rightChild);
        cudaFree(bvh.parent);
        cudaFree(bvh.atomicVisited);
        cudaFree(bvh.aabbMinX);
        cudaFree(bvh.aabbMinY);
        cudaFree(bvh.aabbMinZ);
        cudaFree(bvh.aabbMaxX);
        cudaFree(bvh.aabbMaxY);
        cudaFree(bvh.aabbMaxZ);
        cudaFree(bvh.nodeMass);
        cudaFree(bvh.comX);
        cudaFree(bvh.comY);
        cudaFree(bvh.comZ);
        cudaFree(bvh.nodeMaxH);
        if (bvh.cubTempStorage) {
            cudaFree(bvh.cubTempStorage);
        }
    }

    DeviceBvhGuard(const DeviceBvhGuard&) = delete;
    DeviceBvhGuard& operator=(const DeviceBvhGuard&) = delete;
};

struct DecodedParticleRecord {
    float xKm, yKm, zKm;
    float tempK;
    float pressureGPa;
    uint8_t materialId;
    uint8_t phaseId;
    uint8_t bodyId;
    float damage;
    float radiusKm;
    float speedKmS;
};

float readF32(const uint8_t* p) {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(float));
    return v;
}

uint16_t readU16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

std::vector<DecodedParticleRecord> decodeParticles(
    const std::vector<uint8_t>& particleBytes,
    const std::vector<uint8_t>* speedBytes = nullptr) {
    const size_t n = particleBytes.size() / 20;
    std::vector<DecodedParticleRecord> out(n);
    const double ln20000 = std::log(20000.0);
    const double ln1e7 = std::log(1.0e7);
    const double ln2000 = std::log(2000.0);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* rec = particleBytes.data() + 20 * i;
        DecodedParticleRecord d{};
        d.xKm = readF32(rec + 0);
        d.yKm = readF32(rec + 4);
        d.zKm = readF32(rec + 8);
        const uint16_t tU16 = readU16(rec + 12);
        const uint16_t pU16 = readU16(rec + 14);
        d.tempK = static_cast<float>(10.0 * std::exp((static_cast<double>(tU16) / 65535.0) * ln20000));
        d.pressureGPa = (pU16 == 0)
                            ? 0.0f
                            : static_cast<float>(1.0e-3 * std::exp((static_cast<double>(pU16 - 1) / 65534.0) * ln1e7));
        d.materialId = rec[16];
        d.bodyId = rec[17] >> 4;
        d.phaseId = rec[17] & 0x0F;
        d.damage = static_cast<float>(rec[18]) / 255.0f;
        d.radiusKm = static_cast<float>(std::exp((static_cast<double>(rec[19]) / 255.0) * ln2000));
        if (speedBytes && i < speedBytes->size()) {
            d.speedKmS = static_cast<float>(0.05 * std::exp((static_cast<double>((*speedBytes)[i]) / 255.0) * ln2000));
        }
        out[i] = d;
    }
    return out;
}

TEST(GravitySphTest, BarnesHutGravityMatchesExactN2Reference_AC3) {
    const uint32_t nPerBody = 2048;
    const uint32_t totalN = 2 * nPerBody;

    const BodyInitSpec specA{PlanetArchetype::ProtoTheia, 6.417e23, 240.0, 2200.0, 0.0, 0};
    const BodyInitSpec specB{PlanetArchetype::IronCore, 3.3011e23, 440.0, 3200.0, 0.0, 0};
    const BuiltPlanetHost bodyA = PlanetBuilder::buildPlanet(specA, nPerBody, 0, 8, 42);
    const BuiltPlanetHost bodyB = PlanetBuilder::buildPlanet(specB, nPerBody, 1, 8, 43);

    std::vector<float> hPosX(totalN), hPosY(totalN), hPosZ(totalN), hMass(totalN), hH0(totalN);
    std::vector<uint32_t> hOrigId(totalN);
    for (uint32_t i = 0; i < nPerBody; ++i) {
        hPosX[i] = bodyA.posX[i] - 12.0f;
        hPosY[i] = bodyA.posY[i];
        hPosZ[i] = bodyA.posZ[i];
        hMass[i] = bodyA.mass[i];
        hH0[i] = bodyA.h0[i];
        hOrigId[i] = i;

        const uint32_t j = nPerBody + i;
        hPosX[j] = bodyB.posX[i] + 12.0f;
        hPosY[j] = bodyB.posY[i];
        hPosZ[j] = bodyB.posZ[i];
        hMass[j] = bodyB.mass[i];
        hH0[j] = bodyB.h0[i];
        hOrigId[j] = j;
    }

    DeviceParticlesGuard pGuard(totalN);
    DeviceParticlesGuard altGuard(totalN);
    DeviceBvhGuard bvhGuard(totalN);
    pGuard.dev.count = totalN;
    altGuard.dev.count = totalN;

    CUDA_CHECK(cudaMemcpy(pGuard.dev.posX, hPosX.data(), totalN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pGuard.dev.posY, hPosY.data(), totalN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pGuard.dev.posZ, hPosZ.data(), totalN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pGuard.dev.mass, hMass.data(), totalN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pGuard.dev.h0, hH0.data(), totalN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pGuard.dev.h, hH0.data(), totalN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pGuard.dev.originalId, hOrigId.data(), totalN * sizeof(uint32_t), cudaMemcpyHostToDevice));

    cuda::launchSortAndBuildBvh(pGuard.dev, altGuard.dev, bvhGuard.bvh);
    cuda::launchGravityBarnesHut(pGuard.dev, bvhGuard.bvh, 0.5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> sPosX(totalN), sPosY(totalN), sPosZ(totalN), sMass(totalN), sH0(totalN);
    std::vector<float> bhAccX(totalN), bhAccY(totalN), bhAccZ(totalN), phiGrav(totalN);
    CUDA_CHECK(cudaMemcpy(sPosX.data(), pGuard.dev.posX, totalN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sPosY.data(), pGuard.dev.posY, totalN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sPosZ.data(), pGuard.dev.posZ, totalN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sMass.data(), pGuard.dev.mass, totalN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sH0.data(), pGuard.dev.h0, totalN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bhAccX.data(), pGuard.dev.accX, totalN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bhAccY.data(), pGuard.dev.accY, totalN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bhAccZ.data(), pGuard.dev.accZ, totalN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phiGrav.data(), pGuard.dev.phiGrav, totalN * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<double> relErrors;
    relErrors.reserve(totalN);
    for (uint32_t i = 0; i < totalN; ++i) {
        ASSERT_TRUE(std::isfinite(bhAccX[i]));
        ASSERT_TRUE(std::isfinite(bhAccY[i]));
        ASSERT_TRUE(std::isfinite(bhAccZ[i]));
        ASSERT_TRUE(std::isfinite(phiGrav[i]));
        EXPECT_LT(phiGrav[i], 0.0f);

        const double xi = sPosX[i];
        const double yi = sPosY[i];
        const double zi = sPosZ[i];
        const double epsI = 0.8 * static_cast<double>(sH0[i]);

        double ax = 0.0, ay = 0.0, az = 0.0;
        for (uint32_t j = 0; j < totalN; ++j) {
            if (i == j) continue;
            const double dx = xi - static_cast<double>(sPosX[j]);
            const double dy = yi - static_cast<double>(sPosY[j]);
            const double dz = zi - static_cast<double>(sPosZ[j]);
            const double epsIJ = 0.5 * (epsI + 0.8 * static_cast<double>(sH0[j]));
            const double d2 = dx * dx + dy * dy + dz * dz + epsIJ * epsIJ;
            const double invDist3 = 1.0 / (d2 * std::sqrt(d2));
            const double factor = -units::G_PSU * static_cast<double>(sMass[j]) * invDist3;
            ax += factor * dx;
            ay += factor * dy;
            az += factor * dz;
        }
        const double refMag = std::sqrt(ax * ax + ay * ay + az * az);
        if (refMag > 1e-8) {
            const double ex = static_cast<double>(bhAccX[i]) - ax;
            const double ey = static_cast<double>(bhAccY[i]) - ay;
            const double ez = static_cast<double>(bhAccZ[i]) - az;
            relErrors.push_back(std::sqrt(ex * ex + ey * ey + ez * ez) / refMag);
        }
    }

    ASSERT_EQ(relErrors.size(), totalN);
    std::sort(relErrors.begin(), relErrors.end());
    const double medianErr = relErrors[relErrors.size() / 2];
    const double p99Err = relErrors[static_cast<size_t>(0.99 * static_cast<double>(relErrors.size() - 1))];

    EXPECT_LT(medianErr, 0.015);
    EXPECT_LT(p99Err, 0.05);
}

TEST(GravitySphTest, MultiMaterialCmbStability200Steps_AC4) {
    CollisionScenarioConfig cfg;
    cfg.presetName = "CmbStabilityTest";
    cfg.target = {PlanetArchetype::LivingTerra, units::M_EARTH_SI, 288.0, 5500.0, 0.0, 80};
    cfg.impactor = {PlanetArchetype::ProtoTheia, 0.02 * units::M_EARTH_SI, 240.0, 1800.0, 0.0, 80};
    cfg.impactVelocityRatioVesc = 0.0;
    cfg.impactParamB = 0.0;
    cfg.initialSepRadii = 50.0;
    cfg.totalParticles = 10000;
    cfg.tracerGridRes = 8;
    cfg.randomSeed = 42;

    Simulation sim(12000);
    sim.loadScenario(cfg);

    for (int step = 0; step < 200; ++step) {
        sim.step(5.0);
    }

    std::vector<uint8_t> particleBytes, tracerBytes, speedBytes;
    sim.extractRenderFrame(particleBytes, tracerBytes, &speedBytes);
    const auto recs = decodeParticles(particleBytes, &speedBytes);

    double comX = 0.0, comY = 0.0, comZ = 0.0;
    uint32_t targetCount = 0;
    for (const auto& r : recs) {
        if (r.bodyId == 0) {
            comX += r.xKm;
            comY += r.yKm;
            comZ += r.zKm;
            ++targetCount;
        }
    }
    ASSERT_GT(targetCount, 5000u);
    comX /= targetCount;
    comY /= targetCount;
    comZ /= targetCount;

    std::vector<std::pair<double, DecodedParticleRecord>> ironList;
    std::vector<std::pair<double, DecodedParticleRecord>> graniteList;
    double sumV2 = 0.0;
    for (const auto& r : recs) {
        if (r.bodyId != 0) continue;
        const double dx = r.xKm - comX;
        const double dy = r.yKm - comY;
        const double dz = r.zKm - comZ;
        const double rad = std::sqrt(dx * dx + dy * dy + dz * dz);
        sumV2 += static_cast<double>(r.speedKmS) * static_cast<double>(r.speedKmS);
        if (r.materialId == static_cast<uint8_t>(MaterialId::Iron)) {
            ironList.push_back({rad, r});
        } else if (r.materialId == static_cast<uint8_t>(MaterialId::Granite)) {
            graniteList.push_back({rad, r});
        }
    }

    ASSERT_GT(ironList.size(), 100u);
    ASSERT_GT(graniteList.size(), 200u);

    auto byRad = [](const auto& a, const auto& b) { return a.first < b.first; };
    std::sort(ironList.begin(), ironList.end(), byRad);
    std::sort(graniteList.begin(), graniteList.end(), byRad);

    const double ironP95R = ironList[static_cast<size_t>(0.95 * (ironList.size() - 1))].first;
    const double graniteP05R = graniteList[static_cast<size_t>(0.05 * (graniteList.size() - 1))].first;
    const double graniteP95R = graniteList[static_cast<size_t>(0.95 * (graniteList.size() - 1))].first;
    const double meanSpacingKm =
        std::cbrt((4.0 * std::numbers::pi / 3.0) * std::pow(graniteP95R, 3) / static_cast<double>(targetCount));

    // Core stays strictly inside mantle without layer inversion
    EXPECT_LT(ironP95R, graniteP05R + 1.2 * meanSpacingKm);

    // No interface void gap opens at CMB
    const double cmbGapKm = graniteP05R - ironP95R;
    EXPECT_LT(cmbGapKm, 2.5 * meanSpacingKm);

    // Sharp density discontinuity preserved across CMB (Iron/Granite > 1.8)
    const size_t nEdgeIron = std::max<size_t>(10, ironList.size() / 10);
    const size_t nEdgeGranite = std::max<size_t>(10, graniteList.size() / 10);
    double rhoIronSum = 0.0;
    for (size_t i = ironList.size() - nEdgeIron; i < ironList.size(); ++i) {
        const auto& r = ironList[i].second;
        rhoIronSum += eos::invertDensityFromPressureTempSI(
            MaterialId::Iron, static_cast<double>(r.pressureGPa) * 1.0e9, r.tempK, 9000.0);
    }
    double rhoGraniteSum = 0.0;
    for (size_t i = 0; i < nEdgeGranite; ++i) {
        const auto& r = graniteList[i].second;
        rhoGraniteSum += eos::invertDensityFromPressureTempSI(
            MaterialId::Granite, static_cast<double>(r.pressureGPa) * 1.0e9, r.tempK, 4000.0);
    }
    const double rhoIronCmb = rhoIronSum / static_cast<double>(nEdgeIron);
    const double rhoGraniteCmb = rhoGraniteSum / static_cast<double>(nEdgeGranite);
    EXPECT_GT(rhoIronCmb / rhoGraniteCmb, 1.8);

    // Radial / RMS jitter < 0.015 v_esc
    const double vRmsKmS = std::sqrt(sumV2 / static_cast<double>(targetCount));
    const double vEscKmS =
        std::sqrt(2.0 * units::G_SI * units::M_EARTH_SI / units::R_EARTH_SI) / 1000.0;
    EXPECT_LT(vRmsKmS / vEscKmS, 0.015);
}

TEST(GravitySphTest, MercuryMantleStripShockHeatingAndBrittleDamage_AC6) {
    CollisionScenarioConfig cfg = makePresetConfig(ScenarioPresetId::MercuryMantleStrip, 15000);
    cfg.initialSepRadii = 1.30;

    Simulation sim(20000);
    sim.loadScenario(cfg);

    for (int s = 0; s < 100; ++s) {
        sim.step(5.0);
    }

    const TelemetrySnapshot t = sim.computeTelemetry();
    EXPECT_GT(t.maxTempK, 1600.0);

    std::vector<uint8_t> particleBytes, tracerBytes;
    sim.extractRenderFrame(particleBytes, tracerBytes, nullptr);
    const auto recs = decodeParticles(particleBytes);

    uint32_t moltenOrVaporGranite = 0;
    uint32_t highDamageNearCrater = 0;
    uint32_t intactFarField = 0;
    uint32_t targetCount = 0;

    double damagedDistSum = 0.0;
    double intactDistSum = 0.0;

    for (const auto& r : recs) {
        if (r.bodyId != 0) continue;
        ++targetCount;
        if (r.materialId == static_cast<uint8_t>(MaterialId::Granite) &&
            r.tempK > 1600.0f &&
            (r.phaseId == static_cast<uint8_t>(PhaseId::Molten) ||
             r.phaseId == static_cast<uint8_t>(PhaseId::Vapor))) {
            ++moltenOrVaporGranite;
        }
        const double dx = r.xKm - t.impactorComKm[0];
        const double dy = r.yKm - t.impactorComKm[1];
        const double dz = r.zKm - t.impactorComKm[2];
        const double dImp = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (r.damage > 0.8f) {
            ++highDamageNearCrater;
            damagedDistSum += dImp;
        } else if (r.damage < 0.2f) {
            ++intactFarField;
            intactDistSum += dImp;
        }
    }

    ASSERT_GT(targetCount, 5000u);
    EXPECT_GT(moltenOrVaporGranite, 10u);
    EXPECT_GT(highDamageNearCrater, 20u);
    EXPECT_GT(intactFarField, targetCount / 10);

    // Crater damage is spatially localized closer to impactor than intact antipode
    const double meanDamagedDist = damagedDistSum / static_cast<double>(highDamageNearCrater);
    const double meanIntactDist = intactDistSum / static_cast<double>(intactFarField);
    EXPECT_LT(meanDamagedDist, meanIntactDist);
}

TEST(GravitySphTest, RadiativeCoolingIsolatedSurfaceVsInterior_AC7) {
    // Part A: Isolated particles at 4000 K and 2000 K (f_surf -> 1.0, exact sigma_SB * T^4 scaling)
    auto measureIsolatedCoolingWatts = [&](double tempK, float& outSurfFactor, float& outRadPowerW, double& outWatts) {
        outWatts = 0.0;
        const uint32_t n = 2;
        const auto& g = eos::getParams(MaterialId::Granite);
        const float rhoPsu = static_cast<float>(g.rho0KgM3 / units::RHO_UNIT_KGM3);
        const float massPsu = 1.0e-4f;
        const float hPsu = 1.2f * std::cbrt(massPsu / rhoPsu);
        const double uSi = g.cv_JKgK * tempK;
        const float uPsu = static_cast<float>(uSi / units::U_UNIT_JKG);

        std::vector<float> posX = {-25.0f, 25.0f};
        std::vector<float> zero(n, 0.0f);
        std::vector<float> mass(n, massPsu);
        std::vector<float> h(n, hPsu);
        std::vector<float> rho(n, rhoPsu);
        std::vector<float> u(n, uPsu);
        std::vector<float> tK(n, static_cast<float>(tempK));
        std::vector<float> epsFail(n, static_cast<float>(g.epsFailMin));
        std::vector<float> shepard(n, 1.0f);
        std::vector<uint8_t> mat(n, static_cast<uint8_t>(MaterialId::Granite));
        std::vector<uint32_t> origId = {0u, 1u};

        DeviceParticlesGuard pGuard(n);
        DeviceParticlesGuard altGuard(n);
        DeviceBvhGuard bvhGuard(n);
        pGuard.dev.count = n;
        altGuard.dev.count = n;

        CUDA_CHECK(cudaMemcpy(pGuard.dev.posX, posX.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.posY, zero.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.posZ, zero.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.mass, mass.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.h, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.h0, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.rho, rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.u, u.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.tempK, tK.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.epsFail, epsFail.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.shepardC, shepard.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.materialId, mat.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pGuard.dev.originalId, origId.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

        cuda::launchSortAndBuildBvh(pGuard.dev, altGuard.dev, bvhGuard.bvh);
        cuda::launchSphPass1Neighbors(pGuard.dev, bvhGuard.bvh);
        cuda::launchEvaluateEos(pGuard.dev);
        cuda::launchSphPass2Forces(pGuard.dev, bvhGuard.bvh);
        CUDA_CHECK(cudaDeviceSynchronize());

        float duDt0 = 0.0f;
        CUDA_CHECK(cudaMemcpy(&outSurfFactor, pGuard.dev.surfFactor, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&outRadPowerW, pGuard.dev.radPowerW, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&duDt0, pGuard.dev.duDt, sizeof(float), cudaMemcpyDeviceToHost));

        outWatts =
            -static_cast<double>(duDt0) * static_cast<double>(massPsu) * (units::E_UNIT_J / units::T_UNIT_S);
    };

    float surf4000 = 0.0f, rad4000 = 0.0f;
    double coolingW4000 = 0.0;
    measureIsolatedCoolingWatts(4000.0, surf4000, rad4000, coolingW4000);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_GT(surf4000, 0.95f);
    EXPECT_LE(surf4000, 1.001f);
    EXPECT_GT(rad4000, 0.0f);

    const double mSI = 1.0e-4 * units::M_UNIT_KG;
    const double rhoSI = 2680.0;
    const double rpSI = std::cbrt((3.0 * mSI) / (4.0 * std::numbers::pi * rhoSI));
    const double areaSI = std::numbers::pi * rpSI * rpSI;
    const double expectedW4000 = 0.9 * units::SIGMA_SB_SI * areaSI * (std::pow(4000.0, 4) - std::pow(3.0, 4));
    EXPECT_NEAR(coolingW4000, expectedW4000, 0.05 * expectedW4000);

    float surf2000 = 0.0f, rad2000 = 0.0f;
    double coolingW2000 = 0.0;
    measureIsolatedCoolingWatts(2000.0, surf2000, rad2000, coolingW2000);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_GT(surf2000, 0.95f);
    EXPECT_NEAR(coolingW4000 / coolingW2000, 16.0, 0.25);

    // Part B: Hot sphere interior (f_surf == 0, radPowerW == 0) vs exposed surface (f_surf > 0, radPowerW > 0)
    const BodyInitSpec hotSpec{PlanetArchetype::ProtoTheia, 6.417e23, 4000.0, 4000.0, 0.0, 0};
    const uint32_t sphereN = 6000;
    const BuiltPlanetHost sphere = PlanetBuilder::buildPlanet(hotSpec, sphereN, 0, 8, 42);

    DeviceParticlesGuard sGuard(sphereN);
    DeviceParticlesGuard sAltGuard(sphereN);
    DeviceBvhGuard sBvhGuard(sphereN);
    sGuard.dev.count = sphereN;
    sAltGuard.dev.count = sphereN;

    std::vector<uint32_t> origIds(sphereN);
    for (uint32_t i = 0; i < sphereN; ++i) origIds[i] = i;

    CUDA_CHECK(cudaMemcpy(sGuard.dev.posX, sphere.posX.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.posY, sphere.posY.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.posZ, sphere.posZ.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.mass, sphere.mass.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.h, sphere.h0.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.h0, sphere.h0.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.rho, sphere.rho.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.u, sphere.u.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.tempK, sphere.tempK.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.epsFail, sphere.epsFail.data(), sphereN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.materialId, sphere.materialId.data(), sphereN * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sGuard.dev.originalId, origIds.data(), sphereN * sizeof(uint32_t), cudaMemcpyHostToDevice));

    cuda::launchSortAndBuildBvh(sGuard.dev, sAltGuard.dev, sBvhGuard.bvh);
    cuda::launchSphPass1Neighbors(sGuard.dev, sBvhGuard.bvh);
    cuda::launchEvaluateEos(sGuard.dev);
    cuda::launchSphPass2Forces(sGuard.dev, sBvhGuard.bvh);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> px(sphereN), py(sphereN), pz(sphereN), surf(sphereN), radW(sphereN);
    CUDA_CHECK(cudaMemcpy(px.data(), sGuard.dev.posX, sphereN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(py.data(), sGuard.dev.posY, sphereN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pz.data(), sGuard.dev.posZ, sphereN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(surf.data(), sGuard.dev.surfFactor, sphereN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(radW.data(), sGuard.dev.radPowerW, sphereN * sizeof(float), cudaMemcpyDeviceToHost));

    const double radiusPsu = sphere.equatorialRadiusM / units::L_UNIT_M;
    uint32_t interiorCount = 0, interiorZeroCount = 0;
    uint32_t surfaceCount = 0, surfaceRadiatingCount = 0;
    for (uint32_t i = 0; i < sphereN; ++i) {
        const double r = std::sqrt(
            static_cast<double>(px[i]) * px[i] +
            static_cast<double>(py[i]) * py[i] +
            static_cast<double>(pz[i]) * pz[i]);
        if (r < 0.35 * radiusPsu) {
            ++interiorCount;
            if (surf[i] == 0.0f && radW[i] == 0.0f) {
                ++interiorZeroCount;
            }
        } else if (r > 0.94 * radiusPsu) {
            ++surfaceCount;
            if (surf[i] > 0.3f && radW[i] > 0.0f) {
                ++surfaceRadiatingCount;
            }
        }
    }

    ASSERT_GT(interiorCount, 50u);
    ASSERT_GT(surfaceCount, 50u);
    EXPECT_EQ(interiorZeroCount, interiorCount);
    EXPECT_GT(surfaceRadiatingCount, surfaceCount / 2);
}

} // namespace
