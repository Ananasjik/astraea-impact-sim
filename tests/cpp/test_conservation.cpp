#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "presets.hpp"
#include "simulation.hpp"
#include "types.hpp"
#include "units.hpp"

namespace {

using namespace astraea;

double vecNorm3(const double v[3]) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

double vecDiffNorm3(const double a[3], const double b[3]) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

float readF32(const uint8_t* p) {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(float));
    return v;
}

uint16_t readU16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

std::string findSimHeadlessBinary() {
    std::vector<std::string> candidates;
#ifdef ASTRAEA_BIN_DIR
    candidates.emplace_back(std::string(ASTRAEA_BIN_DIR) + "/sim_headless");
#endif
    candidates.emplace_back("./sim_headless");
    candidates.emplace_back("./build/sim_headless");
    candidates.emplace_back("../sim_headless");
    candidates.emplace_back("../../sim_headless");

    for (const auto& c : candidates) {
        if (::access(c.c_str(), X_OK) == 0) {
            return c;
        }
    }
    return {};
}

int runProcessCapture(const std::string& exe, const std::vector<std::string>& args, std::string& outOutput) {
    int pipefd[2]{-1, -1};
    if (::pipe(pipefd) != 0) {
        return -1;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        return -1;
    }

    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);

        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(exe.c_str()));
        for (const auto& a : args) {
            argv.push_back(const_cast<char*>(a.c_str()));
        }
        argv.push_back(nullptr);
        ::execv(exe.c_str(), argv.data());
        ::_exit(127);
    }

    ::close(pipefd[1]);
    outOutput.clear();
    char buf[4096];
    ssize_t n = 0;
    while ((n = ::read(pipefd[0], buf, sizeof(buf))) > 0) {
        outOutput.append(buf, static_cast<size_t>(n));
    }
    ::close(pipefd[0]);

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (!WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
}

std::string readWholeFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) return {};
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

TEST(ConservationTest, EarthTheiaGiantImpactConservationAudit20k_AC5) {
    const CollisionScenarioConfig cfg = makePresetConfig(ScenarioPresetId::EarthTheiaMoonForming, 20000);
    Simulation sim(25000);
    sim.loadScenario(cfg);

    const TelemetrySnapshot t0 = sim.computeTelemetry();
    ASSERT_NE(t0.initialTotalEnergyJ, 0.0);
    ASSERT_EQ(t0.particleCount, 20000u);

    for (int s = 0; s < 150; ++s) {
        sim.step(5.0);
    }

    const TelemetrySnapshot t1 = sim.computeTelemetry();
    EXPECT_GE(t1.frameSeq, 150u);
    EXPECT_GT(t1.simTimeSec, 0.0);
    EXPECT_TRUE(std::isfinite(t1.energyDriftRel));
    EXPECT_LT(std::abs(t1.energyDriftRel), 0.01);

    const double manualDrift =
        (t1.totalEnergyJ + t1.radiatedEnergyJ - t0.initialTotalEnergyJ) / std::abs(t0.initialTotalEnergyJ);
    EXPECT_LT(std::abs(manualDrift), 0.01);

    const double sumE = t1.kineticEnergyJ + t1.potentialEnergyJ + t1.internalEnergyJ;
    EXPECT_NEAR(t1.totalEnergyJ, sumE, 1.0e-5 * std::abs(t1.totalEnergyJ) + 1.0e20);

    const double totalMassKg = cfg.target.massKg + cfg.impactor.massKg;
    const double vEscApproxMS = std::sqrt(2.0 * units::G_SI * totalMassKg / (1.4 * units::R_EARTH_SI));
    const double dP = vecDiffNorm3(t1.momentumKgMS, t0.momentumKgMS);
    EXPECT_LT(dP / (totalMassKg * vEscApproxMS), 1.0e-4);

    const double L0 = vecNorm3(t0.angularMomentumKgM2S);
    ASSERT_GT(L0, 0.0);
    const double dL = vecDiffNorm3(t1.angularMomentumKgM2S, t0.angularMomentumKgM2S);
    EXPECT_LT(dL / L0, 0.005);
}

TEST(ConservationTest, Rtx4050Throughput100kParticlesAndVramBudget_AC8) {
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    size_t freeBefore = 0, totalMem = 0;
    ASSERT_EQ(cudaMemGetInfo(&freeBefore, &totalMem), cudaSuccess);

    const CollisionScenarioConfig cfg =
        makePresetConfig(ScenarioPresetId::LivingGaiaAsteroidCataclysm, 100000);
    Simulation sim(110000);
    sim.loadScenario(cfg);

    // Warm up CUDA caches / CUB temp storage
    for (int i = 0; i < 3; ++i) {
        sim.step(5.0);
    }

    std::vector<double> stepTimesMs;
    stepTimesMs.reserve(20);
    for (int i = 0; i < 20; ++i) {
        sim.step(5.0);
        const TelemetrySnapshot snap = sim.computeTelemetry();
        EXPECT_TRUE(std::isfinite(snap.gpuStepMs));
        stepTimesMs.push_back(snap.gpuStepMs);
    }

    size_t freeAfter = 0;
    ASSERT_EQ(cudaMemGetInfo(&freeAfter, &totalMem), cudaSuccess);
    if (freeBefore > freeAfter) {
        const size_t usedBytes = freeBefore - freeAfter;
        EXPECT_LT(usedBytes, 512ull * 1024ull * 1024ull);
    }

    std::sort(stepTimesMs.begin(), stepTimesMs.end());
    const double medianMs = stepTimesMs[stepTimesMs.size() / 2];
    EXPECT_GT(medianMs, 0.0);
    EXPECT_LE(medianMs, 33.3);
}

TEST(ConservationTest, SurfaceTracerContinuityAcrossMortonResorts_AC9) {
    const CollisionScenarioConfig cfg =
        makePresetConfig(ScenarioPresetId::LivingGaiaAsteroidCataclysm, 30000);
    ASSERT_EQ(cfg.tracerGridRes, 16u);
    const uint32_t expectedTracers = 6u * 16u * 16u;

    Simulation sim(35000);
    sim.loadScenario(cfg);

    std::vector<uint8_t> particleBytes, tracerBytes;
    for (int step = 0; step <= 100; ++step) {
        if (step % 10 == 0) {
            sim.extractRenderFrame(particleBytes, tracerBytes, nullptr);
            ASSERT_EQ(tracerBytes.size(), 40u * expectedTracers);

            std::vector<uint32_t> seen(expectedTracers, 0u);
            for (uint32_t t = 0; t < expectedTracers; ++t) {
                const uint8_t* rec = tracerBytes.data() + 40u * t;
                const uint16_t anchorIdx = readU16(rec + 0);
                const uint16_t flags = readU16(rec + 2);
                const float xKm = readF32(rec + 4);
                const float yKm = readF32(rec + 8);
                const float zKm = readF32(rec + 12);
                const float dispKm = readF32(rec + 16);
                const float normalDelta = readF32(rec + 20);
                const float tempK = readF32(rec + 24);
                const float damage = readF32(rec + 28);
                const float overpressureKpa = readF32(rec + 32);
                const float tsunamiHeightM = readF32(rec + 36);

                ASSERT_LT(anchorIdx, expectedTracers) << "step=" << step;
                seen[anchorIdx]++;
                EXPECT_LE(flags, 3u);
                EXPECT_TRUE(std::isfinite(xKm));
                EXPECT_TRUE(std::isfinite(yKm));
                EXPECT_TRUE(std::isfinite(zKm));
                EXPECT_LT(std::sqrt(xKm * xKm + yKm * yKm + zKm * zKm), 5.0e5f);
                EXPECT_TRUE(std::isfinite(dispKm));
                EXPECT_TRUE(std::isfinite(normalDelta));
                EXPECT_GE(normalDelta, 0.0f);
                EXPECT_LE(normalDelta, 3.2f);
                EXPECT_TRUE(std::isfinite(tempK));
                EXPECT_GE(tempK, 3.0f);
                EXPECT_LE(tempK, 300000.0f);
                EXPECT_TRUE(std::isfinite(damage));
                EXPECT_GE(damage, 0.0f);
                EXPECT_LE(damage, 1.0f);
                EXPECT_TRUE(std::isfinite(overpressureKpa));
                EXPECT_GE(overpressureKpa, 0.0f);
                EXPECT_TRUE(std::isfinite(tsunamiHeightM));
                EXPECT_GT(tsunamiHeightM, -1.0e5f);
                EXPECT_LT(tsunamiHeightM, 1.0e6f);
            }
            for (uint32_t idx = 0; idx < expectedTracers; ++idx) {
                EXPECT_EQ(seen[idx], 1u) << "step=" << step << " anchor=" << idx;
            }
        }
        if (step < 100) {
            sim.step(5.0);
        }
    }
}

TEST(ConservationTest, ExtractRenderFrameLayoutQuantizationAndOptionalSpeed_AC12) {
    const CollisionScenarioConfig cfg =
        makePresetConfig(ScenarioPresetId::LivingGaiaAsteroidCataclysm, 100000);
    Simulation sim(105000);
    sim.loadScenario(cfg);
    sim.step(5.0);
    sim.step(5.0);

    const TelemetrySnapshot snap = sim.computeTelemetry();
    EXPECT_EQ(snap.particleCount, 100000u);
    EXPECT_EQ(snap.tracerCount, 1536u);
    EXPECT_TRUE(snap.livingBodyPresent);

    std::vector<uint8_t> pBytesNoSpeed, tBytesNoSpeed;
    sim.extractRenderFrame(pBytesNoSpeed, tBytesNoSpeed, nullptr);
    EXPECT_EQ(pBytesNoSpeed.size(), 20u * 100000u);
    EXPECT_EQ(tBytesNoSpeed.size(), 40u * 1536u);

    std::vector<uint8_t> pBytesWithSpeed, tBytesWithSpeed, speedBytes;
    sim.extractRenderFrame(pBytesWithSpeed, tBytesWithSpeed, &speedBytes);
    EXPECT_EQ(pBytesWithSpeed.size(), 20u * 100000u);
    EXPECT_EQ(tBytesWithSpeed.size(), 40u * 1536u);
    EXPECT_EQ(speedBytes.size(), 100000u);
    EXPECT_EQ(pBytesNoSpeed, pBytesWithSpeed);
    EXPECT_EQ(tBytesNoSpeed, tBytesWithSpeed);

    const double ln20000 = std::log(20000.0);
    const double ln1e7 = std::log(1.0e7);
    const double ln2000 = std::log(2000.0);

    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
    double maxTempDecoded = 0.0;
    double maxPressDecoded = 0.0;
    double maxImpactorSpeed = 0.0;
    uint32_t body0Count = 0, body1Count = 0;

    for (size_t i = 0; i < 100000u; ++i) {
        const uint8_t* rec = pBytesWithSpeed.data() + 20u * i;
        const float xKm = readF32(rec + 0);
        const float yKm = readF32(rec + 4);
        const float zKm = readF32(rec + 8);
        const uint16_t tempU16 = readU16(rec + 12);
        const uint16_t pressU16 = readU16(rec + 14);
        const uint8_t matId = rec[16];
        const uint8_t phaseAndBody = rec[17];
        const uint8_t bodyId = phaseAndBody >> 4;
        const uint8_t phaseId = phaseAndBody & 0x0F;
        const uint8_t damageU8 = rec[18];
        const uint8_t radiusU8 = rec[19];

        ASSERT_TRUE(std::isfinite(xKm));
        ASSERT_TRUE(std::isfinite(yKm));
        ASSERT_TRUE(std::isfinite(zKm));
        sumX += xKm;
        sumY += yKm;
        sumZ += zKm;

        EXPECT_LE(matId, 4u);
        EXPECT_LE(phaseId, 5u);
        EXPECT_LE(bodyId, 1u);
        if (bodyId == 0u) ++body0Count;
        if (bodyId == 1u) ++body1Count;

        const double tDec = 10.0 * std::exp((static_cast<double>(tempU16) / 65535.0) * ln20000);
        maxTempDecoded = std::max(maxTempDecoded, tDec);

        const double pDec = (pressU16 == 0)
                                ? 0.0
                                : 1.0e-3 * std::exp((static_cast<double>(pressU16 - 1) / 65534.0) * ln1e7);
        maxPressDecoded = std::max(maxPressDecoded, pDec);

        const double dmg = static_cast<double>(damageU8) / 255.0;
        EXPECT_GE(dmg, 0.0);
        EXPECT_LE(dmg, 1.0);

        const double radKm = std::exp((static_cast<double>(radiusU8) / 255.0) * ln2000);
        EXPECT_GE(radKm, 1.0);
        EXPECT_LE(radKm, 2000.0 + 1e-6);

        const double spdKmS = 0.05 * std::exp((static_cast<double>(speedBytes[i]) / 255.0) * ln2000);
        EXPECT_TRUE(std::isfinite(spdKmS));
        if (bodyId == 1u) {
            maxImpactorSpeed = std::max(maxImpactorSpeed, spdKmS);
        }
    }

    EXPECT_GT(body0Count, 90000u);
    EXPECT_GT(body1Count, 1000u);
    EXPECT_GT(maxImpactorSpeed, 5.0);

    // COM-centered particle coordinates
    const double meanDistKm = std::sqrt(sumX * sumX + sumY * sumY + sumZ * sumZ) / 100000.0;
    EXPECT_LT(meanDistKm, 500.0);

    // Log quantization accuracy < 0.1% relative error
    if (snap.maxTempK >= 10.0 && snap.maxTempK <= 200000.0) {
        EXPECT_NEAR(maxTempDecoded, snap.maxTempK, 0.001 * snap.maxTempK);
    }
    if (snap.maxPressureGPa >= 1.0e-3 && snap.maxPressureGPa <= 1.0e4) {
        EXPECT_NEAR(maxPressDecoded, snap.maxPressureGPa, 0.001 * snap.maxPressureGPa);
    }

    // Non-LivingTerra preset emits 0 tracers
    Simulation noTracerSim(5000);
    noTracerSim.loadScenario(makePresetConfig(ScenarioPresetId::MercuryMantleStrip, 4000));
    std::vector<uint8_t> p2, t2;
    noTracerSim.extractRenderFrame(p2, t2, nullptr);
    EXPECT_EQ(p2.size(), 20u * 4000u);
    EXPECT_TRUE(t2.empty());
}

TEST(ConservationTest, TelemetryExportCsvJsonAndSimulationControls_AC10) {
    const CollisionScenarioConfig cfg =
        makePresetConfig(ScenarioPresetId::LivingGaiaAsteroidCataclysm, 4000);
    Simulation sim(5000);
    sim.loadScenario(cfg);

    EXPECT_EQ(sim.activeConfig().presetName, cfg.presetName);
    EXPECT_EQ(sim.activeConfig().totalParticles, 4000u);

    sim.setTimeScale(1.0);
    sim.setSlicePlane({true, 0.0f, 1.0f, 0.0f, 0.0f});
    sim.setSlicePlane({false, 0.0f, 1.0f, 0.0f, 0.0f});

    for (int i = 0; i < 10; ++i) {
        sim.step(5.0);
    }

    EXPECT_GE(sim.telemetryHistory().size(), 10u);

    const std::string csv = sim.exportTelemetryCsv();
    EXPECT_FALSE(csv.empty());
    EXPECT_NE(csv.find("frameSeq"), std::string::npos);
    EXPECT_NE(csv.find("simTimeSec"), std::string::npos);
    EXPECT_NE(csv.find("kineticEnergyJ"), std::string::npos);
    EXPECT_NE(csv.find("potentialEnergyJ"), std::string::npos);
    EXPECT_NE(csv.find("internalEnergyJ"), std::string::npos);
    EXPECT_NE(csv.find("radiatedEnergyJ"), std::string::npos);
    EXPECT_NE(csv.find("energyDriftRel"), std::string::npos);
    const size_t lineCount = static_cast<size_t>(std::count(csv.begin(), csv.end(), '\n'));
    EXPECT_GE(lineCount, 10u);

    const std::string json = sim.exportTelemetryJson();
    EXPECT_FALSE(json.empty());
    const std::array<const char*, 29> expectedFields = {
        "frameSeq", "simTimeSec", "dtSec", "outcome", "kineticEnergyJ",
        "potentialEnergyJ", "internalEnergyJ", "radiatedEnergyJ", "totalEnergyJ",
        "initialTotalEnergyJ", "energyDriftRel", "momentumKgMS", "angularMomentumKgM2S",
        "virialRatio", "maxTempK", "maxPressureGPa", "boundMassFrac", "diskMassFrac",
        "ejectaMassFrac", "qRatioQrd", "gpuStepMs", "livingBodyComKm",
        "livingBodyRadiusKm", "livingBodySpinRadS", "impactorComKm",
        "particleCount", "tracerCount", "tracerGridRes", "livingBodyPresent"
    };
    for (const char* field : expectedFields) {
        EXPECT_NE(json.find(field), std::string::npos) << "Missing JSON field: " << field;
    }
}

TEST(ConservationTest, SimHeadlessCliExecutionAndExports_AC10) {
    const std::string binPath = findSimHeadlessBinary();
    ASSERT_FALSE(binPath.empty()) << "sim_headless executable not found (checked ASTRAEA_BIN_DIR and cwd)";

    std::string listOut;
    const int listExit = runProcessCapture(binPath, {"--list-presets"}, listOut);
    EXPECT_EQ(listExit, 0);
    EXPECT_NE(listOut.find("EarthTheiaMoonForming"), std::string::npos);
    EXPECT_NE(listOut.find("MercuryMantleStrip"), std::string::npos);
    EXPECT_NE(listOut.find("OceanWorldSteamPlume"), std::string::npos);
    EXPECT_NE(listOut.find("GasGiantCoreImpact"), std::string::npos);
    EXPECT_NE(listOut.find("LivingGaiaAsteroidCataclysm"), std::string::npos);

    const std::string pidStr = std::to_string(static_cast<long>(::getpid()));
    const std::string csvPath = "/tmp/astraea_ac10_" + pidStr + ".csv";
    const std::string jsonPath = "/tmp/astraea_ac10_" + pidStr + ".json";
    std::filesystem::remove(csvPath);
    std::filesystem::remove(jsonPath);

    std::string runOut;
    const int exitCode = runProcessCapture(
        binPath,
        {
            "--preset", "LivingGaiaAsteroidCataclysm",
            "--particles", "10000",
            "--steps", "40",
            "--output-csv", csvPath,
            "--dump-telemetry", jsonPath,
            "--verify-conservation"
        },
        runOut);
    EXPECT_EQ(exitCode, 0) << "sim_headless output:\n" << runOut;

    const std::string csvContent = readWholeFile(csvPath);
    const std::string jsonContent = readWholeFile(jsonPath);
    std::filesystem::remove(csvPath);
    std::filesystem::remove(jsonPath);

    ASSERT_FALSE(csvContent.empty());
    ASSERT_FALSE(jsonContent.empty());

    const size_t csvLines = static_cast<size_t>(std::count(csvContent.begin(), csvContent.end(), '\n'));
    EXPECT_GE(csvLines, 40u);
    EXPECT_NE(csvContent.find("simTimeSec"), std::string::npos);
    EXPECT_NE(csvContent.find("energyDriftRel"), std::string::npos);

    const std::array<const char*, 29> requiredFields = {
        "frameSeq", "simTimeSec", "dtSec", "outcome", "kineticEnergyJ",
        "potentialEnergyJ", "internalEnergyJ", "radiatedEnergyJ", "totalEnergyJ",
        "initialTotalEnergyJ", "energyDriftRel", "momentumKgMS", "angularMomentumKgM2S",
        "virialRatio", "maxTempK", "maxPressureGPa", "boundMassFrac", "diskMassFrac",
        "ejectaMassFrac", "qRatioQrd", "gpuStepMs", "livingBodyComKm",
        "livingBodyRadiusKm", "livingBodySpinRadS", "impactorComKm",
        "particleCount", "tracerCount", "tracerGridRes", "livingBodyPresent"
    };
    for (const char* field : requiredFields) {
        EXPECT_NE(jsonContent.find(field), std::string::npos)
            << "Missing field '" << field << "' in headless JSON";
    }
}

TEST(ConservationTest, CollisionOutcomeClassifications_AC21) {
    Simulation sim(8000);

    // 0) Pre-impact state classifies as Approaching
    {
        const CollisionScenarioConfig cfg =
            makePresetConfig(ScenarioPresetId::EarthTheiaMoonForming, 4000);
        sim.loadScenario(cfg);
        EXPECT_EQ(sim.computeTelemetry().outcome, CollisionOutcome::Approaching);
    }

    // 1) Low-energy head-on / gentle impact -> Merging
    {
        CollisionScenarioConfig cfg =
            makePresetConfig(ScenarioPresetId::OceanWorldSteamPlume, 5000);
        cfg.impactVelocityRatioVesc = 0.95;
        cfg.impactParamB = 0.05;
        cfg.initialSepRadii = 1.25;
        sim.loadScenario(cfg);
        sim.stepForDuration(1600.0, 10.0);
        const TelemetrySnapshot t = sim.computeTelemetry();
        EXPECT_EQ(t.outcome, CollisionOutcome::Merging);
        EXPECT_GT(t.boundMassFrac, 0.70);
    }

    // 2) Oblique fast grazing impact -> HitAndRun
    {
        CollisionScenarioConfig cfg;
        cfg.presetName = "HitAndRunCheck";
        cfg.target = {PlanetArchetype::ProtoTheia, 0.50 * units::M_EARTH_SI, 240.0, 2200.0, 0.0, 60};
        cfg.impactor = {PlanetArchetype::ProtoTheia, 0.35 * units::M_EARTH_SI, 240.0, 2000.0, 0.0, 60};
        cfg.impactVelocityRatioVesc = 1.85;
        cfg.impactParamB = 0.80;
        cfg.initialSepRadii = 1.40;
        cfg.totalParticles = 5000;
        cfg.tracerGridRes = 8;
        cfg.randomSeed = 42;
        sim.loadScenario(cfg);
        sim.stepForDuration(1600.0, 10.0);
        const TelemetrySnapshot t = sim.computeTelemetry();
        EXPECT_EQ(t.outcome, CollisionOutcome::HitAndRun);
    }

    // 3) High-energy head-on shattering collision -> CatastrophicDisruption
    {
        CollisionScenarioConfig cfg;
        cfg.presetName = "CatastrophicDisruptionCheck";
        cfg.target = {PlanetArchetype::ProtoTheia, 0.45 * units::M_EARTH_SI, 440.0, 2200.0, 0.0, 60};
        cfg.impactor = {PlanetArchetype::ProtoTheia, 0.45 * units::M_EARTH_SI, 440.0, 2200.0, 0.0, 60};
        cfg.impactVelocityRatioVesc = 4.20;
        cfg.impactParamB = 0.05;
        cfg.initialSepRadii = 1.25;
        cfg.totalParticles = 5000;
        cfg.tracerGridRes = 8;
        cfg.randomSeed = 42;
        sim.loadScenario(cfg);
        sim.stepForDuration(1500.0, 10.0);
        const TelemetrySnapshot t = sim.computeTelemetry();
        EXPECT_EQ(t.outcome, CollisionOutcome::CatastrophicDisruption);
        EXPECT_LT(t.boundMassFrac, 0.55);
    }
}

} // namespace
