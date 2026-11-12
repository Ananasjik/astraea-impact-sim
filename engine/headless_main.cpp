#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "planet_builder.hpp"
#include "presets.hpp"
#include "simulation.hpp"
#include "units.hpp"

namespace {

const char* outcomeName(astraea::CollisionOutcome outcome) {
    switch (outcome) {
        case astraea::CollisionOutcome::Approaching:            return "Approaching";
        case astraea::CollisionOutcome::Merging:                return "Merging";
        case astraea::CollisionOutcome::GrazeAndMerge:          return "GrazeAndMerge";
        case astraea::CollisionOutcome::HitAndRun:              return "HitAndRun";
        case astraea::CollisionOutcome::CatastrophicDisruption: return "CatastrophicDisruption";
    }
    return "Approaching";
}

double vecNorm3(const double v[3]) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

bool isFiniteSnapshot(const astraea::TelemetrySnapshot& s) {
    return std::isfinite(s.simTimeSec) &&
           std::isfinite(static_cast<double>(s.dtSec)) &&
           std::isfinite(s.kineticEnergyJ) &&
           std::isfinite(s.potentialEnergyJ) &&
           std::isfinite(s.internalEnergyJ) &&
           std::isfinite(s.radiatedEnergyJ) &&
           std::isfinite(s.totalEnergyJ) &&
           std::isfinite(s.initialTotalEnergyJ) &&
           std::isfinite(s.energyDriftRel) &&
           std::isfinite(s.momentumKgMS[0]) && std::isfinite(s.momentumKgMS[1]) && std::isfinite(s.momentumKgMS[2]) &&
           std::isfinite(s.angularMomentumKgM2S[0]) && std::isfinite(s.angularMomentumKgM2S[1]) &&
           std::isfinite(s.angularMomentumKgM2S[2]) &&
           std::isfinite(s.virialRatio) &&
           std::isfinite(s.maxTempK) &&
           std::isfinite(s.maxPressureGPa) &&
           std::isfinite(s.boundMassFrac) &&
           std::isfinite(s.diskMassFrac) &&
           std::isfinite(s.ejectaMassFrac) &&
           std::isfinite(s.qRatioQrd) &&
           std::isfinite(s.gpuStepMs) &&
           std::isfinite(s.livingBodyComKm[0]) && std::isfinite(s.livingBodyComKm[1]) && std::isfinite(s.livingBodyComKm[2]) &&
           std::isfinite(s.livingBodyRadiusKm) &&
           std::isfinite(s.livingBodySpinRadS) &&
           std::isfinite(s.impactorComKm[0]) && std::isfinite(s.impactorComKm[1]) && std::isfinite(s.impactorComKm[2]);
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --preset <name>         Scenario preset (default: LivingGaiaAsteroidCataclysm)\n"
              << "  --particles <uint>      Total particle count (default: 100000)\n"
              << "  --steps <uint>          Number of simulation steps (default: 100)\n"
              << "  --dt-max <double>       Max timestep per step in seconds (default: 5.0)\n"
              << "  --relax-steps <uint>    Override relaxation steps for target & impactor\n"
              << "  --dump-telemetry <path> Write JSON telemetry report to path\n"
              << "  --output-csv <path>     Write CSV telemetry history to path\n"
              << "  --verify-conservation   Exit non-zero if conservation thresholds fail or NaN occurs\n"
              << "  --list-presets          Print available scenario presets and exit\n"
              << "  --help                  Print this help message and exit\n";
}

void listPresets() {
    const astraea::ScenarioPresetId ids[5] = {
        astraea::ScenarioPresetId::EarthTheiaMoonForming,
        astraea::ScenarioPresetId::MercuryMantleStrip,
        astraea::ScenarioPresetId::OceanWorldSteamPlume,
        astraea::ScenarioPresetId::GasGiantCoreImpact,
        astraea::ScenarioPresetId::LivingGaiaAsteroidCataclysm
    };
    for (astraea::ScenarioPresetId id : ids) {
        std::cout << astraea::presetToString(id) << '\n';
    }
}

bool writeTextFile(const std::string& path, const std::string& content) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
    return ofs.good();
}

} // namespace

int main(int argc, char** argv) {
    std::string presetName = "LivingGaiaAsteroidCataclysm";
    uint32_t particles = 100000;
    uint32_t steps = 100;
    double dtMaxSec = 5.0;
    bool hasRelaxOverride = false;
    uint32_t relaxOverride = 80;
    std::string dumpJsonPath;
    std::string outputCsvPath;
    bool verifyConservation = false;

    auto requireArg = [&](int& idx, const char* flag) -> std::string {
        if (idx + 1 >= argc) {
            throw std::invalid_argument(std::string("Missing value for ") + flag);
        }
        return std::string(argv[++idx]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            } else if (arg == "--list-presets") {
                listPresets();
                return 0;
            } else if (arg == "--preset") {
                presetName = requireArg(i, "--preset");
            } else if (arg == "--particles") {
                const long long v = std::stoll(requireArg(i, "--particles"));
                if (v <= 0) {
                    throw std::invalid_argument("--particles must be > 0");
                }
                particles = static_cast<uint32_t>(v);
            } else if (arg == "--steps") {
                const long long v = std::stoll(requireArg(i, "--steps"));
                if (v < 0) {
                    throw std::invalid_argument("--steps must be >= 0");
                }
                steps = static_cast<uint32_t>(v);
            } else if (arg == "--dt-max") {
                dtMaxSec = std::stod(requireArg(i, "--dt-max"));
                if (!(dtMaxSec > 0.0) || !std::isfinite(dtMaxSec)) {
                    throw std::invalid_argument("--dt-max must be > 0");
                }
            } else if (arg == "--relax-steps") {
                const long long v = std::stoll(requireArg(i, "--relax-steps"));
                if (v < 0) {
                    throw std::invalid_argument("--relax-steps must be >= 0");
                }
                hasRelaxOverride = true;
                relaxOverride = static_cast<uint32_t>(v);
            } else if (arg == "--dump-telemetry") {
                dumpJsonPath = requireArg(i, "--dump-telemetry");
            } else if (arg == "--output-csv") {
                outputCsvPath = requireArg(i, "--output-csv");
            } else if (arg == "--verify-conservation") {
                verifyConservation = true;
            } else {
                throw std::invalid_argument("Unknown option: " + arg);
            }
        }

        const astraea::ScenarioPresetId presetId = astraea::parsePresetName(presetName);
        astraea::CollisionScenarioConfig cfg = astraea::makePresetConfig(presetId, particles);
        if (hasRelaxOverride) {
            cfg.target.relaxSteps = relaxOverride;
            cfg.impactor.relaxSteps = relaxOverride;
        }

        // Determine mutual escape velocity scale for momentum normalization
        double rTargetM = 0.0, pTargetC = 0.0;
        astraea::PlanetBuilder::solveHydrostatic1D(cfg.target, rTargetM, pTargetC);
        double rImpM = 0.0, pImpC = 0.0;
        if (cfg.impactor.massKg > 0.0) {
            astraea::PlanetBuilder::solveHydrostatic1D(cfg.impactor, rImpM, pImpC);
        }
        const double totalMassKg = cfg.target.massKg + std::max(0.0, cfg.impactor.massKg);
        const double rScaleM = std::max(rTargetM + rImpM, 1.0);
        const double vEscMs = std::sqrt((2.0 * astraea::units::G_SI * totalMassKg) / rScaleM);

        astraea::Simulation sim(particles);
        sim.loadScenario(cfg);

        const astraea::TelemetrySnapshot t0 = sim.computeTelemetry();
        if (!isFiniteSnapshot(t0)) {
            std::cerr << "ERROR: Non-finite initial telemetry at t=0\n";
            return 1;
        }

        std::vector<double> stepTimesMs;
        stepTimesMs.reserve(steps);
        astraea::TelemetrySnapshot latest = t0;

        for (uint32_t s = 0; s < steps; ++s) {
            latest = sim.step(dtMaxSec);
            stepTimesMs.push_back(latest.gpuStepMs);
            if (!isFiniteSnapshot(latest)) {
                std::cerr << "ERROR: Non-finite telemetry at step " << (s + 1) << '\n';
                return 1;
            }
        }

        double medianGpuMs = 0.0;
        if (!stepTimesMs.empty()) {
            std::vector<double> sorted = stepTimesMs;
            std::sort(sorted.begin(), sorted.end());
            const size_t mid = sorted.size() / 2;
            medianGpuMs = (sorted.size() % 2 == 0)
                              ? 0.5 * (sorted[mid - 1] + sorted[mid])
                              : sorted[mid];
        }

        double dP[3] = {
            latest.momentumKgMS[0] - t0.momentumKgMS[0],
            latest.momentumKgMS[1] - t0.momentumKgMS[1],
            latest.momentumKgMS[2] - t0.momentumKgMS[2]
        };
        double dL[3] = {
            latest.angularMomentumKgM2S[0] - t0.angularMomentumKgM2S[0],
            latest.angularMomentumKgM2S[1] - t0.angularMomentumKgM2S[1],
            latest.angularMomentumKgM2S[2] - t0.angularMomentumKgM2S[2]
        };
        const double dpNorm = vecNorm3(dP);
        const double dlNorm = vecNorm3(dL);
        const double l0Norm = vecNorm3(t0.angularMomentumKgM2S);

        const double pRel = (totalMassKg * vEscMs > 0.0) ? (dpNorm / (totalMassKg * vEscMs)) : 0.0;
        const double lRef = std::max(l0Norm, totalMassKg * rScaleM * vEscMs * 1.0e-6);
        const double lRel = dlNorm / lRef;

        if (!dumpJsonPath.empty()) {
            if (!writeTextFile(dumpJsonPath, sim.exportTelemetryJson())) {
                std::cerr << "ERROR: Failed to write JSON telemetry to " << dumpJsonPath << '\n';
                return 1;
            }
        }
        if (!outputCsvPath.empty()) {
            if (!writeTextFile(outputCsvPath, sim.exportTelemetryCsv())) {
                std::cerr << "ERROR: Failed to write CSV telemetry to " << outputCsvPath << '\n';
                return 1;
            }
        }

        std::cout << std::setprecision(6)
                  << "preset=" << cfg.presetName
                  << " particles=" << latest.particleCount
                  << " tracers=" << latest.tracerCount
                  << " steps=" << steps
                  << " simTimeSec=" << latest.simTimeSec
                  << " dtSec=" << latest.dtSec
                  << " medianGpuStepMs=" << medianGpuMs
                  << " energyDriftRel=" << latest.energyDriftRel
                  << " dPRel=" << pRel
                  << " dLRel=" << lRel
                  << " maxTempK=" << latest.maxTempK
                  << " maxPressureGPa=" << latest.maxPressureGPa
                  << " boundMassFrac=" << latest.boundMassFrac
                  << " diskMassFrac=" << latest.diskMassFrac
                  << " ejectaMassFrac=" << latest.ejectaMassFrac
                  << " qRatioQrd=" << latest.qRatioQrd
                  << " outcome=" << outcomeName(latest.outcome)
                  << '\n';

        if (verifyConservation) {
            bool ok = true;
            if (!(std::abs(latest.energyDriftRel) < 0.01)) {
                std::cerr << "CONSERVATION FAIL: |energyDriftRel| = " << std::abs(latest.energyDriftRel)
                          << " >= 0.01\n";
                ok = false;
            }
            if (!(pRel < 1.0e-4)) {
                std::cerr << "CONSERVATION FAIL: |dP|/(M*v_esc) = " << pRel << " >= 1e-4\n";
                ok = false;
            }
            if (!(lRel < 0.005)) {
                std::cerr << "CONSERVATION FAIL: |dL|/|L0| = " << lRel << " >= 0.005\n";
                ok = false;
            }
            if (!ok) {
                return 1;
            }
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 2;
    }
}
