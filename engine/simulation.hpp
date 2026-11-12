#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "types.hpp"

namespace astraea {

class Simulation {
public:
    explicit Simulation(uint32_t maxParticles = 250000);
    ~Simulation();

    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    Simulation(Simulation&&) noexcept;
    Simulation& operator=(Simulation&&) noexcept;

    void loadScenario(const CollisionScenarioConfig& config);
    void relaxBody(uint8_t bodyId, uint32_t steps, double& outFinalVrmsOverVesc);

    TelemetrySnapshot step(double maxDtSec = 5.0);
    TelemetrySnapshot stepForDuration(double durationSec, double maxDtSec = 5.0);

    void setTimeScale(double scale);
    void setSlicePlane(const SlicePlaneConfig& slice);

    TelemetrySnapshot computeTelemetry();
    void extractRenderFrame(
        std::vector<uint8_t>& outParticleBytes,
        std::vector<uint8_t>& outTracerBytes,
        std::vector<uint8_t>* outSpeedBytes = nullptr);

    std::string exportTelemetryCsv() const;
    std::string exportTelemetryJson() const;
    const std::vector<TelemetrySnapshot>& telemetryHistory() const;
    const CollisionScenarioConfig& activeConfig() const;
    const SlicePlaneConfig& slicePlane() const;
    double timeScale() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace astraea
