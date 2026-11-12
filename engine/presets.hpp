#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "types.hpp"

namespace astraea {

struct PlanetLayerDef {
    MaterialId material = MaterialId::Granite;
    double     massFraction = 0.0;
};

struct ArchetypeComposition {
    std::array<PlanetLayerDef, 3> layers{};
    uint32_t                      layerCount = 0;
    double                        defaultMassKg = units::M_EARTH_SI;
    double                        defaultSurfaceTempK = 288.0;
    double                        defaultCoreTempK = 5500.0;
    double                        defaultSpinOmegaRadS = 0.0;
};

const ArchetypeComposition& getArchetypeComposition(PlanetArchetype archetype);
BodyInitSpec makeDefaultBodySpec(PlanetArchetype archetype);

CollisionScenarioConfig makePresetConfig(ScenarioPresetId preset, uint32_t totalParticles = 100000);
ScenarioPresetId parsePresetName(const std::string& name); // throws std::invalid_argument on unknown
std::string presetToString(ScenarioPresetId id);

} // namespace astraea
