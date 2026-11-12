#include "presets.hpp"

#include <stdexcept>

namespace astraea {

namespace {

constexpr ArchetypeComposition kArchetypes[6] = {
    // 0: LivingTerra
    {
        {{
            {MaterialId::Iron, 0.320},
            {MaterialId::Granite, 0.675},
            {MaterialId::WaterIce, 0.005}
        }},
        3,
        5.97219e24,
        288.0,
        5500.0,
        7.292115e-5
    },
    // 1: IronCore
    {
        {{
            {MaterialId::Iron, 0.680},
            {MaterialId::Granite, 0.320},
            {MaterialId::Granite, 0.0}
        }},
        2,
        3.3011e23,
        440.0,
        3200.0,
        1.24e-6
    },
    // 2: OceanIce
    {
        {{
            {MaterialId::Iron, 0.150},
            {MaterialId::Granite, 0.450},
            {MaterialId::WaterIce, 0.400}
        }},
        3,
        4.0e24,
        275.0,
        3800.0,
        5.0e-5
    },
    // 3: LavaIo
    {
        {{
            {MaterialId::Iron, 0.250},
            {MaterialId::Granite, 0.750},
            {MaterialId::Granite, 0.0}
        }},
        2,
        2.0e24,
        1550.0,
        4200.0,
        4.11e-5
    },
    // 4: GasGiant
    {
        {{
            {MaterialId::Iron, 0.050},
            {MaterialId::Granite, 0.100},
            {MaterialId::HydrogenHelium, 0.850}
        }},
        3,
        1.2e26,
        165.0,
        18000.0,
        1.76e-4
    },
    // 5: ProtoTheia
    {
        {{
            {MaterialId::Iron, 0.220},
            {MaterialId::Granite, 0.580},
            {MaterialId::Regolith, 0.200}
        }},
        3,
        6.417e23,
        240.0,
        2200.0,
        7.09e-5
    }
};

} // namespace

const ArchetypeComposition& getArchetypeComposition(PlanetArchetype archetype) {
    uint8_t idx = static_cast<uint8_t>(archetype);
    if (idx > 5) {
        idx = 0;
    }
    return kArchetypes[idx];
}

BodyInitSpec makeDefaultBodySpec(PlanetArchetype archetype) {
    const ArchetypeComposition& comp = getArchetypeComposition(archetype);
    BodyInitSpec spec;
    spec.archetype = archetype;
    spec.massKg = comp.defaultMassKg;
    spec.surfaceTempK = comp.defaultSurfaceTempK;
    spec.coreTempK = comp.defaultCoreTempK;
    spec.spinOmegaRadS = comp.defaultSpinOmegaRadS;
    spec.relaxSteps = 80;
    return spec;
}

std::string presetToString(ScenarioPresetId id) {
    switch (id) {
        case ScenarioPresetId::EarthTheiaMoonForming:
            return "EarthTheiaMoonForming";
        case ScenarioPresetId::MercuryMantleStrip:
            return "MercuryMantleStrip";
        case ScenarioPresetId::OceanWorldSteamPlume:
            return "OceanWorldSteamPlume";
        case ScenarioPresetId::GasGiantCoreImpact:
            return "GasGiantCoreImpact";
        case ScenarioPresetId::LivingGaiaAsteroidCataclysm:
            return "LivingGaiaAsteroidCataclysm";
    }
    throw std::invalid_argument("Unknown ScenarioPresetId");
}

ScenarioPresetId parsePresetName(const std::string& name) {
    if (name == "EarthTheiaMoonForming") {
        return ScenarioPresetId::EarthTheiaMoonForming;
    }
    if (name == "MercuryMantleStrip") {
        return ScenarioPresetId::MercuryMantleStrip;
    }
    if (name == "OceanWorldSteamPlume") {
        return ScenarioPresetId::OceanWorldSteamPlume;
    }
    if (name == "GasGiantCoreImpact") {
        return ScenarioPresetId::GasGiantCoreImpact;
    }
    if (name == "LivingGaiaAsteroidCataclysm") {
        return ScenarioPresetId::LivingGaiaAsteroidCataclysm;
    }
    throw std::invalid_argument("Unknown preset name: " + name);
}

CollisionScenarioConfig makePresetConfig(ScenarioPresetId preset, uint32_t totalParticles) {
    CollisionScenarioConfig cfg;
    cfg.presetName = presetToString(preset);
    cfg.totalParticles = totalParticles;
    cfg.tracerGridRes = 16;
    cfg.randomSeed = 42;

    switch (preset) {
        case ScenarioPresetId::EarthTheiaMoonForming:
            cfg.target = makeDefaultBodySpec(PlanetArchetype::LivingTerra);
            cfg.target.massKg = 1.0 * units::M_EARTH_SI;
            cfg.impactor = makeDefaultBodySpec(PlanetArchetype::ProtoTheia);
            cfg.impactor.massKg = 0.13 * units::M_EARTH_SI;
            cfg.impactVelocityRatioVesc = 1.05;
            cfg.impactParamB = 0.72;
            cfg.initialSepRadii = 3.0;
            break;

        case ScenarioPresetId::MercuryMantleStrip:
            cfg.target = makeDefaultBodySpec(PlanetArchetype::ProtoTheia);
            cfg.target.massKg = 0.45 * units::M_EARTH_SI;
            cfg.target.surfaceTempK = 440.0;
            cfg.impactor = makeDefaultBodySpec(PlanetArchetype::IronCore);
            cfg.impactor.massKg = 0.22 * units::M_EARTH_SI;
            cfg.impactVelocityRatioVesc = 2.80;
            cfg.impactParamB = 0.35;
            cfg.initialSepRadii = 3.2;
            break;

        case ScenarioPresetId::OceanWorldSteamPlume:
            cfg.target = makeDefaultBodySpec(PlanetArchetype::OceanIce);
            cfg.target.massKg = 0.75 * units::M_EARTH_SI;
            cfg.impactor = makeDefaultBodySpec(PlanetArchetype::ProtoTheia);
            cfg.impactor.massKg = 0.15 * units::M_EARTH_SI;
            cfg.impactVelocityRatioVesc = 1.35;
            cfg.impactParamB = 0.25;
            cfg.initialSepRadii = 3.2;
            break;

        case ScenarioPresetId::GasGiantCoreImpact:
            cfg.target = makeDefaultBodySpec(PlanetArchetype::GasGiant);
            cfg.target.massKg = 20.0 * units::M_EARTH_SI;
            cfg.impactor = makeDefaultBodySpec(PlanetArchetype::IronCore);
            cfg.impactor.massKg = 1.0 * units::M_EARTH_SI;
            cfg.impactVelocityRatioVesc = 1.40;
            cfg.impactParamB = 0.15;
            cfg.initialSepRadii = 2.8;
            break;

        case ScenarioPresetId::LivingGaiaAsteroidCataclysm:
            cfg.target = makeDefaultBodySpec(PlanetArchetype::LivingTerra);
            cfg.target.massKg = 1.0 * units::M_EARTH_SI;
            cfg.impactor = makeDefaultBodySpec(PlanetArchetype::ProtoTheia);
            cfg.impactor.massKg = 0.025 * units::M_EARTH_SI;
            cfg.impactor.coreTempK = 1800.0;
            cfg.impactor.spinOmegaRadS = 0.0;
            cfg.impactVelocityRatioVesc = 1.65;
            cfg.impactParamB = 0.30;
            cfg.initialSepRadii = 3.2;
            break;
    }

    return cfg;
}

} // namespace astraea
