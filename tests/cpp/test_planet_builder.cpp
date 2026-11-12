#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "planet_builder.hpp"
#include "presets.hpp"
#include "simulation.hpp"
#include "types.hpp"
#include "units.hpp"

namespace {

using namespace astraea;

BodyInitSpec defaultSpecForArchetype(PlanetArchetype arch) {
    BodyInitSpec s{};
    s.archetype = arch;
    s.relaxSteps = 80;
    switch (arch) {
        case PlanetArchetype::LivingTerra:
            s.massKg = 5.97219e24;
            s.coreTempK = 5500.0;
            s.surfaceTempK = 288.0;
            s.spinOmegaRadS = 7.2921e-5;
            break;
        case PlanetArchetype::IronCore:
            s.massKg = 3.3011e23;
            s.coreTempK = 3200.0;
            s.surfaceTempK = 440.0;
            s.spinOmegaRadS = 1.24e-6;
            break;
        case PlanetArchetype::OceanIce:
            s.massKg = 4.0e24;
            s.coreTempK = 3800.0;
            s.surfaceTempK = 275.0;
            s.spinOmegaRadS = 5.0e-5;
            break;
        case PlanetArchetype::LavaIo:
            s.massKg = 2.0e24;
            s.coreTempK = 4200.0;
            s.surfaceTempK = 1550.0;
            s.spinOmegaRadS = 4.11e-5;
            break;
        case PlanetArchetype::GasGiant:
            s.massKg = 1.2e26;
            s.coreTempK = 18000.0;
            s.surfaceTempK = 165.0;
            s.spinOmegaRadS = 1.76e-4;
            break;
        case PlanetArchetype::ProtoTheia:
            s.massKg = 6.417e23;
            s.coreTempK = 2200.0;
            s.surfaceTempK = 240.0;
            s.spinOmegaRadS = 7.09e-5;
            break;
    }
    return s;
}

void expectedCubeNormal(uint32_t face, float u, float v, float& nx, float& ny, float& nz) {
    const float sx = 2.0f * u - 1.0f;
    const float sy = 2.0f * v - 1.0f;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    switch (face) {
        case 0: x =  1.0f; y =  sy;   z = -sx;  break;
        case 1: x = -1.0f; y =  sy;   z =  sx;  break;
        case 2: x =  sx;   y =  1.0f; z = -sy;  break;
        case 3: x =  sx;   y = -1.0f; z =  sy;  break;
        case 4: x =  sx;   y =  sy;   z =  1.0f; break;
        case 5: x = -sx;   y =  sy;   z = -1.0f; break;
        default: break;
    }
    const float invLen = 1.0f / std::sqrt(x * x + y * y + z * z);
    nx = x * invLen;
    ny = y * invLen;
    nz = z * invLen;
}

TEST(PlanetBuilderTest, CubeSphereFaceUvRoundtrip) {
    const std::array<float, 5> coords = {0.05f, 0.25f, 0.5f, 0.75f, 0.95f};
    for (uint32_t face = 0; face < 6; ++face) {
        for (float u : coords) {
            for (float v : coords) {
                float nx = 0.0f, ny = 0.0f, nz = 0.0f;
                cubeFaceUvToUnitNormal(face, u, v, nx, ny, nz);

                float ex = 0.0f, ey = 0.0f, ez = 0.0f;
                expectedCubeNormal(face, u, v, ex, ey, ez);

                EXPECT_NEAR(nx * nx + ny * ny + nz * nz, 1.0f, 1e-5f);
                EXPECT_NEAR(nx, ex, 1e-5f);
                EXPECT_NEAR(ny, ey, 1e-5f);
                EXPECT_NEAR(nz, ez, 1e-5f);

                uint32_t outFace = 99;
                float outU = -1.0f, outV = -1.0f;
                unitNormalToCubeFaceUv(nx, ny, nz, outFace, outU, outV);
                EXPECT_EQ(outFace, face);
                EXPECT_NEAR(outU, u, 1e-4f);
                EXPECT_NEAR(outV, v, 1e-4f);
            }
        }
    }

    // Boundary / corner points roundtrip cleanly to identical normal
    const std::array<float, 2> edges = {0.0f, 1.0f};
    for (uint32_t face = 0; face < 6; ++face) {
        for (float u : edges) {
            for (float v : edges) {
                float nx = 0.0f, ny = 0.0f, nz = 0.0f;
                cubeFaceUvToUnitNormal(face, u, v, nx, ny, nz);

                uint32_t outFace = 99;
                float outU = -1.0f, outV = -1.0f;
                unitNormalToCubeFaceUv(nx, ny, nz, outFace, outU, outV);
                ASSERT_LT(outFace, 6u);
                EXPECT_GE(outU, -1e-5f);
                EXPECT_LE(outU, 1.0f + 1e-5f);
                EXPECT_GE(outV, -1e-5f);
                EXPECT_LE(outV, 1.0f + 1e-5f);

                float nx2 = 0.0f, ny2 = 0.0f, nz2 = 0.0f;
                cubeFaceUvToUnitNormal(outFace, outU, outV, nx2, ny2, nz2);
                EXPECT_NEAR(nx2, nx, 1e-4f);
                EXPECT_NEAR(ny2, ny, 1e-4f);
                EXPECT_NEAR(nz2, nz, 1e-4f);
            }
        }
    }
}

TEST(PlanetBuilderTest, SolveHydrostatic1DAndBuildAllSixArchetypes_AC2) {
    const std::array<PlanetArchetype, 6> archetypes = {
        PlanetArchetype::LivingTerra,
        PlanetArchetype::IronCore,
        PlanetArchetype::OceanIce,
        PlanetArchetype::LavaIo,
        PlanetArchetype::GasGiant,
        PlanetArchetype::ProtoTheia,
    };

    for (PlanetArchetype arch : archetypes) {
        const BodyInitSpec spec = defaultSpecForArchetype(arch);

        double radiusM = 0.0;
        double centralPressurePa = 0.0;
        const auto profile = PlanetBuilder::solveHydrostatic1D(spec, radiusM, centralPressurePa);

        ASSERT_GE(profile.size(), 16u);
        EXPECT_GT(radiusM, 1.0e5);
        EXPECT_GT(centralPressurePa, 1.0e7);
        EXPECT_NEAR(profile.back().enclosedMassKg, spec.massKg, 2.0e-3 * spec.massKg);
        EXPECT_NEAR(profile.back().rM, radiusM, 1.0e-2 * radiusM);

        std::set<MaterialId> materialsSeen;
        for (size_t i = 0; i < profile.size(); ++i) {
            EXPECT_TRUE(std::isfinite(profile[i].rM));
            EXPECT_TRUE(std::isfinite(profile[i].rhoKgM3));
            EXPECT_TRUE(std::isfinite(profile[i].pressurePa));
            EXPECT_TRUE(std::isfinite(profile[i].enclosedMassKg));
            EXPECT_TRUE(std::isfinite(profile[i].tempK));
            EXPECT_GT(profile[i].rhoKgM3, 0.0);
            EXPECT_GE(profile[i].pressurePa, -1.0e5);
            materialsSeen.insert(profile[i].material);

            if (i > 0) {
                EXPECT_GE(profile[i].rM, profile[i - 1].rM);
                EXPECT_GE(profile[i].enclosedMassKg, profile[i - 1].enclosedMassKg);
                EXPECT_LE(profile[i].pressurePa, profile[i - 1].pressurePa + 1.0e-4 * centralPressurePa);

                // Density decreases monotonically outward within each material layer
                if (profile[i].material == profile[i - 1].material) {
                    EXPECT_LE(profile[i].rhoKgM3, profile[i - 1].rhoKgM3 * (1.0 + 1.0e-5))
                        << "Non-monotonic density in archetype " << static_cast<int>(arch)
                        << " at sample " << i;
                }
            }
        }

        EXPECT_TRUE(materialsSeen.count(MaterialId::Iron) > 0);
        EXPECT_TRUE(materialsSeen.count(MaterialId::Granite) > 0);
        if (arch == PlanetArchetype::LivingTerra || arch == PlanetArchetype::OceanIce) {
            EXPECT_TRUE(materialsSeen.count(MaterialId::WaterIce) > 0);
        } else if (arch == PlanetArchetype::GasGiant) {
            EXPECT_TRUE(materialsSeen.count(MaterialId::HydrogenHelium) > 0);
        } else if (arch == PlanetArchetype::ProtoTheia) {
            EXPECT_TRUE(materialsSeen.count(MaterialId::Regolith) > 0);
        }

        // Build 3D particle realization
        const uint32_t kN = 6000;
        const uint32_t kGrid = 8;
        const BuiltPlanetHost built = PlanetBuilder::buildPlanet(spec, kN, 0, kGrid, 42);
        ASSERT_EQ(built.posX.size(), kN);
        ASSERT_EQ(built.posY.size(), kN);
        ASSERT_EQ(built.posZ.size(), kN);
        ASSERT_EQ(built.velX.size(), kN);
        ASSERT_EQ(built.velY.size(), kN);
        ASSERT_EQ(built.velZ.size(), kN);
        ASSERT_EQ(built.mass.size(), kN);
        ASSERT_EQ(built.h0.size(), kN);
        ASSERT_EQ(built.rho.size(), kN);
        ASSERT_EQ(built.u.size(), kN);
        ASSERT_EQ(built.tempK.size(), kN);
        ASSERT_EQ(built.epsFail.size(), kN);
        ASSERT_EQ(built.materialId.size(), kN);
        ASSERT_EQ(built.bodyId.size(), kN);
        ASSERT_EQ(built.tracerIndex.size(), kN);

        double totalMassPsu = 0.0;
        for (uint32_t i = 0; i < kN; ++i) {
            EXPECT_TRUE(std::isfinite(built.posX[i]));
            EXPECT_TRUE(std::isfinite(built.posY[i]));
            EXPECT_TRUE(std::isfinite(built.posZ[i]));
            EXPECT_TRUE(std::isfinite(built.mass[i]));
            EXPECT_TRUE(std::isfinite(built.h0[i]));
            EXPECT_TRUE(std::isfinite(built.rho[i]));
            EXPECT_TRUE(std::isfinite(built.u[i]));
            EXPECT_TRUE(std::isfinite(built.tempK[i]));
            EXPECT_GT(built.mass[i], 0.0f);
            EXPECT_GT(built.h0[i], 0.0f);
            EXPECT_GT(built.rho[i], 0.0f);
            EXPECT_GT(built.tempK[i], 0.0f);
            EXPECT_EQ(built.bodyId[i], 0u);
            totalMassPsu += static_cast<double>(built.mass[i]);
        }

        EXPECT_NEAR(totalMassPsu * units::M_UNIT_KG, spec.massKg, 0.02 * spec.massKg);
        EXPECT_NEAR(built.actualMassKg, spec.massKg, 0.02 * spec.massKg);
        EXPECT_GT(built.equatorialRadiusM, 1.0e5);

        // Verify radial density trend in 3D particles within each material
        std::map<uint8_t, std::vector<std::pair<double, float>>> byMat;
        for (uint32_t i = 0; i < kN; ++i) {
            const double r = std::sqrt(
                static_cast<double>(built.posX[i]) * built.posX[i] +
                static_cast<double>(built.posY[i]) * built.posY[i] +
                static_cast<double>(built.posZ[i]) * built.posZ[i]);
            byMat[built.materialId[i]].push_back({r, built.rho[i]});
        }
        for (auto& [mat, vec] : byMat) {
            if (vec.size() < 20) continue;
            std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            const size_t q = vec.size() / 4;
            double innerAvg = 0.0, outerAvg = 0.0;
            for (size_t k = 0; k < q; ++k) innerAvg += vec[k].second;
            for (size_t k = vec.size() - q; k < vec.size(); ++k) outerAvg += vec[k].second;
            innerAvg /= static_cast<double>(q);
            outerAvg /= static_cast<double>(q);
            EXPECT_GE(innerAvg, outerAvg * 0.99);
        }

        // Non-LivingTerra bodies have zero tracers
        if (arch != PlanetArchetype::LivingTerra) {
            EXPECT_EQ(built.tracerCount, 0u);
            for (int32_t idx : built.tracerIndex) {
                EXPECT_EQ(idx, -1);
            }
        }
    }

    // Determinism check for fixed seed = 42
    const BodyInitSpec terraSpec = defaultSpecForArchetype(PlanetArchetype::LivingTerra);
    const BuiltPlanetHost a = PlanetBuilder::buildPlanet(terraSpec, 4000, 0, 8, 42);
    const BuiltPlanetHost b = PlanetBuilder::buildPlanet(terraSpec, 4000, 0, 8, 42);
    EXPECT_EQ(a.posX, b.posX);
    EXPECT_EQ(a.posY, b.posY);
    EXPECT_EQ(a.posZ, b.posZ);
    EXPECT_EQ(a.rho, b.rho);
    EXPECT_EQ(a.u, b.u);
    EXPECT_EQ(a.tracerIndex, b.tracerIndex);
}

TEST(PlanetBuilderTest, LivingTerraTracerGridUniqueness6xG2) {
    const BodyInitSpec terraSpec = defaultSpecForArchetype(PlanetArchetype::LivingTerra);
    for (uint32_t gridRes : {8u, 16u}) {
        const uint32_t particleCount = 20000;
        const BuiltPlanetHost built = PlanetBuilder::buildPlanet(terraSpec, particleCount, 0, gridRes, 42);
        const uint32_t expectedTracers = 6u * gridRes * gridRes;
        ASSERT_EQ(built.tracerCount, expectedTracers);
        ASSERT_EQ(built.tracerIndex.size(), particleCount);
        ASSERT_EQ(built.tracerRestRadiusPsu.size(), particleCount);
        ASSERT_EQ(built.tracerRestNormalX.size(), particleCount);
        ASSERT_EQ(built.tracerRestNormalY.size(), particleCount);
        ASSERT_EQ(built.tracerRestNormalZ.size(), particleCount);
        ASSERT_EQ(built.tracerIsOcean.size(), particleCount);

        std::vector<uint32_t> seenCount(expectedTracers, 0u);
        uint32_t oceanTracers = 0;
        uint32_t landTracers = 0;
        for (uint32_t i = 0; i < particleCount; ++i) {
            const int32_t tIdx = built.tracerIndex[i];
            if (tIdx < 0) {
                EXPECT_EQ(tIdx, -1);
                continue;
            }
            ASSERT_LT(static_cast<uint32_t>(tIdx), expectedTracers);
            seenCount[static_cast<uint32_t>(tIdx)]++;
            EXPECT_GT(built.tracerRestRadiusPsu[i], 0.0f);
            const float nLen = std::sqrt(
                built.tracerRestNormalX[i] * built.tracerRestNormalX[i] +
                built.tracerRestNormalY[i] * built.tracerRestNormalY[i] +
                built.tracerRestNormalZ[i] * built.tracerRestNormalZ[i]);
            EXPECT_NEAR(nLen, 1.0f, 1e-4f);
            EXPECT_LE(built.tracerIsOcean[i], 1u);
            if (built.tracerIsOcean[i] != 0u) {
                ++oceanTracers;
            } else {
                ++landTracers;
            }
        }
        for (uint32_t t = 0; t < expectedTracers; ++t) {
            EXPECT_EQ(seenCount[t], 1u) << "Tracer index " << t << " missing or duplicated for G=" << gridRes;
        }
        EXPECT_GT(landTracers, 0u);
        EXPECT_GT(oceanTracers + landTracers, 0u);
    }
}

TEST(PlanetBuilderTest, RelaxBodyConvergesBelowOnePercentVesc_AC2) {
    const std::array<PlanetArchetype, 6> archetypes = {
        PlanetArchetype::LivingTerra,
        PlanetArchetype::IronCore,
        PlanetArchetype::OceanIce,
        PlanetArchetype::LavaIo,
        PlanetArchetype::GasGiant,
        PlanetArchetype::ProtoTheia,
    };

    Simulation sim(12000);
    for (PlanetArchetype arch : archetypes) {
        CollisionScenarioConfig cfg;
        cfg.presetName = "RelaxCheck";
        cfg.target = defaultSpecForArchetype(arch);
        cfg.target.relaxSteps = 0;
        cfg.impactor = defaultSpecForArchetype(PlanetArchetype::ProtoTheia);
        cfg.impactor.massKg = 0.25 * cfg.target.massKg;
        cfg.impactor.relaxSteps = 0;
        cfg.impactVelocityRatioVesc = 1.0;
        cfg.impactParamB = 0.0;
        cfg.initialSepRadii = 10.0;
        cfg.totalParticles = 8000;
        cfg.tracerGridRes = 8;
        cfg.randomSeed = 42;

        sim.loadScenario(cfg);
        double vrmsOverVesc = 1.0;
        sim.relaxBody(0, 80, vrmsOverVesc);
        EXPECT_TRUE(std::isfinite(vrmsOverVesc));
        EXPECT_LT(vrmsOverVesc, 0.01) << "Relaxation failed for archetype " << static_cast<int>(arch);
    }
}

TEST(PlanetBuilderTest, AllFivePresetsAndInvalidPresetThrows) {
    struct ExpectedPreset {
        ScenarioPresetId id;
        const char* name;
        PlanetArchetype targetArch;
        double targetMassKg;
        PlanetArchetype impactorArch;
        double impactorMassKg;
        double vRatio;
        double b;
        double sep;
    };

    const std::array<ExpectedPreset, 5> expected = {{
        {ScenarioPresetId::EarthTheiaMoonForming, "EarthTheiaMoonForming",
         PlanetArchetype::LivingTerra, 1.0 * units::M_EARTH_SI,
         PlanetArchetype::ProtoTheia, 0.13 * units::M_EARTH_SI, 1.05, 0.72, 3.0},
        {ScenarioPresetId::MercuryMantleStrip, "MercuryMantleStrip",
         PlanetArchetype::ProtoTheia, 0.45 * units::M_EARTH_SI,
         PlanetArchetype::IronCore, 0.22 * units::M_EARTH_SI, 2.80, 0.35, 3.2},
        {ScenarioPresetId::OceanWorldSteamPlume, "OceanWorldSteamPlume",
         PlanetArchetype::OceanIce, 0.75 * units::M_EARTH_SI,
         PlanetArchetype::ProtoTheia, 0.15 * units::M_EARTH_SI, 1.35, 0.25, 3.2},
        {ScenarioPresetId::GasGiantCoreImpact, "GasGiantCoreImpact",
         PlanetArchetype::GasGiant, 20.0 * units::M_EARTH_SI,
         PlanetArchetype::IronCore, 1.0 * units::M_EARTH_SI, 1.40, 0.15, 2.8},
        {ScenarioPresetId::LivingGaiaAsteroidCataclysm, "LivingGaiaAsteroidCataclysm",
         PlanetArchetype::LivingTerra, 1.0 * units::M_EARTH_SI,
         PlanetArchetype::ProtoTheia, 0.025 * units::M_EARTH_SI, 1.65, 0.30, 3.2},
    }};

    for (const auto& e : expected) {
        const std::string s = presetToString(e.id);
        EXPECT_EQ(s, e.name);
        EXPECT_EQ(parsePresetName(s), e.id);

        const CollisionScenarioConfig cfg = makePresetConfig(e.id, 50000);
        EXPECT_EQ(cfg.presetName, e.name);
        EXPECT_EQ(cfg.totalParticles, 50000u);
        EXPECT_EQ(cfg.randomSeed, 42u);
        EXPECT_EQ(cfg.target.archetype, e.targetArch);
        EXPECT_NEAR(cfg.target.massKg, e.targetMassKg, 1e-4 * e.targetMassKg);
        EXPECT_EQ(cfg.impactor.archetype, e.impactorArch);
        EXPECT_NEAR(cfg.impactor.massKg, e.impactorMassKg, 1e-4 * e.impactorMassKg);
        EXPECT_DOUBLE_EQ(cfg.impactVelocityRatioVesc, e.vRatio);
        EXPECT_DOUBLE_EQ(cfg.impactParamB, e.b);
        EXPECT_DOUBLE_EQ(cfg.initialSepRadii, e.sep);

        if (e.id == ScenarioPresetId::MercuryMantleStrip) {
            EXPECT_DOUBLE_EQ(cfg.target.surfaceTempK, 440.0);
        }
    }

    EXPECT_THROW(parsePresetName("UnknownPreset"), std::invalid_argument);
    EXPECT_THROW(parsePresetName(""), std::invalid_argument);
    EXPECT_THROW(parsePresetName("earth_theia"), std::invalid_argument);
}

} // namespace
