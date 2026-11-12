#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "eos.hpp"
#include "simulation.hpp"
#include "types.hpp"
#include "units.hpp"

namespace {

using namespace astraea;

constexpr std::array<MaterialId, 5> kAllMaterials = {
    MaterialId::Iron,
    MaterialId::Granite,
    MaterialId::WaterIce,
    MaterialId::HydrogenHelium,
    MaterialId::Regolith,
};

double analyticalColdEnergySI(MaterialId mat, double rhoSI) {
    const auto& p = eos::getParams(mat);
    if (mat == MaterialId::HydrogenHelium) {
        return p.kH2_Poly * rhoSI;
    }
    const double eta = rhoSI / p.rho0KgM3;
    if (eta <= 1.0) {
        return 0.0;
    }
    const double lnEta = std::log(eta);
    const double invEta = 1.0 / eta;
    return (p.A_Pa * (lnEta + invEta - 1.0) + p.B_Pa * (eta - invEta - 2.0 * lnEta)) / p.rho0KgM3;
}

double analyticalTillotsonPcSI(const eos::TillotsonParams& p, double rhoSI, double uEosSI) {
    const double eta = rhoSI / p.rho0KgM3;
    const double mu = eta - 1.0;
    const double bEff = (mu >= 0.0) ? p.B_Pa : 0.0;
    const double denom = (uEosSI / (p.E0_JKg * eta * eta)) + 1.0;
    return (p.a + p.b / denom) * rhoSI * uEosSI + p.A_Pa * mu + bEff * mu * mu;
}

double analyticalTillotsonPeSI(const eos::TillotsonParams& p, double rhoSI, double uEosSI) {
    const double eta = rhoSI / p.rho0KgM3;
    const double mu = eta - 1.0;
    const double x = (p.rho0KgM3 / rhoSI) - 1.0;
    const double denom = (uEosSI / (p.E0_JKg * eta * eta)) + 1.0;
    return p.a * rhoSI * uEosSI +
           ((p.b * rhoSI * uEosSI / denom) + p.A_Pa * mu * std::exp(-p.beta * x)) *
               std::exp(-p.alpha * x * x);
}

TEST(EosTest, UnitsConversionConstantsConsistent) {
    EXPECT_DOUBLE_EQ(units::L_UNIT_M, 1.0e6);
    EXPECT_DOUBLE_EQ(units::M_UNIT_KG, 1.0e24);
    EXPECT_DOUBLE_EQ(units::T_UNIT_S, 1.0e3);
    EXPECT_DOUBLE_EQ(units::V_UNIT_MS, 1.0e3);
    EXPECT_DOUBLE_EQ(units::A_UNIT_MS2, 1.0);
    EXPECT_DOUBLE_EQ(units::RHO_UNIT_KGM3, 1.0e6);
    EXPECT_DOUBLE_EQ(units::P_UNIT_PA, 1.0e12);
    EXPECT_DOUBLE_EQ(units::U_UNIT_JKG, 1.0e6);
    EXPECT_DOUBLE_EQ(units::E_UNIT_J, 1.0e30);
    EXPECT_DOUBLE_EQ(units::L_ANG_UNIT, 1.0e33);
    const double expectedGPsu =
        units::G_SI * units::M_UNIT_KG * units::T_UNIT_S * units::T_UNIT_S /
        (units::L_UNIT_M * units::L_UNIT_M * units::L_UNIT_M);
    EXPECT_NEAR(units::G_PSU, expectedGPsu, 1e-12);
    EXPECT_NEAR(static_cast<double>(units::G_PSU_F), expectedGPsu, 1e-4);
}

TEST(EosTest, ParameterTableMatchesPublishedConstants) {
    {
        const auto& p = eos::getParams(MaterialId::Iron);
        EXPECT_DOUBLE_EQ(p.rho0KgM3, 7800.0);
        EXPECT_DOUBLE_EQ(p.a, 0.5);
        EXPECT_DOUBLE_EQ(p.b, 1.5);
        EXPECT_DOUBLE_EQ(p.A_Pa, 1.28e11);
        EXPECT_DOUBLE_EQ(p.B_Pa, 1.05e11);
        EXPECT_DOUBLE_EQ(p.E0_JKg, 9.50e6);
        EXPECT_DOUBLE_EQ(p.Eiv_JKg, 2.40e6);
        EXPECT_DOUBLE_EQ(p.Ecv_JKg, 8.67e6);
        EXPECT_DOUBLE_EQ(p.alpha, 5.0);
        EXPECT_DOUBLE_EQ(p.beta, 5.0);
        EXPECT_DOUBLE_EQ(p.cv_JKgK, 449.0);
        EXPECT_DOUBLE_EQ(p.tMelt0K, 1811.0);
        EXPECT_DOUBLE_EQ(p.tRefK, 293.0);
        EXPECT_DOUBLE_EQ(p.muShear_Pa, 7.70e10);
        EXPECT_DOUBLE_EQ(p.yieldY0_Pa, 6.80e8);
        EXPECT_DOUBLE_EQ(p.epsFailMin, 1.0e-3);
    }
    {
        const auto& p = eos::getParams(MaterialId::Granite);
        EXPECT_DOUBLE_EQ(p.rho0KgM3, 2680.0);
        EXPECT_DOUBLE_EQ(p.a, 0.5);
        EXPECT_DOUBLE_EQ(p.b, 1.3);
        EXPECT_DOUBLE_EQ(p.A_Pa, 1.80e10);
        EXPECT_DOUBLE_EQ(p.B_Pa, 1.80e10);
        EXPECT_DOUBLE_EQ(p.E0_JKg, 1.60e7);
        EXPECT_DOUBLE_EQ(p.Eiv_JKg, 3.50e6);
        EXPECT_DOUBLE_EQ(p.Ecv_JKg, 1.80e7);
        EXPECT_DOUBLE_EQ(p.alpha, 5.0);
        EXPECT_DOUBLE_EQ(p.beta, 5.0);
        EXPECT_DOUBLE_EQ(p.cv_JKgK, 790.0);
        EXPECT_DOUBLE_EQ(p.tMelt0K, 1600.0);
        EXPECT_DOUBLE_EQ(p.tRefK, 293.0);
        EXPECT_DOUBLE_EQ(p.muShear_Pa, 2.20e10);
        EXPECT_DOUBLE_EQ(p.yieldY0_Pa, 3.50e9);
        EXPECT_DOUBLE_EQ(p.epsFailMin, 1.0e-4);
    }
    {
        const auto& p = eos::getParams(MaterialId::WaterIce);
        EXPECT_DOUBLE_EQ(p.rho0KgM3, 1000.0);
        EXPECT_DOUBLE_EQ(p.a, 0.7);
        EXPECT_DOUBLE_EQ(p.b, 0.15);
        EXPECT_DOUBLE_EQ(p.A_Pa, 2.18e9);
        EXPECT_DOUBLE_EQ(p.B_Pa, 1.325e10);
        EXPECT_DOUBLE_EQ(p.E0_JKg, 7.00e6);
        EXPECT_DOUBLE_EQ(p.Eiv_JKg, 4.19e5);
        EXPECT_DOUBLE_EQ(p.Ecv_JKg, 2.68e6);
        EXPECT_DOUBLE_EQ(p.alpha, 10.0);
        EXPECT_DOUBLE_EQ(p.beta, 5.0);
        EXPECT_DOUBLE_EQ(p.cv_JKgK, 4184.0);
        EXPECT_DOUBLE_EQ(p.tMelt0K, 273.15);
        EXPECT_DOUBLE_EQ(p.tRefK, 273.15);
        EXPECT_DOUBLE_EQ(p.muShear_Pa, 3.50e9);
        EXPECT_DOUBLE_EQ(p.yieldY0_Pa, 1.00e7);
        EXPECT_DOUBLE_EQ(p.epsFailMin, 5.0e-4);
    }
    {
        const auto& p = eos::getParams(MaterialId::HydrogenHelium);
        EXPECT_DOUBLE_EQ(p.rho0KgM3, 0.178);
        EXPECT_DOUBLE_EQ(p.cv_JKgK, 8260.0);
        EXPECT_DOUBLE_EQ(p.tMelt0K, 0.0);
        EXPECT_DOUBLE_EQ(p.tRefK, 0.0);
        EXPECT_DOUBLE_EQ(p.kH2_Poly, 2.08e5);
        EXPECT_DOUBLE_EQ(p.gammaIdeal, 1.4);
    }
    {
        const auto& p = eos::getParams(MaterialId::Regolith);
        EXPECT_DOUBLE_EQ(p.rho0KgM3, 2000.0);
        EXPECT_DOUBLE_EQ(p.a, 0.5);
        EXPECT_DOUBLE_EQ(p.b, 1.3);
        EXPECT_DOUBLE_EQ(p.A_Pa, 1.20e10);
        EXPECT_DOUBLE_EQ(p.B_Pa, 1.20e10);
        EXPECT_DOUBLE_EQ(p.E0_JKg, 1.60e7);
        EXPECT_DOUBLE_EQ(p.Eiv_JKg, 3.50e6);
        EXPECT_DOUBLE_EQ(p.Ecv_JKg, 1.80e7);
        EXPECT_DOUBLE_EQ(p.alpha, 5.0);
        EXPECT_DOUBLE_EQ(p.beta, 5.0);
        EXPECT_DOUBLE_EQ(p.cv_JKgK, 800.0);
        EXPECT_DOUBLE_EQ(p.tMelt0K, 1500.0);
        EXPECT_DOUBLE_EQ(p.tRefK, 293.0);
        EXPECT_DOUBLE_EQ(p.muShear_Pa, 8.00e9);
        EXPECT_DOUBLE_EQ(p.yieldY0_Pa, 5.00e7);
        EXPECT_DOUBLE_EQ(p.epsFailMin, 5.0e-5);
    }
}

TEST(EosTest, ColdCompressionEnergyIsMonotonicAndExact) {
    for (MaterialId mat : kAllMaterials) {
        const auto& p = eos::getParams(mat);
        if (mat == MaterialId::HydrogenHelium) {
            double prev = -1.0;
            for (double rhoSI : {0.178, 10.0, 100.0, 1000.0, 5000.0}) {
                const double uColdPsu = eos::computeColdEnergyPsu(mat, rhoSI / units::RHO_UNIT_KGM3);
                const double expectedPsu = analyticalColdEnergySI(mat, rhoSI) / units::U_UNIT_JKG;
                EXPECT_GT(uColdPsu, prev);
                EXPECT_NEAR(uColdPsu, expectedPsu, 1e-10 * expectedPsu + 1e-12);
                prev = uColdPsu;
            }
        } else {
            for (double eta : {0.5, 0.9, 1.0}) {
                const double rhoPsu = (eta * p.rho0KgM3) / units::RHO_UNIT_KGM3;
                EXPECT_DOUBLE_EQ(eos::computeColdEnergyPsu(mat, rhoPsu), 0.0);
            }
            double prev = 0.0;
            for (double eta : {1.05, 1.2, 1.5, 2.0, 3.0}) {
                const double rhoSI = eta * p.rho0KgM3;
                const double uColdPsu = eos::computeColdEnergyPsu(mat, rhoSI / units::RHO_UNIT_KGM3);
                const double expectedPsu = analyticalColdEnergySI(mat, rhoSI) / units::U_UNIT_JKG;
                EXPECT_GT(uColdPsu, prev);
                EXPECT_NEAR(uColdPsu, expectedPsu, 1e-10 * expectedPsu + 1e-12);
                prev = uColdPsu;
            }
        }
    }
}

TEST(EosTest, AllMaterialsRegimesAndDamagedTension_AC1) {
    for (MaterialId mat : kAllMaterials) {
        const auto& p = eos::getParams(mat);
        const double rho0Psu = p.rho0KgM3 / units::RHO_UNIT_KGM3;

        // 1) Compressed (eta = 1.5), including analytical P_c and D = 1.0 preservation
        {
            const float rhoPsu = static_cast<float>(1.5 * rho0Psu);
            const double rhoSI = static_cast<double>(rhoPsu) * units::RHO_UNIT_KGM3;
            const double uColdPsu = eos::computeColdEnergyPsu(mat, rhoPsu);
            const double uEosSI = uColdPsu * units::U_UNIT_JKG + p.cv_JKgK * 1000.0;
            const double uSi = uColdPsu * units::U_UNIT_JKG + p.cv_JKgK * (p.tRefK + 1000.0);
            const float uPsu = static_cast<float>(uSi / units::U_UNIT_JKG);
            const auto res = eos::evaluateEosPsu(mat, rhoPsu, uPsu, 0.0f);
            EXPECT_TRUE(std::isfinite(res.pressurePsu));
            EXPECT_TRUE(std::isfinite(res.soundSpeedPsu));
            EXPECT_TRUE(std::isfinite(res.tempK));
            EXPECT_GT(res.soundSpeedPsu, 0.0f);
            EXPECT_GE(res.tempK, 3.0f);
            EXPECT_GT(res.pressurePsu, 0.0f);

            if (mat != MaterialId::HydrogenHelium) {
                const double expectedPSi = analyticalTillotsonPcSI(p, rhoSI, uEosSI);
                const double actualPSi = static_cast<double>(res.pressurePsu) * units::P_UNIT_PA;
                EXPECT_NEAR(actualPSi, expectedPSi, 1e-4 * std::abs(expectedPSi) + 1e3);
            }

            // Damage D=1.0 must NOT attenuate positive compressive pressure
            const auto resDamaged = eos::evaluateEosPsu(mat, rhoPsu, uPsu, 1.0f);
            EXPECT_GT(resDamaged.pressurePsu, 0.0f);
            EXPECT_NEAR(resDamaged.pressurePsu, res.pressurePsu,
                        1e-5f * std::abs(res.pressurePsu) + 1e-12f);
        }

        // 2) Two-phase transition (Eiv <= u_EOS < Ecv, eta = 0.7)
        {
            const float rhoPsu = static_cast<float>(0.7 * rho0Psu);
            const double rhoSI = static_cast<double>(rhoPsu) * units::RHO_UNIT_KGM3;
            if (mat != MaterialId::HydrogenHelium) {
                const double uEosSI = 0.5 * (p.Eiv_JKg + p.Ecv_JKg);
                const double uTotalSI = p.cv_JKgK * p.tRefK + uEosSI;
                const float uPsu = static_cast<float>(uTotalSI / units::U_UNIT_JKG);
                const auto res = eos::evaluateEosPsu(mat, rhoPsu, uPsu, 0.0f);

                const double pc = analyticalTillotsonPcSI(p, rhoSI, uEosSI);
                const double pe = analyticalTillotsonPeSI(p, rhoSI, uEosSI);
                const double expectedPSi =
                    ((uEosSI - p.Eiv_JKg) * pe + (p.Ecv_JKg - uEosSI) * pc) / (p.Ecv_JKg - p.Eiv_JKg);
                const double actualPSi = static_cast<double>(res.pressurePsu) * units::P_UNIT_PA;

                EXPECT_TRUE(std::isfinite(res.pressurePsu));
                EXPECT_TRUE(std::isfinite(res.soundSpeedPsu));
                EXPECT_GT(res.soundSpeedPsu, 0.0f);
                EXPECT_GE(res.tempK, 3.0f);
                EXPECT_NEAR(actualPSi, expectedPSi, 1e-4 * std::abs(expectedPSi) + 1e3);
            }
        }

        // 3) Hot expanded (eta = 0.2, u_EOS = 2 * Ecv)
        {
            const float rhoPsu = static_cast<float>(0.2 * rho0Psu);
            const double rhoSI = static_cast<double>(rhoPsu) * units::RHO_UNIT_KGM3;
            const double uEosSI = (mat == MaterialId::HydrogenHelium) ? 5.0e7 : (2.0 * p.Ecv_JKg);
            const double uTotalSI =
                (mat == MaterialId::HydrogenHelium) ? (p.kH2_Poly * rhoSI + uEosSI) : (p.cv_JKgK * p.tRefK + uEosSI);
            const float uPsu = static_cast<float>(uTotalSI / units::U_UNIT_JKG);
            const auto res = eos::evaluateEosPsu(mat, rhoPsu, uPsu, 0.0f);

            EXPECT_TRUE(std::isfinite(res.pressurePsu));
            EXPECT_TRUE(std::isfinite(res.soundSpeedPsu));
            EXPECT_GT(res.soundSpeedPsu, 0.0f);
            EXPECT_GE(res.tempK, 3.0f);
            EXPECT_GT(res.pressurePsu, 0.0f);

            if (mat != MaterialId::HydrogenHelium) {
                const double expectedPSi = analyticalTillotsonPeSI(p, rhoSI, uEosSI);
                const double actualPSi = static_cast<double>(res.pressurePsu) * units::P_UNIT_PA;
                EXPECT_NEAR(actualPSi, expectedPSi, 1e-4 * std::abs(expectedPSi) + 1e3);
            }
        }

        // 4) Damaged tensile state (eta = 0.9, D = 1.0)
        {
            const float rhoPsu = static_cast<float>(0.9 * rho0Psu);
            const double uTotalSI = p.cv_JKgK * p.tRefK;
            const float uPsu = static_cast<float>(uTotalSI / units::U_UNIT_JKG);
            const auto intact = eos::evaluateEosPsu(mat, rhoPsu, uPsu, 0.0f);
            const auto damaged = eos::evaluateEosPsu(mat, rhoPsu, uPsu, 1.0f);

            EXPECT_TRUE(std::isfinite(damaged.pressurePsu));
            EXPECT_TRUE(std::isfinite(damaged.soundSpeedPsu));
            EXPECT_GT(damaged.soundSpeedPsu, 0.0f);
            EXPECT_GE(damaged.tempK, 3.0f);

            if (mat != MaterialId::HydrogenHelium) {
                EXPECT_LT(intact.pressurePsu, 0.0f);
                EXPECT_NEAR(damaged.pressurePsu, 0.0f, 1e-12f);

                const auto halfDamaged = eos::evaluateEosPsu(mat, rhoPsu, uPsu, 0.5f);
                EXPECT_NEAR(halfDamaged.pressurePsu, 0.5f * intact.pressurePsu,
                            1e-5f * std::abs(intact.pressurePsu) + 1e-12f);
            }
        }
    }
}

TEST(EosTest, HydrogenHeliumExactPressureAndSoundSpeedIdentity_AC1) {
    const auto& p = eos::getParams(MaterialId::HydrogenHelium);
    struct Case {
        double rhoSI;
        double uThSI;
    };
    const std::vector<Case> cases = {
        {0.178, 2.0e6},
        {50.0, 0.0},
        {200.0, 5.0e6},
        {1200.0, 2.5e7},
        {3500.0, 6.0e7},
    };

    for (const auto& c : cases) {
        const double uColdSI = p.kH2_Poly * c.rhoSI;
        const double uTotalSI = uColdSI + c.uThSI;
        const float rhoPsu = static_cast<float>(c.rhoSI / units::RHO_UNIT_KGM3);
        const float uPsu = static_cast<float>(uTotalSI / units::U_UNIT_JKG);

        const auto res = eos::evaluateEosPsu(MaterialId::HydrogenHelium, rhoPsu, uPsu, 0.0f);
        const double actualPSi = static_cast<double>(res.pressurePsu) * units::P_UNIT_PA;
        const double expectedPSi = p.kH2_Poly * c.rhoSI * c.rhoSI + (p.gammaIdeal - 1.0) * c.rhoSI * c.uThSI;
        EXPECT_NEAR(actualPSi, expectedPSi, 1e-4 * expectedPSi + 1.0);

        const double cs2ExactSI =
            2.0 * p.kH2_Poly * c.rhoSI + p.gammaIdeal * (p.gammaIdeal - 1.0) * c.uThSI;
        const double csExactPsu = std::sqrt(cs2ExactSI) / units::V_UNIT_MS;
        const double expectedCsPsu = std::max(0.05, csExactPsu);
        EXPECT_NEAR(static_cast<double>(res.soundSpeedPsu), expectedCsPsu, 1e-4 * expectedCsPsu + 1e-6);
        EXPECT_GE(res.soundSpeedPsu, 0.05f);
    }

    // Low density & zero thermal energy enforces sound speed floor >= 0.05 km/s
    const double rhoLowSI = 0.001;
    const float rhoLowPsu = static_cast<float>(rhoLowSI / units::RHO_UNIT_KGM3);
    const float uLowPsu = static_cast<float>((p.kH2_Poly * rhoLowSI) / units::U_UNIT_JKG);
    const auto lowRes = eos::evaluateEosPsu(MaterialId::HydrogenHelium, rhoLowPsu, uLowPsu, 0.0f);
    EXPECT_GE(lowRes.soundSpeedPsu, 0.05f - 1e-6f);
}

TEST(EosTest, WaterOceanZeroPressureEquilibriumNear288KAndTRefK) {
    const double rhoEq = eos::invertDensityFromPressureTempSI(MaterialId::WaterIce, 0.0, 288.0, 1000.0);
    EXPECT_TRUE(std::isfinite(rhoEq));
    EXPECT_GE(rhoEq, 940.0);
    EXPECT_LE(rhoEq, 1005.0);

    double uSi = 0.0;
    double pSi = 0.0;
    eos::evaluateFromRhoTempSI(MaterialId::WaterIce, rhoEq, 288.0, uSi, pSi);
    EXPECT_NEAR(pSi, 0.0, 1.0e4);

    const float rhoPsu = static_cast<float>(rhoEq / units::RHO_UNIT_KGM3);
    const float uPsu = static_cast<float>(uSi / units::U_UNIT_JKG);
    const auto res = eos::evaluateEosPsu(MaterialId::WaterIce, rhoPsu, uPsu, 0.0f);
    EXPECT_EQ(res.phaseId, static_cast<uint8_t>(PhaseId::LiquidWater));
    EXPECT_NEAR(res.tempK, 288.0f, 0.5f);

    // STP reference states at tRefK yield zero pressure at rho0
    for (MaterialId mat : {MaterialId::Iron, MaterialId::Granite, MaterialId::WaterIce, MaterialId::Regolith}) {
        const auto& params = eos::getParams(mat);
        double uStp = 0.0;
        double pStp = 0.0;
        eos::evaluateFromRhoTempSI(mat, params.rho0KgM3, params.tRefK, uStp, pStp);
        EXPECT_NEAR(pStp, 0.0, 1.0);
    }
}

TEST(EosTest, PhaseClassificationBranches) {
    // Granite: Solid, Molten, Vapor, Plasma
    {
        const auto& g = eos::getParams(MaterialId::Granite);
        const float rho0 = static_cast<float>(g.rho0KgM3 / units::RHO_UNIT_KGM3);
        const auto solid = eos::evaluateEosPsu(
            MaterialId::Granite, rho0, static_cast<float>((g.cv_JKgK * 300.0) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(solid.phaseId, static_cast<uint8_t>(PhaseId::Solid));

        const auto molten = eos::evaluateEosPsu(
            MaterialId::Granite, rho0, static_cast<float>((g.cv_JKgK * 2200.0) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(molten.phaseId, static_cast<uint8_t>(PhaseId::Molten));

        const auto highEnergy = eos::evaluateEosPsu(
            MaterialId::Granite, 0.5f * rho0,
            static_cast<float>((g.cv_JKgK * g.tRefK + 1.2 * g.Ecv_JKg) / units::U_UNIT_JKG), 0.0f);
        EXPECT_GT(highEnergy.tempK, 12000.0f);
        EXPECT_TRUE(highEnergy.phaseId == static_cast<uint8_t>(PhaseId::Plasma) ||
                    highEnergy.phaseId == static_cast<uint8_t>(PhaseId::Vapor));

        const auto plasma = eos::evaluateEosPsu(
            MaterialId::Granite, rho0, static_cast<float>((g.cv_JKgK * 15000.0) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(plasma.phaseId, static_cast<uint8_t>(PhaseId::Plasma));
    }

    // WaterIce: Solid, LiquidWater, Vapor, Plasma
    {
        const auto& w = eos::getParams(MaterialId::WaterIce);
        const float rho0 = static_cast<float>(w.rho0KgM3 / units::RHO_UNIT_KGM3);
        const auto ice = eos::evaluateEosPsu(
            MaterialId::WaterIce, rho0, static_cast<float>((w.cv_JKgK * 200.0) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(ice.phaseId, static_cast<uint8_t>(PhaseId::Solid));

        const auto liquid = eos::evaluateEosPsu(
            MaterialId::WaterIce, rho0, static_cast<float>((w.cv_JKgK * 300.0) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(liquid.phaseId, static_cast<uint8_t>(PhaseId::LiquidWater));

        const auto vapor = eos::evaluateEosPsu(
            MaterialId::WaterIce, 0.1f * rho0,
            static_cast<float>((w.cv_JKgK * 1000.0 + w.Ecv_JKg) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(vapor.phaseId, static_cast<uint8_t>(PhaseId::Vapor));

        const auto plasma = eos::evaluateEosPsu(
            MaterialId::WaterIce, rho0, static_cast<float>((w.cv_JKgK * 18000.0) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(plasma.phaseId, static_cast<uint8_t>(PhaseId::Plasma));
    }

    // HydrogenHelium: Gas, Plasma
    {
        const auto& h = eos::getParams(MaterialId::HydrogenHelium);
        const float rho = static_cast<float>(100.0 / units::RHO_UNIT_KGM3);
        const double uCold = eos::computeColdEnergyPsu(MaterialId::HydrogenHelium, rho);
        const auto gas = eos::evaluateEosPsu(
            MaterialId::HydrogenHelium, rho, static_cast<float>(uCold + (h.cv_JKgK * 500.0) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(gas.phaseId, static_cast<uint8_t>(PhaseId::Gas));

        const auto plasma = eos::evaluateEosPsu(
            MaterialId::HydrogenHelium, rho, static_cast<float>(uCold + (h.cv_JKgK * 12000.0) / units::U_UNIT_JKG), 0.0f);
        EXPECT_EQ(plasma.phaseId, static_cast<uint8_t>(PhaseId::Plasma));
    }
}

TEST(EosTest, TemperatureFloorAndRadiativeEnergyTracking_AC7) {
    // Temperature clamp >= 3.0 K under cold/zero internal energy
    for (MaterialId mat : kAllMaterials) {
        const auto& p = eos::getParams(mat);
        const float rhoPsu = static_cast<float>((1.4 * p.rho0KgM3) / units::RHO_UNIT_KGM3);
        const auto res = eos::evaluateEosPsu(mat, rhoPsu, 0.0f, 0.0f);
        EXPECT_GE(res.tempK, 3.0f);
    }

    // Radiative energy loss is positive & tracked in Simulation telemetry on hot sphere
    CollisionScenarioConfig cfg;
    cfg.presetName = "HotSphereRadiativeCheck";
    cfg.target = {PlanetArchetype::ProtoTheia, 0.2 * units::M_EARTH_SI, 4000.0, 4000.0, 0.0, 20};
    cfg.impactor = {PlanetArchetype::IronCore, 0.01 * units::M_EARTH_SI, 4000.0, 4000.0, 0.0, 20};
    cfg.impactVelocityRatioVesc = 0.0;
    cfg.impactParamB = 0.0;
    cfg.initialSepRadii = 15.0;
    cfg.totalParticles = 4000;
    cfg.tracerGridRes = 8;
    cfg.randomSeed = 42;

    Simulation sim(5000);
    sim.loadScenario(cfg);
    const auto t0 = sim.computeTelemetry();
    for (int i = 0; i < 20; ++i) {
        sim.step(5.0);
    }
    const auto t1 = sim.computeTelemetry();

    EXPECT_GT(t1.radiatedEnergyJ, t0.radiatedEnergyJ);
    EXPECT_GT(t1.radiatedEnergyJ, 0.0);
    EXPECT_LT(std::abs(t1.energyDriftRel), 0.01);
}

} // namespace
