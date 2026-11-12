#include "eos.hpp"

#include <algorithm>
#include <cmath>

namespace astraea::eos {

namespace {

constexpr TillotsonParams kParamsHost[5] = {
    // 0: Iron (Tillotson 1962 / Melosh 1989)
    {
        7800.0,   // rho0KgM3
        0.5,      // a
        1.5,      // b
        1.28e11,  // A_Pa
        1.05e11,  // B_Pa
        9.50e6,   // E0_JKg
        2.40e6,   // Eiv_JKg
        8.67e6,   // Ecv_JKg
        5.0,      // alpha
        5.0,      // beta
        449.0,    // cv_JKgK
        1811.0,   // tMelt0K
        293.0,    // tRefK
        7.70e10,  // muShear_Pa
        6.80e8,   // yieldY0_Pa
        1.0e-3,   // epsFailMin
        0.0,      // kH2_Poly
        1.0       // gammaIdeal
    },
    // 1: Granite (Benz & Asphaug 1999)
    {
        2680.0,   // rho0KgM3
        0.5,      // a
        1.3,      // b
        1.80e10,  // A_Pa
        1.80e10,  // B_Pa
        1.60e7,   // E0_JKg
        3.50e6,   // Eiv_JKg
        1.80e7,   // Ecv_JKg
        5.0,      // alpha
        5.0,      // beta
        790.0,    // cv_JKgK
        1600.0,   // tMelt0K
        293.0,    // tRefK
        2.20e10,  // muShear_Pa
        3.50e9,   // yieldY0_Pa
        1.0e-4,   // epsFailMin
        0.0,      // kH2_Poly
        1.0       // gammaIdeal
    },
    // 2: WaterIce (Allen 1967 / Tillotson Water)
    {
        1000.0,   // rho0KgM3
        0.7,      // a
        0.15,     // b
        2.18e9,   // A_Pa
        1.325e10, // B_Pa
        7.00e6,   // E0_JKg
        4.19e5,   // Eiv_JKg
        2.68e6,   // Ecv_JKg
        10.0,     // alpha
        5.0,      // beta
        4184.0,   // cv_JKgK
        273.15,   // tMelt0K
        273.15,   // tRefK
        3.50e9,   // muShear_Pa (active only T < 273.15)
        1.00e7,   // yieldY0_Pa (active only T < 273.15)
        5.0e-4,   // epsFailMin
        0.0,      // kH2_Poly
        1.0       // gammaIdeal
    },
    // 3: HydrogenHelium (Polytropic n=1 + Ideal)
    {
        0.178,    // rho0KgM3 (1-bar ref)
        0.4,      // a (gamma - 1)
        0.0,      // b
        0.0,      // A_Pa
        0.0,      // B_Pa
        0.0,      // E0_JKg
        0.0,      // Eiv_JKg
        0.0,      // Ecv_JKg
        0.0,      // alpha
        0.0,      // beta
        8260.0,   // cv_JKgK
        0.0,      // tMelt0K (gas)
        0.0,      // tRefK
        0.0,      // muShear_Pa
        0.0,      // yieldY0_Pa
        1.0,      // epsFailMin
        2.08e5,   // kH2_Poly (Pa m^6 / kg^2)
        1.4       // gammaIdeal
    },
    // 4: Regolith (Porous Silicate / Jutzi 2008)
    {
        2000.0,   // rho0KgM3
        0.5,      // a
        1.3,      // b
        1.20e10,  // A_Pa
        1.20e10,  // B_Pa
        1.60e7,   // E0_JKg
        3.50e6,   // Eiv_JKg
        1.80e7,   // Ecv_JKg
        5.0,      // alpha
        5.0,      // beta
        800.0,    // cv_JKgK
        1500.0,   // tMelt0K
        293.0,    // tRefK
        8.00e9,   // muShear_Pa
        5.00e7,   // yieldY0_Pa
        5.0e-5,   // epsFailMin
        0.0,      // kH2_Poly
        1.0       // gammaIdeal
    }
};

#ifdef __CUDACC__
__device__ __constant__ TillotsonParams kParamsDevice[5] = {
    kParamsHost[0],
    kParamsHost[1],
    kParamsHost[2],
    kParamsHost[3],
    kParamsHost[4]
};
#endif

ASTRAEA_HD inline double clampD(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

ASTRAEA_HD inline double maxD(double a, double b) {
    return a > b ? a : b;
}

ASTRAEA_HD inline void computeTillotsonCompressedPsu(
    const TillotsonParams& p,
    double rhoPsu,
    double rho0Psu,
    double eta,
    double mu,
    double uEosPsu,
    double damageClamped,
    double& outP,
    double& outDpDrho,
    double& outDpDu,
    bool& outDamagedTension) {
    const double A = p.A_Pa / units::P_UNIT_PA;
    const double B = p.B_Pa / units::P_UNIT_PA;
    const double E0 = p.E0_JKg / units::U_UNIT_JKG;
    const double bEff = (mu >= 0.0) ? B : 0.0;

    const double denom = E0 * eta * eta;
    const double ratio = (denom > 1.0e-30) ? (uEosPsu / denom) : 1.0e30;
    const double w = ratio + 1.0;
    const double invW = 1.0 / w;
    const double wMinus1OverW2 = (ratio > 1.0e15) ? 0.0 : (ratio * invW * invW);

    double pc = (p.a + p.b * invW) * rhoPsu * uEosPsu + A * mu + bEff * mu * mu;
    double dPcDrho = (p.a + p.b * invW) * uEosPsu + 2.0 * p.b * uEosPsu * wMinus1OverW2
                     + (A + 2.0 * bEff * mu) / rho0Psu;
    double dPcDu = (p.a + p.b * invW * invW) * rhoPsu;

    outDamagedTension = false;
    if (pc < 0.0 && damageClamped > 0.0) {
        const double intact = 1.0 - damageClamped;
        pc *= intact;
        dPcDrho *= intact;
        dPcDu *= intact;
        outDamagedTension = true;
    }

    outP = pc;
    outDpDrho = dPcDrho;
    outDpDu = dPcDu;
}

ASTRAEA_HD inline void computeTillotsonExpandedPsu(
    const TillotsonParams& p,
    double rhoPsu,
    double rho0Psu,
    double eta,
    double mu,
    double uEosPsu,
    double& outP,
    double& outDpDrho,
    double& outDpDu) {
    const double A = p.A_Pa / units::P_UNIT_PA;
    const double E0 = p.E0_JKg / units::U_UNIT_JKG;

    const double x = 1.0 / eta - 1.0;
    const double ax2 = p.alpha * x * x;
    const double bx = p.beta * x;
    const double eAlpha = (ax2 < 700.0) ? std::exp(-ax2) : 0.0;

    if (eAlpha == 0.0) {
        outP = p.a * rhoPsu * uEosPsu;
        outDpDrho = p.a * uEosPsu;
        outDpDu = p.a * rhoPsu;
        return;
    }

    const double eBeta = (bx < 700.0) ? std::exp(-bx) : 0.0;
    const double denom = E0 * eta * eta;
    const double ratio = (denom > 1.0e-30) ? (uEosPsu / denom) : 1.0e30;
    const double w = ratio + 1.0;
    const double invW = 1.0 / w;
    const double wMinus1OverW2 = (ratio > 1.0e15) ? 0.0 : (ratio * invW * invW);

    const double thermalIdeal = p.a * rhoPsu * uEosPsu;
    const double nonIdealInner = (p.b * rhoPsu * uEosPsu * invW) + A * mu * eBeta;
    const double pe = thermalIdeal + nonIdealInner * eAlpha;

    const double rho0OverRho2 = rho0Psu / (rhoPsu * rhoPsu);
    const double expTermDrho = (eBeta > 0.0)
        ? ((A / rho0Psu + A * mu * p.beta * rho0OverRho2) * eBeta)
        : 0.0;

    const double dPeDrho = p.a * uEosPsu
        + eAlpha * (p.b * uEosPsu * invW + 2.0 * p.b * uEosPsu * wMinus1OverW2 + expTermDrho)
        + 2.0 * p.alpha * x * rho0OverRho2 * (pe - thermalIdeal);
    const double dPeDu = (p.a + p.b * eAlpha * invW * invW) * rhoPsu;

    outP = pe;
    outDpDrho = dPeDrho;
    outDpDu = dPeDu;
}

ASTRAEA_HD inline uint8_t classifyPhase(
    MaterialId mat,
    const TillotsonParams& p,
    double rhoPsu,
    double uPsu,
    double tempK) {
    if (mat == MaterialId::HydrogenHelium) {
        return (tempK > 10000.0) ? static_cast<uint8_t>(PhaseId::Plasma)
                                 : static_cast<uint8_t>(PhaseId::Gas);
    }

    const double rhoKgM3 = rhoPsu * units::RHO_UNIT_KGM3;
    const double ecvPsu = p.Ecv_JKg / units::U_UNIT_JKG;

    if (mat == MaterialId::WaterIce) {
        if (tempK > 15000.0) {
            return static_cast<uint8_t>(PhaseId::Plasma);
        }
        if (uPsu >= ecvPsu || (tempK > 647.0 && rhoKgM3 < 322.0)) {
            return static_cast<uint8_t>(PhaseId::Vapor);
        }
        if (tempK >= 273.15) {
            return static_cast<uint8_t>(PhaseId::LiquidWater);
        }
        return static_cast<uint8_t>(PhaseId::Solid);
    }

    // Iron, Granite, Regolith
    if (tempK > 12000.0) {
        return static_cast<uint8_t>(PhaseId::Plasma);
    }
    if (uPsu >= ecvPsu && rhoKgM3 < 0.8 * p.rho0KgM3) {
        return static_cast<uint8_t>(PhaseId::Vapor);
    }
    if (tempK >= p.tMelt0K) {
        return static_cast<uint8_t>(PhaseId::Molten);
    }
    return static_cast<uint8_t>(PhaseId::Solid);
}

} // namespace

ASTRAEA_HD const TillotsonParams& getParams(MaterialId mat) {
    uint8_t idx = static_cast<uint8_t>(mat);
    if (idx > 4) {
        idx = 1; // Granite fallback
    }
#ifdef __CUDA_ARCH__
    return kParamsDevice[idx];
#else
    return kParamsHost[idx];
#endif
}

ASTRAEA_HD double computeColdEnergyPsu(MaterialId mat, double rhoPsu) {
    const TillotsonParams& p = getParams(mat);
    const double rho = maxD(rhoPsu, 0.0);

    if (mat == MaterialId::HydrogenHelium) {
        // K_H2 in PSU numerically equals K_H2_SI since P_UNIT / RHO_UNIT^2 == 1
        return p.kH2_Poly * rho;
    }

    const double rho0Psu = p.rho0KgM3 / units::RHO_UNIT_KGM3;
    const double eta = rho / rho0Psu;
    if (eta <= 1.0) {
        return 0.0;
    }

    const double mu = eta - 1.0;
    const double A = p.A_Pa / units::P_UNIT_PA;
    const double B = p.B_Pa / units::P_UNIT_PA;

    double termA = 0.0;
    double termB = 0.0;
    if (mu < 1.0e-5) {
        const double mu2 = mu * mu;
        const double mu3 = mu2 * mu;
        const double mu4 = mu3 * mu;
        const double mu5 = mu4 * mu;
        termA = 0.5 * mu2 - (2.0 / 3.0) * mu3 + 0.75 * mu4 - 0.8 * mu5;
        termB = (1.0 / 3.0) * mu3 - 0.5 * mu4 + 0.6 * mu5;
    } else {
        const double logEta = std::log1p(mu);
        termA = logEta - (mu / eta);
        termB = (mu * (2.0 + mu)) / eta - 2.0 * logEta;
    }

    return (A * termA + B * termB) / rho0Psu;
}

ASTRAEA_HD EosEvalResult evaluateEosPsu(MaterialId mat, float rhoPsuIn, float uPsuIn, float damageIn) {
    const TillotsonParams& p = getParams(mat);
    const double rho = maxD(static_cast<double>(rhoPsuIn), 1.0e-12);
    const double u = static_cast<double>(uPsuIn);
    const double damage = clampD(static_cast<double>(damageIn), 0.0, 1.0);
    const double cvPsu = p.cv_JKgK / units::U_UNIT_JKG;
    const double uCold = computeColdEnergyPsu(mat, rho);

    if (mat == MaterialId::HydrogenHelium) {
        const double uTh = maxD(0.0, u - uCold);
        const double tempK = maxD(3.0, uTh / cvPsu);
        const double P = p.kH2_Poly * rho * rho + (p.gammaIdeal - 1.0) * rho * uTh;
        const double cs2 = 2.0 * p.kH2_Poly * rho + p.gammaIdeal * (p.gammaIdeal - 1.0) * uTh;
        const double cs = maxD(0.05, std::sqrt(maxD(0.0, cs2)));
        return EosEvalResult{
            static_cast<float>(P),
            static_cast<float>(cs),
            static_cast<float>(tempK),
            classifyPhase(mat, p, rho, u, tempK)
        };
    }

    const double rho0Psu = p.rho0KgM3 / units::RHO_UNIT_KGM3;
    const double eta = rho / rho0Psu;
    const double mu = eta - 1.0;

    const double tempK = maxD(3.0, (u - uCold) / cvPsu);
    const double uEos = uCold + maxD(0.0, u - uCold - cvPsu * p.tRefK);

    const double Eiv = p.Eiv_JKg / units::U_UNIT_JKG;
    const double Ecv = p.Ecv_JKg / units::U_UNIT_JKG;
    const double A = p.A_Pa / units::P_UNIT_PA;

    double pressure = 0.0;
    double dPDrho = 0.0;
    double dPDu = 0.0;
    bool damagedTension = false;

    if (eta >= 1.0 || uEos < Eiv) {
        computeTillotsonCompressedPsu(
            p, rho, rho0Psu, eta, mu, uEos, damage, pressure, dPDrho, dPDu, damagedTension);
    } else if (uEos >= Ecv) {
        computeTillotsonExpandedPsu(p, rho, rho0Psu, eta, mu, uEos, pressure, dPDrho, dPDu);
    } else {
        double pc = 0.0, dPcDrho = 0.0, dPcDu = 0.0;
        double pe = 0.0, dPeDrho = 0.0, dPeDu = 0.0;
        computeTillotsonCompressedPsu(
            p, rho, rho0Psu, eta, mu, uEos, damage, pc, dPcDrho, dPcDu, damagedTension);
        computeTillotsonExpandedPsu(p, rho, rho0Psu, eta, mu, uEos, pe, dPeDrho, dPeDu);

        const double denom = maxD(1.0e-30, Ecv - Eiv);
        const double f = clampD((uEos - Eiv) / denom, 0.0, 1.0);
        pressure = (1.0 - f) * pc + f * pe;
        dPDrho = (1.0 - f) * dPcDrho + f * dPeDrho;
        dPDu = (1.0 - f) * dPcDu + f * dPeDu + (pe - pc) / denom;
    }

    const double cs2 = dPDrho + (pressure / (rho * rho)) * dPDu;
    const bool isCondensed = (eta >= 1.0) || (uEos < Eiv && rho >= 0.9 * rho0Psu && !damagedTension);
    const double cMin = isCondensed ? maxD(0.05, std::sqrt(A / rho0Psu)) : 0.05;
    const double cs = maxD(cMin, std::sqrt(maxD(0.0, cs2)));

    return EosEvalResult{
        static_cast<float>(pressure),
        static_cast<float>(cs),
        static_cast<float>(tempK),
        classifyPhase(mat, p, rho, u, tempK)
    };
}

ASTRAEA_HOST void evaluateFromRhoTempSI(
    MaterialId mat,
    double rhoKgM3,
    double tempK,
    double& outUSi,
    double& outPSi) {
    const TillotsonParams& p = getParams(mat);
    const double rho = maxD(rhoKgM3, 1.0e-12);
    const double tClamped = maxD(tempK, 0.0);

    if (mat == MaterialId::HydrogenHelium) {
        const double uColdSi = p.kH2_Poly * rho;
        const double uThSi = p.cv_JKgK * tClamped;
        outUSi = uColdSi + uThSi;
        outPSi = p.kH2_Poly * rho * rho + (p.gammaIdeal - 1.0) * rho * uThSi;
        return;
    }

    const double rhoPsu = rho / units::RHO_UNIT_KGM3;
    const double uColdPsu = computeColdEnergyPsu(mat, rhoPsu);
    const double uColdSi = uColdPsu * units::U_UNIT_JKG;
    outUSi = uColdSi + p.cv_JKgK * tClamped;

    const double uThermalExcessSi = maxD(0.0, outUSi - uColdSi - p.cv_JKgK * p.tRefK);
    const double uEosPsu = uColdPsu + uThermalExcessSi / units::U_UNIT_JKG;

    const double rho0Psu = p.rho0KgM3 / units::RHO_UNIT_KGM3;
    const double eta = rhoPsu / rho0Psu;
    const double mu = eta - 1.0;
    const double Eiv = p.Eiv_JKg / units::U_UNIT_JKG;
    const double Ecv = p.Ecv_JKg / units::U_UNIT_JKG;

    double pressurePsu = 0.0;
    double dPDrho = 0.0;
    double dPDu = 0.0;
    bool dummyDamaged = false;

    if (eta >= 1.0 || uEosPsu < Eiv) {
        computeTillotsonCompressedPsu(
            p, rhoPsu, rho0Psu, eta, mu, uEosPsu, 0.0, pressurePsu, dPDrho, dPDu, dummyDamaged);
    } else if (uEosPsu >= Ecv) {
        computeTillotsonExpandedPsu(p, rhoPsu, rho0Psu, eta, mu, uEosPsu, pressurePsu, dPDrho, dPDu);
    } else {
        double pc = 0.0, pe = 0.0;
        computeTillotsonCompressedPsu(
            p, rhoPsu, rho0Psu, eta, mu, uEosPsu, 0.0, pc, dPDrho, dPDu, dummyDamaged);
        computeTillotsonExpandedPsu(p, rhoPsu, rho0Psu, eta, mu, uEosPsu, pe, dPDrho, dPDu);
        const double f = clampD((uEosPsu - Eiv) / maxD(1.0e-30, Ecv - Eiv), 0.0, 1.0);
        pressurePsu = (1.0 - f) * pc + f * pe;
    }

    outPSi = pressurePsu * units::P_UNIT_PA;
}

ASTRAEA_HOST double invertDensityFromPressureTempSI(
    MaterialId mat,
    double targetPSi,
    double tempK,
    double rhoGuessKgM3) {
    const TillotsonParams& p = getParams(mat);

    if (mat == MaterialId::HydrogenHelium) {
        if (targetPSi <= 0.0) {
            return 1.0e-6;
        }
        const double bLin = (p.gammaIdeal - 1.0) * p.cv_JKgK * maxD(tempK, 0.0);
        const double disc = std::sqrt(bLin * bLin + 4.0 * p.kH2_Poly * targetPSi);
        return (2.0 * targetPSi) / maxD(1.0e-30, bLin + disc);
    }

    auto evalP = [&](double rho) -> double {
        double uSi = 0.0;
        double pSi = 0.0;
        evaluateFromRhoTempSI(mat, rho, tempK, uSi, pSi);
        return pSi;
    };

    double rhoStart = (rhoGuessKgM3 > 0.0) ? rhoGuessKgM3 : p.rho0KgM3;
    rhoStart = clampD(rhoStart, 0.05 * p.rho0KgM3, 50.0 * p.rho0KgM3);

    double rhoLow = rhoStart;
    double rhoHigh = rhoStart;
    double pStart = evalP(rhoStart);

    if (pStart > targetPSi) {
        for (int i = 0; i < 64; ++i) {
            rhoLow *= (i < 12) ? 0.85 : 0.5;
            if (rhoLow < 1.0e-6) {
                rhoLow = 1.0e-6;
            }
            if (evalP(rhoLow) <= targetPSi || rhoLow <= 1.0e-6) {
                break;
            }
        }
    } else if (pStart < targetPSi) {
        for (int i = 0; i < 64; ++i) {
            rhoHigh *= (i < 12) ? 1.25 : 2.0;
            if (evalP(rhoHigh) >= targetPSi || rhoHigh > 1.0e8) {
                break;
            }
        }
    } else {
        return rhoStart;
    }

    double fLow = evalP(rhoLow) - targetPSi;
    double fHigh = evalP(rhoHigh) - targetPSi;
    if (fLow >= 0.0) {
        return rhoLow;
    }
    if (fHigh <= 0.0) {
        return rhoHigh;
    }

    double rho = 0.5 * (rhoLow + rhoHigh);
    for (int iter = 0; iter < 64; ++iter) {
        const double fMid = evalP(rho) - targetPSi;
        if (std::abs(fMid) <= 1.0e-9 * maxD(1.0, std::abs(targetPSi))) {
            return rho;
        }

        if (fMid < 0.0) {
            rhoLow = rho;
            fLow = fMid;
        } else {
            rhoHigh = rho;
            fHigh = fMid;
        }

        if (rhoHigh - rhoLow <= 1.0e-12 * maxD(1.0, rho)) {
            return 0.5 * (rhoLow + rhoHigh);
        }

        // Secant proposal inside bracket, falling back to bisection
        double nextRho = 0.5 * (rhoLow + rhoHigh);
        const double denom = fHigh - fLow;
        if (std::abs(denom) > 1.0e-30) {
            const double secant = rhoLow - fLow * (rhoHigh - rhoLow) / denom;
            if (secant > rhoLow && secant < rhoHigh) {
                nextRho = secant;
            }
        }
        rho = nextRho;
    }

    return 0.5 * (rhoLow + rhoHigh);
}

} // namespace astraea::eos
