#include "planet_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "eos.hpp"
#include "presets.hpp"
#include "units.hpp"

namespace astraea {

namespace {

constexpr double kPi = 3.14159265358979323846;

inline double evalGeothermK(
    const BodyInitSpec& spec,
    double pressurePa,
    double centralPressurePa,
    double pSurfPa) {
    const double denom = std::max(centralPressurePa - pSurfPa, 1.0);
    const double pFrac = std::clamp((pressurePa - pSurfPa) / denom, 0.0, 1.0);
    return spec.surfaceTempK + (spec.coreTempK - spec.surfaceTempK) * pFrac;
}

inline double evalDensityForMaterialSI(
    const BodyInitSpec& spec,
    MaterialId mat,
    double pressurePa,
    double centralPressurePa,
    double pSurfPa,
    double rhoGuessKgM3,
    double& outTempK) {
    outTempK = evalGeothermK(spec, pressurePa, centralPressurePa, pSurfPa);
    double rho = eos::invertDensityFromPressureTempSI(
        mat, std::max(pressurePa, 0.0), outTempK, rhoGuessKgM3);
    if (mat == MaterialId::HydrogenHelium) {
        rho = std::max(rho, 1.0e-3);
    }
    return rho;
}

struct ShootingResult {
    double                           radiusM = 0.0;
    double                           totalMassKg = 0.0;
    bool                             hitMassCap = false;
    std::vector<RadialProfileSample> profile;
};

ShootingResult integrateHydrostaticOutward(
    const BodyInitSpec& spec,
    const ArchetypeComposition& comp,
    double centralPressurePa,
    double pSurfPa,
    double drM,
    bool recordProfile) {
    ShootingResult res;
    if (recordProfile) {
        res.profile.reserve(1500);
    }

    const double targetMassKg = spec.massKg;
    const double massBailKg = 20.0 * targetMassKg;
    constexpr int kMaxSteps = 12000;

    std::array<double, 3> layerEndMass{};
    double cumFrac = 0.0;
    for (uint32_t i = 0; i < comp.layerCount; ++i) {
        cumFrac += comp.layers[i].massFraction;
        layerEndMass[i] = (i + 1 == comp.layerCount)
            ? std::numeric_limits<double>::infinity()
            : (cumFrac * targetMassKg);
    }

    uint32_t layerIdx = 0;
    double r = 0.0;
    double P = centralPressurePa;
    double M = 0.0;

    MaterialId mat0 = comp.layers[0].material;
    double t0 = spec.coreTempK;
    double rhoGuess = eos::getParams(mat0).rho0KgM3;
    double rho = evalDensityForMaterialSI(
        spec, mat0, P, centralPressurePa, pSurfPa, rhoGuess, t0);
    rhoGuess = rho;

    if (recordProfile) {
        res.profile.push_back(RadialProfileSample{0.0, rho, P, 0.0, t0, mat0});
    }

    auto derivatives = [&](MaterialId mat, double rVal, double pVal, double mVal, double rhoHint, double& dPdr, double& dMdr) {
        double tempK = t0;
        const double rhoLocal = evalDensityForMaterialSI(
            spec, mat, pVal, centralPressurePa, pSurfPa, rhoHint, tempK);
        dMdr = 4.0 * kPi * rVal * rVal * rhoLocal;
        if (rVal <= 1.0e-9) {
            dPdr = -(4.0 / 3.0) * kPi * units::G_SI * rhoLocal * rhoLocal * rVal;
        } else {
            dPdr = -units::G_SI * mVal * rhoLocal / (rVal * rVal);
        }
        return rhoLocal;
    };

    auto rk4Advance = [&](MaterialId mat, double r0, double p0, double m0, double hStep, double rhoHint, double& outP, double& outM) {
        double k1P = 0.0, k1M = 0.0;
        double k2P = 0.0, k2M = 0.0;
        double k3P = 0.0, k3M = 0.0;
        double k4P = 0.0, k4M = 0.0;
        const double rho1 = derivatives(mat, r0, p0, m0, rhoHint, k1P, k1M);
        const double rho2 = derivatives(
            mat, r0 + 0.5 * hStep, std::max(pSurfPa, p0 + 0.5 * hStep * k1P), m0 + 0.5 * hStep * k1M, rho1, k2P, k2M);
        const double rho3 = derivatives(
            mat, r0 + 0.5 * hStep, std::max(pSurfPa, p0 + 0.5 * hStep * k2P), m0 + 0.5 * hStep * k2M, rho2, k3P, k3M);
        derivatives(
            mat, r0 + hStep, std::max(pSurfPa, p0 + hStep * k3P), m0 + hStep * k3M, rho3, k4P, k4M);
        outP = p0 + (hStep / 6.0) * (k1P + 2.0 * k2P + 2.0 * k3P + k4P);
        outM = m0 + (hStep / 6.0) * (k1M + 2.0 * k2M + 2.0 * k3M + k4M);
    };

    for (int step = 0; step < kMaxSteps; ++step) {
        const MaterialId currMat = comp.layers[layerIdx].material;
        const double hStep = (spec.archetype == PlanetArchetype::GasGiant && layerIdx + 1 < comp.layerCount)
            ? (0.25 * drM)
            : drM;

        double nextP = P;
        double nextM = M;
        if (r == 0.0) {
            nextP = P - (2.0 / 3.0) * kPi * units::G_SI * rho * rho * hStep * hStep;
            nextM = (4.0 / 3.0) * kPi * rho * hStep * hStep * hStep;
        } else {
            rk4Advance(currMat, r, P, M, hStep, rhoGuess, nextP, nextM);
        }

        const bool crossesSurface = (nextP <= pSurfPa);
        const bool crossesLayer = (layerIdx + 1 < comp.layerCount) && (nextM >= layerEndMass[layerIdx]);

        const double fracP = crossesSurface
            ? ((P - nextP > 1.0e-30) ? std::clamp((P - pSurfPa) / (P - nextP), 0.0, 1.0) : 1.0)
            : 2.0;
        const double fracM = crossesLayer
            ? ((nextM - M > 1.0e-30) ? std::clamp((layerEndMass[layerIdx] - M) / (nextM - M), 0.0, 1.0) : 1.0)
            : 2.0;

        if (crossesSurface && (!crossesLayer || fracP <= fracM)) {
            double rSurf = r + fracP * hStep;
            double mSurf = M;
            if (r == 0.0) {
                const double dp = std::max(0.0, P - pSurfPa);
                const double rAnalytic = std::sqrt(dp / ((2.0 / 3.0) * kPi * units::G_SI * rho * rho));
                rSurf = std::min(hStep, rAnalytic);
                mSurf = (4.0 / 3.0) * kPi * rho * rSurf * rSurf * rSurf;
            } else {
                double dummyP = 0.0;
                rk4Advance(currMat, r, P, M, fracP * hStep, rhoGuess, dummyP, mSurf);
            }

            if (recordProfile) {
                double tSurf = spec.surfaceTempK;
                const double rhoSurf = evalDensityForMaterialSI(
                    spec, currMat, pSurfPa, centralPressurePa, pSurfPa, rhoGuess, tSurf);
                res.profile.push_back(RadialProfileSample{rSurf, rhoSurf, pSurfPa, mSurf, tSurf, currMat});
            }
            res.radiusM = rSurf;
            res.totalMassKg = mSurf;
            return res;
        }

        if (crossesLayer) {
            const double mBoundary = layerEndMass[layerIdx];
            double subDr = 0.0;
            if (r == 0.0) {
                subDr = std::cbrt((3.0 * mBoundary) / (4.0 * kPi * rho));
                subDr = std::clamp(subDr, 0.0, hStep);
                nextP = P - (2.0 / 3.0) * kPi * units::G_SI * rho * rho * subDr * subDr;
                nextM = mBoundary;
            } else {
                // Refine subDr to hit mBoundary cleanly via 3 secant steps
                double lo = 0.0;
                double hi = hStep;
                subDr = fracM * hStep;
                for (int it = 0; it < 4; ++it) {
                    rk4Advance(currMat, r, P, M, subDr, rhoGuess, nextP, nextM);
                    if (std::abs(nextM - mBoundary) <= 1.0e-10 * targetMassKg) {
                        break;
                    }
                    if (nextM < mBoundary) {
                        lo = subDr;
                    } else {
                        hi = subDr;
                    }
                    subDr = 0.5 * (lo + hi);
                }
                rk4Advance(currMat, r, P, M, subDr, rhoGuess, nextP, nextM);
            }

            if (nextP <= pSurfPa) {
                const double denom = P - nextP;
                const double subFrac = (denom > 1.0e-30) ? std::clamp((P - pSurfPa) / denom, 0.0, 1.0) : 1.0;
                const double rSurf = r + subFrac * subDr;
                double dummyP = 0.0, mSurf = M;
                rk4Advance(currMat, r, P, M, subFrac * subDr, rhoGuess, dummyP, mSurf);
                if (recordProfile) {
                    double tSurf = spec.surfaceTempK;
                    const double rhoSurf = evalDensityForMaterialSI(
                        spec, currMat, pSurfPa, centralPressurePa, pSurfPa, rhoGuess, tSurf);
                    res.profile.push_back(RadialProfileSample{rSurf, rhoSurf, pSurfPa, mSurf, tSurf, currMat});
                }
                res.radiusM = rSurf;
                res.totalMassKg = mSurf;
                return res;
            }

            r += subDr;
            P = nextP;
            M = mBoundary;
            if (recordProfile) {
                double tEdge = 0.0;
                const double rhoEdge = evalDensityForMaterialSI(
                    spec, currMat, P, centralPressurePa, pSurfPa, rhoGuess, tEdge);
                res.profile.push_back(RadialProfileSample{r, rhoEdge, P, M, tEdge, currMat});
            }
            ++layerIdx;
            const MaterialId nextMat = comp.layers[layerIdx].material;
            double tNew = 0.0;
            rhoGuess = eos::getParams(nextMat).rho0KgM3;
            rho = evalDensityForMaterialSI(
                spec, nextMat, P, centralPressurePa, pSurfPa, rhoGuess, tNew);
            rhoGuess = rho;
            if (recordProfile) {
                res.profile.push_back(RadialProfileSample{r, rho, P, M, tNew, nextMat});
            }
            continue;
        }

        r += hStep;
        P = nextP;
        M = nextM;

        double tCurr = spec.coreTempK;
        rho = evalDensityForMaterialSI(
            spec, currMat, P, centralPressurePa, pSurfPa, rhoGuess, tCurr);
        rhoGuess = rho;

        if (recordProfile) {
            res.profile.push_back(RadialProfileSample{r, rho, P, M, tCurr, currMat});
        }

        if (M > massBailKg) {
            res.radiusM = r;
            res.totalMassKg = M;
            res.hitMassCap = true;
            return res;
        }
    }

    res.radiusM = r;
    res.totalMassKg = M;
    return res;
}

inline RadialProfileSample interpolateProfileAtMass(
    const std::vector<RadialProfileSample>& profile,
    double targetMassKg) {
    if (profile.empty()) {
        return {};
    }
    if (targetMassKg <= profile.front().enclosedMassKg) {
        return profile.front();
    }
    if (targetMassKg >= profile.back().enclosedMassKg) {
        return profile.back();
    }
    auto it = std::lower_bound(
        profile.begin(),
        profile.end(),
        targetMassKg,
        [](const RadialProfileSample& s, double m) { return s.enclosedMassKg < m; });
    if (it == profile.begin()) {
        return *it;
    }
    const RadialProfileSample& hi = *it;
    const RadialProfileSample& lo = *(it - 1);
    const double dm = hi.enclosedMassKg - lo.enclosedMassKg;
    const double t = (dm > 1.0e-30) ? std::clamp((targetMassKg - lo.enclosedMassKg) / dm, 0.0, 1.0) : 0.0;

    RadialProfileSample out;
    out.rM = lo.rM + t * (hi.rM - lo.rM);
    out.rhoKgM3 = lo.rhoKgM3 + t * (hi.rhoKgM3 - lo.rhoKgM3);
    out.pressurePa = lo.pressurePa + t * (hi.pressurePa - lo.pressurePa);
    out.enclosedMassKg = targetMassKg;
    out.tempK = lo.tempK + t * (hi.tempK - lo.tempK);
    out.material = (t < 0.5) ? lo.material : hi.material;
    return out;
}

// Generate deterministic uniform random SO(3) rotation matrix via Shoemake (1992)
inline std::array<double, 9> randomRotationMatrix(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double u1 = u01(rng);
    const double u2 = 2.0 * kPi * u01(rng);
    const double u3 = 2.0 * kPi * u01(rng);

    const double s1 = std::sqrt(std::max(0.0, 1.0 - u1));
    const double s2 = std::sqrt(std::max(0.0, u1));
    const double qx = s1 * std::sin(u2);
    const double qy = s1 * std::cos(u2);
    const double qz = s2 * std::sin(u3);
    const double qw = s2 * std::cos(u3);

    const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const double wx = qw * qx, wy = qw * qy, wz = qw * qz;

    return {
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)
    };
}

} // namespace

void cubeFaceUvToUnitNormal(uint32_t face, float u, float v, float& nx, float& ny, float& nz) {
    const float sx = 2.0f * u - 1.0f;
    const float sy = 2.0f * v - 1.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;

    switch (face % 6u) {
        case 0: vx =  1.0f; vy =  sy;   vz = -sx;   break; // +X
        case 1: vx = -1.0f; vy =  sy;   vz =  sx;   break; // -X
        case 2: vx =  sx;   vy =  1.0f; vz = -sy;   break; // +Y
        case 3: vx =  sx;   vy = -1.0f; vz =  sy;   break; // -Y
        case 4: vx =  sx;   vy =  sy;   vz =  1.0f; break; // +Z
        case 5: vx = -sx;   vy =  sy;   vz = -1.0f; break; // -Z
    }

    const float invLen = 1.0f / std::sqrt(vx * vx + vy * vy + vz * vz);
    nx = vx * invLen;
    ny = vy * invLen;
    nz = vz * invLen;
}

void unitNormalToCubeFaceUv(float nx, float ny, float nz, uint32_t& outFace, float& outU, float& outV) {
    const float ax = std::abs(nx);
    const float ay = std::abs(ny);
    const float az = std::abs(nz);

    float sx = 0.0f;
    float sy = 0.0f;

    if (ax >= ay && ax >= az) {
        const float denom = std::max(ax, 1.0e-12f);
        if (nx >= 0.0f) {
            outFace = 0;
            sx = -nz / denom;
            sy =  ny / denom;
        } else {
            outFace = 1;
            sx =  nz / denom;
            sy =  ny / denom;
        }
    } else if (ay >= ax && ay >= az) {
        const float denom = std::max(ay, 1.0e-12f);
        if (ny >= 0.0f) {
            outFace = 2;
            sx =  nx / denom;
            sy = -nz / denom;
        } else {
            outFace = 3;
            sx =  nx / denom;
            sy =  nz / denom;
        }
    } else {
        const float denom = std::max(az, 1.0e-12f);
        if (nz >= 0.0f) {
            outFace = 4;
            sx =  nx / denom;
            sy =  ny / denom;
        } else {
            outFace = 5;
            sx = -nx / denom;
            sy =  ny / denom;
        }
    }

    outU = std::clamp(0.5f * (sx + 1.0f), 0.0f, 1.0f);
    outV = std::clamp(0.5f * (sy + 1.0f), 0.0f, 1.0f);
}

std::vector<RadialProfileSample> PlanetBuilder::solveHydrostatic1D(
    const BodyInitSpec& spec,
    double& outRadiusM,
    double& outCentralPressurePa) {
    if (spec.massKg <= 0.0) {
        throw std::invalid_argument("BodyInitSpec::massKg must be positive");
    }

    const ArchetypeComposition& comp = getArchetypeComposition(spec.archetype);
    const double pSurfPa = (spec.archetype == PlanetArchetype::GasGiant) ? 1.0e5 : 0.0;

    // Estimate characteristic radius for step sizing and initial log(P_c) bracket
    double invRhoWeighted = 0.0;
    for (uint32_t i = 0; i < comp.layerCount; ++i) {
        double rho0 = eos::getParams(comp.layers[i].material).rho0KgM3;
        if (comp.layers[i].material == MaterialId::HydrogenHelium) {
            rho0 = 600.0; // typical compressed giant-planet envelope density
        }
        invRhoWeighted += comp.layers[i].massFraction / std::max(rho0, 1.0);
    }
    const double rhoEst = 1.0 / std::max(invRhoWeighted, 1.0e-6);
    double rEst = std::cbrt((3.0 * spec.massKg) / (4.0 * kPi * rhoEst));
    if (spec.archetype == PlanetArchetype::GasGiant) {
        rEst = std::max(rEst, 6.5e7);
    }

    const double drM = std::max(rEst / 800.0, 1.0e2);
    const double pEst = (3.0 * units::G_SI * spec.massKg * spec.massKg) /
                        (8.0 * kPi * rEst * rEst * rEst * rEst);

    double logPLow = std::log(std::max(pSurfPa + 1.0e4, 1.0e-3 * pEst));
    double logPHigh = std::log(std::max(pSurfPa + 1.0e6, 1.0e3 * pEst));

    for (int i = 0; i < 20; ++i) {
        const ShootingResult resLow = integrateHydrostaticOutward(
            spec, comp, std::exp(logPLow), pSurfPa, drM, false);
        if (!resLow.hitMassCap && resLow.totalMassKg <= spec.massKg) {
            break;
        }
        logPLow -= 2.0;
    }

    for (int i = 0; i < 20; ++i) {
        const ShootingResult resHigh = integrateHydrostaticOutward(
            spec, comp, std::exp(logPHigh), pSurfPa, drM, false);
        if (resHigh.hitMassCap || resHigh.totalMassKg >= spec.massKg) {
            break;
        }
        logPHigh += 2.0;
    }

    double logPMid = 0.5 * (logPLow + logPHigh);
    for (int iter = 0; iter < 60; ++iter) {
        logPMid = 0.5 * (logPLow + logPHigh);
        const double pc = std::exp(logPMid);
        const ShootingResult res = integrateHydrostaticOutward(spec, comp, pc, pSurfPa, drM, false);
        const double relErr = (res.totalMassKg - spec.massKg) / spec.massKg;
        if (!res.hitMassCap && std::abs(relErr) <= 1.0e-6) {
            break;
        }
        if (res.hitMassCap || res.totalMassKg > spec.massKg) {
            logPHigh = logPMid;
        } else {
            logPLow = logPMid;
        }
        if (logPHigh - logPLow < 1.0e-12) {
            break;
        }
    }

    outCentralPressurePa = std::exp(logPMid);
    ShootingResult finalRes = integrateHydrostaticOutward(
        spec, comp, outCentralPressurePa, pSurfPa, drM, true);
    outRadiusM = finalRes.radiusM;
    return finalRes.profile;
}

BuiltPlanetHost PlanetBuilder::buildPlanet(
    const BodyInitSpec& spec,
    uint32_t particleCount,
    uint8_t bodyId,
    uint32_t tracerGridRes,
    uint64_t seed) {
    if (particleCount == 0) {
        throw std::invalid_argument("PlanetBuilder::buildPlanet requires particleCount > 0");
    }

    BuiltPlanetHost host;
    host.radialProfile = solveHydrostatic1D(spec, host.equatorialRadiusM, host.centralPressurePa);
    if (host.radialProfile.empty()) {
        throw std::runtime_error("Hydrostatic solver returned empty profile");
    }
    host.actualMassKg = host.radialProfile.back().enclosedMassKg;

    const ArchetypeComposition& comp = getArchetypeComposition(spec.archetype);
    const uint32_t L = comp.layerCount;

    // Partition particleCount across layers proportional to layer mass fractions
    std::array<uint32_t, 3> layerParticles{};
    std::array<double, 3> layerMassKg{};
    std::array<double, 3> layerCumMassEndKg{};

    double cumFrac = 0.0;
    uint32_t assignedParticles = 0;
    for (uint32_t k = 0; k < L; ++k) {
        layerMassKg[k] = comp.layers[k].massFraction * host.actualMassKg;
        cumFrac += comp.layers[k].massFraction;
        layerCumMassEndKg[k] = (k + 1 == L) ? host.actualMassKg : (cumFrac * host.actualMassKg);
        if (k + 1 == L) {
            layerParticles[k] = particleCount - assignedParticles;
        } else {
            uint32_t n = static_cast<uint32_t>(std::llround(comp.layers[k].massFraction * particleCount));
            if (particleCount >= L && n == 0) {
                n = 1;
            }
            const uint32_t maxAvail = particleCount - assignedParticles - ((particleCount >= L) ? (L - 1 - k) : 0u);
            n = std::min(n, maxAvail);
            layerParticles[k] = n;
            assignedParticles += n;
        }
    }

    host.posX.reserve(particleCount);
    host.posY.reserve(particleCount);
    host.posZ.reserve(particleCount);
    host.velX.reserve(particleCount);
    host.velY.reserve(particleCount);
    host.velZ.reserve(particleCount);
    host.mass.reserve(particleCount);
    host.h0.reserve(particleCount);
    host.rho.reserve(particleCount);
    host.u.reserve(particleCount);
    host.tempK.reserve(particleCount);
    host.epsFail.reserve(particleCount);
    host.materialId.reserve(particleCount);
    host.bodyId.reserve(particleCount);

    std::mt19937_64 rng(seed ^ (static_cast<uint64_t>(bodyId) * 0x9e3779b97f4a7c15ULL));
    std::uniform_real_distribution<double> u01(1.0e-6, 1.0 - 1.0e-6);

    const double nominalParticleMassKg = host.actualMassKg / static_cast<double>(particleCount);
    const double goldenAngle = kPi * (3.0 - std::sqrt(5.0));

    double mStartKg = 0.0;
    for (uint32_t k = 0; k < L; ++k) {
        const uint32_t nk = layerParticles[k];
        const double mEndKg = layerCumMassEndKg[k];
        if (nk == 0) {
            mStartKg = mEndKg;
            continue;
        }

        const double layerTotalMassKg = std::max(mEndKg - mStartKg, 0.0);
        const double particleMassKg = layerTotalMassKg / static_cast<double>(nk);
        const double particleMassPsu = particleMassKg / units::M_UNIT_KG;
        const MaterialId layerMat = comp.layers[k].material;
        const eos::TillotsonParams& eosParams = eos::getParams(layerMat);

        const RadialProfileSample sIn = interpolateProfileAtMass(host.radialProfile, mStartKg);
        const RadialProfileSample sOut = interpolateProfileAtMass(host.radialProfile, mEndKg);

        // Determine shell boundaries so radial spacing ~ isotropic particle spacing (m/rho)^(1/3)
        std::vector<double> shellMassBounds;
        shellMassBounds.push_back(mStartKg);
        double currM = mStartKg;
        double currR = sIn.rM;
        while (currM < mEndKg) {
            const RadialProfileSample sCurr = interpolateProfileAtMass(host.radialProfile, currM);
            const double spacingM = std::cbrt(nominalParticleMassKg / std::max(sCurr.rhoKgM3, 1.0e-3));
            const double nextR = currR + spacingM;
            if (nextR >= sOut.rM - 0.35 * spacingM) {
                shellMassBounds.push_back(mEndKg);
                break;
            }
            // Find enclosed mass at nextR by linear interpolation on profile
            auto it = std::lower_bound(
                host.radialProfile.begin(),
                host.radialProfile.end(),
                nextR,
                [](const RadialProfileSample& s, double r) { return s.rM < r; });
            double nextM = mEndKg;
            if (it != host.radialProfile.end() && it != host.radialProfile.begin()) {
                const auto& hi = *it;
                const auto& lo = *(it - 1);
                const double dr = hi.rM - lo.rM;
                const double t = (dr > 1.0e-30) ? std::clamp((nextR - lo.rM) / dr, 0.0, 1.0) : 0.0;
                nextM = lo.enclosedMassKg + t * (hi.enclosedMassKg - lo.enclosedMassKg);
            }
            nextM = std::clamp(nextM, currM + 1.0e-9 * layerTotalMassKg, mEndKg);
            shellMassBounds.push_back(nextM);
            if (nextM >= mEndKg) {
                break;
            }
            currM = nextM;
            currR = nextR;
        }
        if (shellMassBounds.size() < 2) {
            shellMassBounds.push_back(mEndKg);
        }

        const size_t numShells = shellMassBounds.size() - 1;
        std::vector<uint32_t> shellCounts(numShells, 0);
        std::vector<double> remainders(numShells, 0.0);
        uint32_t sumCounts = 0;
        for (size_t s = 0; s < numShells; ++s) {
            const double dm = shellMassBounds[s + 1] - shellMassBounds[s];
            const double exactN = (layerTotalMassKg > 0.0) ? (static_cast<double>(nk) * (dm / layerTotalMassKg)) : 0.0;
            const uint32_t floored = static_cast<uint32_t>(std::floor(exactN));
            shellCounts[s] = floored;
            remainders[s] = exactN - static_cast<double>(floored);
            sumCounts += floored;
        }
        // Largest-remainder adjustment so sum(shellCounts) == nk exactly
        if (sumCounts < nk) {
            std::vector<size_t> idx(numShells);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                return remainders[a] > remainders[b];
            });
            for (uint32_t rem = 0; rem < nk - sumCounts; ++rem) {
                shellCounts[idx[rem % numShells]] += 1;
            }
        }

        for (size_t s = 0; s < numShells; ++s) {
            const uint32_t ns = shellCounts[s];
            if (ns == 0) {
                continue;
            }

            const double mMidKg = 0.5 * (shellMassBounds[s] + shellMassBounds[s + 1]);
            const RadialProfileSample sample = interpolateProfileAtMass(host.radialProfile, mMidKg);

            // Force sample material to this layer's material (guards exact boundary rounding)
            const double rShellM = (s == 0 && k == 0 && ns == 1) ? 0.0 : sample.rM;
            const double rhoKgM3 = std::max(sample.rhoKgM3, 1.0e-3);
            const double tempK = sample.tempK;

            double uSi = 0.0;
            double pSi = 0.0;
            eos::evaluateFromRhoTempSI(layerMat, rhoKgM3, tempK, uSi, pSi);

            const double rPsu = rShellM / units::L_UNIT_M;
            const double rhoPsu = rhoKgM3 / units::RHO_UNIT_KGM3;
            const double uPsu = uSi / units::U_UNIT_JKG;
            // Target ~56 neighbors inside support radius H = 2 * h0
            const double h0Psu = 1.2 * std::cbrt(particleMassPsu / rhoPsu);

            const std::array<double, 9> rot = randomRotationMatrix(rng);

            for (uint32_t p = 0; p < ns; ++p) {
                double px = 0.0;
                double py = 0.0;
                double pz = 0.0;
                if (rPsu > 0.0) {
                    const double zLocal = (ns == 1) ? 1.0 : (1.0 - 2.0 * (static_cast<double>(p) + 0.5) / static_cast<double>(ns));
                    const double rxy = std::sqrt(std::max(0.0, 1.0 - zLocal * zLocal));
                    const double theta = goldenAngle * static_cast<double>(p);
                    const double ux = rxy * std::cos(theta);
                    const double uy = rxy * std::sin(theta);
                    const double uz = zLocal;

                    const double rx = rot[0] * ux + rot[1] * uy + rot[2] * uz;
                    const double ry = rot[3] * ux + rot[4] * uy + rot[5] * uz;
                    const double rz = rot[6] * ux + rot[7] * uy + rot[8] * uz;

                    px = rPsu * rx;
                    py = rPsu * ry;
                    pz = rPsu * rz;
                }

                // Rigid-body spin around +Y axis: v = omega x r = (+omega * z, 0, -omega * x)
                const double vxPsu = spec.spinOmegaRadS * units::T_UNIT_S * pz;
                const double vyPsu = 0.0;
                const double vzPsu = -spec.spinOmegaRadS * units::T_UNIT_S * px;

                // Deterministic Weibull tensile fracture threshold >= epsFailMin
                const double wDraw = -std::log(u01(rng));
                const double weibullScale = std::clamp(std::pow(std::max(wDraw, 1.0e-6), 1.0 / 9.0), 1.0, 3.0);
                const float epsFailVal = static_cast<float>(eosParams.epsFailMin * weibullScale);

                host.posX.push_back(static_cast<float>(px));
                host.posY.push_back(static_cast<float>(py));
                host.posZ.push_back(static_cast<float>(pz));
                host.velX.push_back(static_cast<float>(vxPsu));
                host.velY.push_back(static_cast<float>(vyPsu));
                host.velZ.push_back(static_cast<float>(vzPsu));
                host.mass.push_back(static_cast<float>(particleMassPsu));
                host.h0.push_back(static_cast<float>(h0Psu));
                host.rho.push_back(static_cast<float>(rhoPsu));
                host.u.push_back(static_cast<float>(uPsu));
                host.tempK.push_back(static_cast<float>(tempK));
                host.epsFail.push_back(epsFailVal);
                host.materialId.push_back(static_cast<uint8_t>(layerMat));
                host.bodyId.push_back(bodyId);
            }
        }

        mStartKg = mEndKg;
    }

    const uint32_t nTotal = static_cast<uint32_t>(host.posX.size());
    host.tracerIndex.assign(nTotal, -1);
    host.tracerRestRadiusPsu.assign(nTotal, 0.0f);
    host.tracerRestNormalX.assign(nTotal, 0.0f);
    host.tracerRestNormalY.assign(nTotal, 0.0f);
    host.tracerRestNormalZ.assign(nTotal, 0.0f);
    host.tracerIsOcean.assign(nTotal, 0);
    host.tracerCount = 0;

    // Select 6 * G^2 unique surface tracers from outermost crust + ocean shell for LivingTerra
    if (spec.archetype == PlanetArchetype::LivingTerra && tracerGridRes > 0) {
        const uint32_t targetTracers = 6u * tracerGridRes * tracerGridRes;
        if (nTotal >= targetTracers) {
            std::vector<uint32_t> candidates;
            candidates.reserve(nTotal);
            std::vector<float> radii(nTotal);
            for (uint32_t i = 0; i < nTotal; ++i) {
                const float x = host.posX[i];
                const float y = host.posY[i];
                const float z = host.posZ[i];
                radii[i] = std::sqrt(x * x + y * y + z * z);
                if (radii[i] > 1.0e-6f) {
                    candidates.push_back(i);
                }
            }
            // Sort candidates by descending radius so outermost crust + ocean shell comes first
            std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
                return radii[a] > radii[b];
            });

            // Restrict pool to outermost crust + ocean shell (at least 3x targetTracers for clean cell matching)
            const float rMaxPsu = radii[candidates.front()];
            const float drSurfPsu = host.h0[candidates.front()];
            size_t poolSize = 0;
            const size_t minPool = std::min(candidates.size(), static_cast<size_t>(targetTracers) * 3u);
            for (; poolSize < candidates.size(); ++poolSize) {
                const uint32_t pid = candidates[poolSize];
                const bool inShell = (radii[pid] >= rMaxPsu - 1.8f * drSurfPsu) ||
                                     (host.materialId[pid] == static_cast<uint8_t>(MaterialId::WaterIce));
                if (!inShell && poolSize >= minPool) {
                    break;
                }
            }
            if (poolSize < targetTracers) {
                poolSize = std::min(candidates.size(), static_cast<size_t>(targetTracers));
            }
            candidates.resize(poolSize);

            std::vector<float> candNx(poolSize), candNy(poolSize), candNz(poolSize);
            std::vector<uint8_t> candUsed(poolSize, 0);
            std::vector<std::vector<uint32_t>> binCandidates(targetTracers);

            for (size_t c = 0; c < poolSize; ++c) {
                const uint32_t pid = candidates[c];
                const float invR = 1.0f / radii[pid];
                candNx[c] = host.posX[pid] * invR;
                candNy[c] = host.posY[pid] * invR;
                candNz[c] = host.posZ[pid] * invR;

                uint32_t face = 0;
                float u = 0.0f, v = 0.0f;
                unitNormalToCubeFaceUv(candNx[c], candNy[c], candNz[c], face, u, v);
                const uint32_t gx = std::min(tracerGridRes - 1u, static_cast<uint32_t>(u * static_cast<float>(tracerGridRes)));
                const uint32_t gy = std::min(tracerGridRes - 1u, static_cast<uint32_t>(v * static_cast<float>(tracerGridRes)));
                const uint32_t binIdx = face * (tracerGridRes * tracerGridRes) + gy * tracerGridRes + gx;
                binCandidates[binIdx].push_back(static_cast<uint32_t>(c));
            }

            std::vector<float> binTargetX(targetTracers), binTargetY(targetTracers), binTargetZ(targetTracers);
            std::vector<uint8_t> binFilled(targetTracers, 0);

            auto assignParticleToBin = [&](uint32_t binIdx, size_t candIdx) {
                const uint32_t pid = candidates[candIdx];
                candUsed[candIdx] = 1;
                binFilled[binIdx] = 1;
                host.tracerIndex[pid] = static_cast<int32_t>(binIdx);
                host.tracerRestRadiusPsu[pid] = radii[pid];
                host.tracerRestNormalX[pid] = candNx[candIdx];
                host.tracerRestNormalY[pid] = candNy[candIdx];
                host.tracerRestNormalZ[pid] = candNz[candIdx];
                host.tracerIsOcean[pid] =
                    (host.materialId[pid] == static_cast<uint8_t>(MaterialId::WaterIce)) ? 1u : 0u;
            };

            // Pass A: pick best matching candidate inside each cube-sphere bin
            for (uint32_t face = 0; face < 6; ++face) {
                for (uint32_t gy = 0; gy < tracerGridRes; ++gy) {
                    for (uint32_t gx = 0; gx < tracerGridRes; ++gx) {
                        const uint32_t binIdx = face * (tracerGridRes * tracerGridRes) + gy * tracerGridRes + gx;
                        const float u = (static_cast<float>(gx) + 0.5f) / static_cast<float>(tracerGridRes);
                        const float v = (static_cast<float>(gy) + 0.5f) / static_cast<float>(tracerGridRes);
                        cubeFaceUvToUnitNormal(face, u, v, binTargetX[binIdx], binTargetY[binIdx], binTargetZ[binIdx]);

                        float bestDot = -2.0f;
                        size_t bestCand = std::numeric_limits<size_t>::max();
                        for (uint32_t c : binCandidates[binIdx]) {
                            if (candUsed[c]) {
                                continue;
                            }
                            const float dot = candNx[c] * binTargetX[binIdx] +
                                              candNy[c] * binTargetY[binIdx] +
                                              candNz[c] * binTargetZ[binIdx];
                            if (dot > bestDot) {
                                bestDot = dot;
                                bestCand = c;
                            }
                        }
                        if (bestCand != std::numeric_limits<size_t>::max()) {
                            assignParticleToBin(binIdx, bestCand);
                        }
                    }
                }
            }

            // Pass B: fill any remaining empty bins with nearest unassigned surface candidate
            for (uint32_t binIdx = 0; binIdx < targetTracers; ++binIdx) {
                if (binFilled[binIdx]) {
                    continue;
                }
                float bestDot = -2.0f;
                size_t bestCand = std::numeric_limits<size_t>::max();
                for (size_t c = 0; c < poolSize; ++c) {
                    if (candUsed[c]) {
                        continue;
                    }
                    const float dot = candNx[c] * binTargetX[binIdx] +
                                      candNy[c] * binTargetY[binIdx] +
                                      candNz[c] * binTargetZ[binIdx];
                    if (dot > bestDot) {
                        bestDot = dot;
                        bestCand = c;
                    }
                }
                if (bestCand != std::numeric_limits<size_t>::max()) {
                    assignParticleToBin(binIdx, bestCand);
                }
            }

            host.tracerCount = targetTracers;
        }
    }

    return host;
}

} // namespace astraea
