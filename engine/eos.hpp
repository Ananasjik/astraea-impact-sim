#pragma once

#include <cstdint>

#include "types.hpp"
#include "units.hpp"

#ifdef __CUDACC__
#define ASTRAEA_HD __host__ __device__
#define ASTRAEA_HOST __host__
#else
#define ASTRAEA_HD
#define ASTRAEA_HOST
#endif

namespace astraea::eos {

struct TillotsonParams {
    double rho0KgM3;      // reference density (kg/m^3)
    double a;             // dimensionless high-energy polytropic factor
    double b;             // dimensionless thermal coupling factor
    double A_Pa;          // bulk modulus linear term (Pa)
    double B_Pa;          // quadratic compression term (Pa)
    double E0_JKg;        // characteristic specific energy (J/kg)
    double Eiv_JKg;       // incipient vaporization energy (J/kg)
    double Ecv_JKg;       // complete vaporization energy (J/kg)
    double alpha;         // exponential cutoff parameter in expanded regime
    double beta;          // exponential cutoff parameter in expanded regime
    double cv_JKgK;       // specific heat capacity (J / (kg K))
    double tMelt0K;       // zero-pressure melting temperature (K)
    double tRefK;         // STP reference temperature for rho0 (K)
    double muShear_Pa;    // intact shear modulus (Pa)
    double yieldY0_Pa;    // intact Von Mises yield strength (Pa)
    double epsFailMin;    // baseline Weibull tensile fracture strain
    double kH2_Poly;      // used only for HydrogenHelium (Pa m^6 / kg^2)
    double gammaIdeal;    // used only for HydrogenHelium
};

struct EosEvalResult {
    float   pressurePsu;
    float   soundSpeedPsu;
    float   tempK;
    uint8_t phaseId;
};

// Host + Device callable exact parameter table accessor
ASTRAEA_HD const TillotsonParams& getParams(MaterialId mat);

// Evaluates EOS in PSU (rhoPsu, uPsu) -> pressure, soundSpeed, tempK, phaseId
// Stored uPsu is total specific internal energy from 0 K so T = max(3.0, (u - u_cold)/cv) and energy
// conservation remains exact; Tillotson P_c, P_e, two-phase blend, and c_s evaluate at thermal excess
// u_EOS = u_cold(rho) + max(0.0, u - u_cold(rho) - cv * tRefK).
ASTRAEA_HD EosEvalResult evaluateEosPsu(MaterialId mat, float rhoPsu, float uPsu, float damage);

// Cold compression energy u_cold(rho) in PSU (used for temperature extraction & hydrostatic init)
ASTRAEA_HD double computeColdEnergyPsu(MaterialId mat, double rhoPsu);

// Converts (rho_SI, T_K) -> (u_SI, P_SI) for 1D hydrostatic profile generation
// Outputs total outUSi = u_cold(rho) + cv * tempK from 0 K while evaluating Tillotson P_SI using
// u_EOS = u_cold(rho) + max(0.0, outUSi - u_cold(rho) - cv * tRefK).
ASTRAEA_HOST void evaluateFromRhoTempSI(MaterialId mat, double rhoKgM3, double tempK, double& outUSi, double& outPSi);

// Bisects/Newtons P_SI(rho, T) == targetPSi for 1D planet builder
ASTRAEA_HOST double invertDensityFromPressureTempSI(MaterialId mat, double targetPSi, double tempK, double rhoGuessKgM3);

} // namespace astraea::eos
