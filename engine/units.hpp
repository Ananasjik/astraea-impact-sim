#pragma once

namespace astraea::units {

inline constexpr double G_SI           = 6.67430e-11;         // m^3 kg^-1 s^-2
inline constexpr double SIGMA_SB_SI    = 5.670374419e-8;      // W m^-2 K^-4
inline constexpr double R_GAS_SI       = 8.314462618;         // J mol^-1 K^-1
inline constexpr double SOLAR_FLUX_1AU = 1361.0;              // W m^-2
inline constexpr double M_EARTH_SI     = 5.97219e24;          // kg
inline constexpr double R_EARTH_SI     = 6.371e6;             // m

// Base PSU Scaling Constants
inline constexpr double L_UNIT_M       = 1.0e6;               // 1 PSU length = 1,000 km
inline constexpr double M_UNIT_KG      = 1.0e24;              // 1 PSU mass   = 10^24 kg
inline constexpr double T_UNIT_S       = 1.0e3;               // 1 PSU time   = 1,000 s

// Derived PSU -> SI Conversion Multipliers
inline constexpr double V_UNIT_MS      = L_UNIT_M / T_UNIT_S;                       // 1.0e3 m/s (1 km/s)
inline constexpr double A_UNIT_MS2     = L_UNIT_M / (T_UNIT_S * T_UNIT_S);          // 1.0 m/s^2
inline constexpr double RHO_UNIT_KGM3  = M_UNIT_KG / (L_UNIT_M * L_UNIT_M * L_UNIT_M); // 1.0e6 kg/m^3
inline constexpr double P_UNIT_PA      = M_UNIT_KG / (L_UNIT_M * T_UNIT_S * T_UNIT_S); // 1.0e12 Pa (1 TPa)
inline constexpr double U_UNIT_JKG     = (L_UNIT_M * L_UNIT_M) / (T_UNIT_S * T_UNIT_S); // 1.0e6 J/kg (1 MJ/kg)
inline constexpr double E_UNIT_J       = M_UNIT_KG * U_UNIT_JKG;                    // 1.0e30 J
inline constexpr double L_ANG_UNIT     = M_UNIT_KG * L_UNIT_M * V_UNIT_MS;          // 1.0e33 kg m^2/s
inline constexpr double G_PSU          = G_SI * M_UNIT_KG * T_UNIT_S * T_UNIT_S / (L_UNIT_M * L_UNIT_M * L_UNIT_M); // 66.7430

inline constexpr float  G_PSU_F        = static_cast<float>(G_PSU);                 // 66.7430f

} // namespace astraea::units
