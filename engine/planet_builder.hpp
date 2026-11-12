#pragma once

#include <cstdint>
#include <vector>

#include "types.hpp"

namespace astraea {

struct RadialProfileSample {
    double     rM = 0.0;
    double     rhoKgM3 = 0.0;
    double     pressurePa = 0.0;
    double     enclosedMassKg = 0.0;
    double     tempK = 0.0;
    MaterialId material = MaterialId::Granite;
};

struct BuiltPlanetHost {
    double                           equatorialRadiusM = 0.0;
    double                           actualMassKg = 0.0;
    double                           centralPressurePa = 0.0;
    std::vector<RadialProfileSample> radialProfile;
    std::vector<float>               posX, posY, posZ; // PSU
    std::vector<float>               velX, velY, velZ; // PSU
    std::vector<float>               mass, h0, rho, u, tempK, epsFail;
    std::vector<uint8_t>             materialId, bodyId;
    std::vector<int32_t>             tracerIndex;
    std::vector<float>               tracerRestRadiusPsu;
    std::vector<float>               tracerRestNormalX, tracerRestNormalY, tracerRestNormalZ;
    std::vector<uint8_t>             tracerIsOcean;
    uint32_t                         tracerCount = 0;
};

class PlanetBuilder {
public:
    static std::vector<RadialProfileSample> solveHydrostatic1D(
        const BodyInitSpec& spec,
        double& outRadiusM,
        double& outCentralPressurePa);

    static BuiltPlanetHost buildPlanet(
        const BodyInitSpec& spec,
        uint32_t particleCount,
        uint8_t bodyId,
        uint32_t tracerGridRes,
        uint64_t seed);
};

// Shared Cube-Sphere Face Orientation (identical in C++ planet_builder and TS CubeSphereLod):
// face 0 (+X): ( 1,  sy, -sx), face 1 (-X): (-1,  sy,  sx)
// face 2 (+Y): ( sx,  1, -sy), face 3 (-Y): ( sx, -1,  sy)
// face 4 (+Z): ( sx, sy,   1), face 5 (-Z): (-sx, sy,  -1)
// where sx = 2*u - 1, sy = 2*v - 1, u,v in [0, 1], normalized to unit sphere.
void cubeFaceUvToUnitNormal(uint32_t face, float u, float v, float& nx, float& ny, float& nz);
void unitNormalToCubeFaceUv(float nx, float ny, float nz, uint32_t& outFace, float& outU, float& outV);

} // namespace astraea
