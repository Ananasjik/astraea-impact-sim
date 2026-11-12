#include "sph_gravity.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <utility>

#include <cub/cub.cuh>

#include "eos.hpp"
#include "units.hpp"

namespace astraea::cuda {

namespace {

constexpr int   kBlockSize      = 256;
constexpr int   kWarpBlockSize  = 128;
constexpr int   kWarpsPerBlock  = kWarpBlockSize / 32;
constexpr int   kBvhStackSize   = 192;
constexpr float kPiF            = 3.14159265358979323846f;

// Compile-time PSU reference specific internal energies (cv * 300 K) / U_UNIT_JKG
// matching eos.cu cold+300K scale for Iron, Granite, WaterIce, HydrogenHelium, Regolith.
__device__ __constant__ float kU0RefPsu[5] = {
    static_cast<float>((449.0 * 300.0) / astraea::units::U_UNIT_JKG),
    static_cast<float>((790.0 * 300.0) / astraea::units::U_UNIT_JKG),
    static_cast<float>((4184.0 * 300.0) / astraea::units::U_UNIT_JKG),
    static_cast<float>((8260.0 * 300.0) / astraea::units::U_UNIT_JKG),
    static_cast<float>((800.0 * 300.0) / astraea::units::U_UNIT_JKG)
};

struct MatStrengthConstants {
    float muShearPsu;
    float tMelt0K;
    float bulkModPsu;
    float yieldY0Psu;
};

__device__ __constant__ MatStrengthConstants kMatStrength[5] = {
    // 0: Iron
    {
        static_cast<float>(7.70e10 / astraea::units::P_UNIT_PA),
        1811.0f,
        static_cast<float>(1.28e11 / astraea::units::P_UNIT_PA),
        static_cast<float>(6.80e8 / astraea::units::P_UNIT_PA)
    },
    // 1: Granite
    {
        static_cast<float>(2.20e10 / astraea::units::P_UNIT_PA),
        1600.0f,
        static_cast<float>(1.80e10 / astraea::units::P_UNIT_PA),
        static_cast<float>(3.50e9 / astraea::units::P_UNIT_PA)
    },
    // 2: WaterIce
    {
        static_cast<float>(3.50e9 / astraea::units::P_UNIT_PA),
        273.15f,
        static_cast<float>(2.18e9 / astraea::units::P_UNIT_PA),
        static_cast<float>(1.00e7 / astraea::units::P_UNIT_PA)
    },
    // 3: HydrogenHelium
    {0.0f, 0.0f, 0.0f, 0.0f},
    // 4: Regolith
    {
        static_cast<float>(8.00e9 / astraea::units::P_UNIT_PA),
        1500.0f,
        static_cast<float>(1.20e10 / astraea::units::P_UNIT_PA),
        static_cast<float>(5.00e7 / astraea::units::P_UNIT_PA)
    }
};

// Persistent fallback scratch (used only by launchReduceDtPsu or if a caller manually
// initialized only the spec fields of RadixBvhDevice without calling allocateRadixBvh).
struct GlobalScratch {
    std::mutex mutex;
    uint32_t*  dtMinBits = nullptr;
    uint32_t*  boundsOrdered = nullptr;
    float*     boundsFloat = nullptr;
    uint32_t*  livingFlags = nullptr;

    ~GlobalScratch() {
        if (dtMinBits != nullptr)     { cudaFree(dtMinBits);     dtMinBits = nullptr; }
        if (boundsOrdered != nullptr) { cudaFree(boundsOrdered); boundsOrdered = nullptr; }
        if (boundsFloat != nullptr)   { cudaFree(boundsFloat);   boundsFloat = nullptr; }
        if (livingFlags != nullptr)   { cudaFree(livingFlags);   livingFlags = nullptr; }
    }
};

GlobalScratch& globalScratch() {
    static GlobalScratch s;
    return s;
}

uint32_t* ensureDtMinBits() {
    GlobalScratch& s = globalScratch();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.dtMinBits == nullptr) {
        ASTRAEA_CUDA_CHECK(cudaMalloc(&s.dtMinBits, sizeof(uint32_t)));
    }
    return s.dtMinBits;
}

void ensureFallbackBvhScratch(uint32_t*& boundsOrdered, float*& boundsFloat, uint32_t*& livingFlags) {
    GlobalScratch& s = globalScratch();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.boundsOrdered == nullptr) {
        ASTRAEA_CUDA_CHECK(cudaMalloc(&s.boundsOrdered, 6 * sizeof(uint32_t)));
    }
    if (s.boundsFloat == nullptr) {
        ASTRAEA_CUDA_CHECK(cudaMalloc(&s.boundsFloat, 6 * sizeof(float)));
    }
    if (s.livingFlags == nullptr) {
        ASTRAEA_CUDA_CHECK(cudaMalloc(&s.livingFlags, 2 * sizeof(uint32_t)));
    }
    boundsOrdered = s.boundsOrdered;
    boundsFloat = s.boundsFloat;
    livingFlags = s.livingFlags;
}

template <typename T>
void allocArray(T*& ptr, size_t count) {
    if (count == 0) {
        ptr = nullptr;
        return;
    }
    ASTRAEA_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
}

template <typename T>
void freeArray(T*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

__device__ __forceinline__ uint32_t floatToOrderedUint(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(u >> 31)) | 0x80000000u;
    return u ^ mask;
}

__device__ __forceinline__ float orderedUintToFloat(uint32_t u) {
    const uint32_t mask = ((u >> 31) - 1u) | 0x80000000u;
    return __uint_as_float(u ^ mask);
}

__device__ __forceinline__ uint64_t expandBits21(uint64_t v) {
    v &= 0x1fffffULL;
    v = (v | (v << 32)) & 0x001f00000000ffffULL;
    v = (v | (v << 16)) & 0x001f0000ff0000ffULL;
    v = (v | (v << 8))  & 0x100f00f00f00f00fULL;
    v = (v | (v << 4))  & 0x10c30c30c30c30c3ULL;
    v = (v | (v << 2))  & 0x1249249249249249ULL;
    return v;
}

__device__ __forceinline__ uint64_t morton3D64(float x, float y, float z,
                                               float minX, float minY, float minZ,
                                               float invSpanX, float invSpanY, float invSpanZ) {
    const float fx = fminf(fmaxf((x - minX) * invSpanX, 0.0f), 2097151.0f);
    const float fy = fminf(fmaxf((y - minY) * invSpanY, 0.0f), 2097151.0f);
    const float fz = fminf(fmaxf((z - minZ) * invSpanZ, 0.0f), 2097151.0f);
    const uint64_t ix = static_cast<uint64_t>(static_cast<uint32_t>(fx));
    const uint64_t iy = static_cast<uint64_t>(static_cast<uint32_t>(fy));
    const uint64_t iz = static_cast<uint64_t>(static_cast<uint32_t>(fz));
    return (expandBits21(ix) << 2) | (expandBits21(iy) << 1) | expandBits21(iz);
}

__device__ __forceinline__ int karrasDelta(
    const uint64_t* __restrict__ codes,
    int n,
    int i,
    int j) {
    if (j < 0 || j >= n) {
        return -1;
    }
    const uint64_t ci = __ldg(&codes[i]);
    const uint64_t cj = __ldg(&codes[j]);
    if (ci != cj) {
        return __clzll(ci ^ cj);
    }
    return 64 + __clz(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(j));
}

__device__ __forceinline__ float pointAabbDistSq(
    float px, float py, float pz,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ) {
    const float dx = fmaxf(0.0f, fmaxf(minX - px, px - maxX));
    const float dy = fmaxf(0.0f, fmaxf(minY - py, py - maxY));
    const float dz = fmaxf(0.0f, fmaxf(minZ - pz, pz - maxZ));
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float wendlandC2(float r, float H) {
    if (r >= H || H <= 0.0f) {
        return 0.0f;
    }
    const float q = r / H;
    const float omq = 1.0f - q;
    const float omq2 = omq * omq;
    const float norm = 21.0f / (2.0f * kPiF * H * H * H);
    return norm * omq2 * omq2 * (1.0f + 4.0f * q);
}

// Returns scalar g such that grad_i W(r_i - r_j, H) = g * (r_i - r_j).
__device__ __forceinline__ float wendlandC2GradCoeff(float r, float H) {
    if (r >= H || H <= 0.0f) {
        return 0.0f;
    }
    const float q = r / H;
    const float omq = 1.0f - q;
    const float H2 = H * H;
    const float H5 = H2 * H2 * H;
    return -(210.0f / (kPiF * H5)) * omq * omq * omq;
}

__device__ __forceinline__ float computeBalsaraSwitch(
    float divV, float curlX, float curlY, float curlZ, float cs, float h) {
    const float absDiv = fabsf(divV);
    const float absCurl = sqrtf(curlX * curlX + curlY * curlY + curlZ * curlZ);
    const float denom = absDiv + absCurl + 1.0e-4f * fmaxf(cs, 0.05f) / fmaxf(h, 1.0e-6f);
    return absDiv / fmaxf(denom, 1.0e-30f);
}

__device__ __forceinline__ float maxPrincipalStressFromTraceless(
    float pressure,
    float sXX, float sYY, float sZZ,
    float sXY, float sXZ, float sYZ) {
    const float p2 = sXX * sXX + sYY * sYY + sZZ * sZZ
                   + 2.0f * (sXY * sXY + sXZ * sXZ + sYZ * sYZ);
    if (p2 <= 1.0e-24f) {
        return -pressure;
    }
    const float q = sqrtf(p2 * (1.0f / 6.0f));
    const float invQ = 1.0f / q;
    const float bXX = sXX * invQ;
    const float bYY = sYY * invQ;
    const float bZZ = sZZ * invQ;
    const float bXY = sXY * invQ;
    const float bXZ = sXZ * invQ;
    const float bYZ = sYZ * invQ;
    const float detB = bXX * (bYY * bZZ - bYZ * bYZ)
                     - bXY * (bXY * bZZ - bYZ * bXZ)
                     + bXZ * (bXY * bYZ - bYY * bXZ);
    const float r = fminf(1.0f, fmaxf(-1.0f, 0.5f * detB));
    const float phi = acosf(r) * (1.0f / 3.0f);
    const float lambdaMaxS = 2.0f * q * cosf(phi);
    return -pressure + lambdaMaxS;
}

// ---------- Scene AABB + Morton Code + Permutation + Karras BVH Kernels ----------

__global__ void initSceneBoundsKernel(uint32_t* boundsOrdered, uint32_t* livingFlags) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const uint32_t maxOrd = floatToOrderedUint(1.0e30f);
        const uint32_t minOrd = floatToOrderedUint(-1.0e30f);
        boundsOrdered[0] = maxOrd;
        boundsOrdered[1] = maxOrd;
        boundsOrdered[2] = maxOrd;
        boundsOrdered[3] = minOrd;
        boundsOrdered[4] = minOrd;
        boundsOrdered[5] = minOrd;
        livingFlags[0] = 0u;
        livingFlags[1] = 0u;
    }
}

__global__ void reduceSceneBoundsKernel(
    const float* __restrict__ posX,
    const float* __restrict__ posY,
    const float* __restrict__ posZ,
    uint32_t n,
    uint32_t* __restrict__ boundsOrdered) {
    __shared__ float sMinX[kBlockSize];
    __shared__ float sMinY[kBlockSize];
    __shared__ float sMinZ[kBlockSize];
    __shared__ float sMaxX[kBlockSize];
    __shared__ float sMaxY[kBlockSize];
    __shared__ float sMaxZ[kBlockSize];

    float minX = 1.0e30f, minY = 1.0e30f, minZ = 1.0e30f;
    float maxX = -1.0e30f, maxY = -1.0e30f, maxZ = -1.0e30f;

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = tid; i < n; i += stride) {
        const float x = __ldg(&posX[i]);
        const float y = __ldg(&posY[i]);
        const float z = __ldg(&posZ[i]);
        minX = fminf(minX, x); maxX = fmaxf(maxX, x);
        minY = fminf(minY, y); maxY = fmaxf(maxY, y);
        minZ = fminf(minZ, z); maxZ = fmaxf(maxZ, z);
    }

    const int lt = threadIdx.x;
    sMinX[lt] = minX; sMinY[lt] = minY; sMinZ[lt] = minZ;
    sMaxX[lt] = maxX; sMaxY[lt] = maxY; sMaxZ[lt] = maxZ;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (lt < offset) {
            sMinX[lt] = fminf(sMinX[lt], sMinX[lt + offset]);
            sMinY[lt] = fminf(sMinY[lt], sMinY[lt + offset]);
            sMinZ[lt] = fminf(sMinZ[lt], sMinZ[lt + offset]);
            sMaxX[lt] = fmaxf(sMaxX[lt], sMaxX[lt + offset]);
            sMaxY[lt] = fmaxf(sMaxY[lt], sMaxY[lt + offset]);
            sMaxZ[lt] = fmaxf(sMaxZ[lt], sMaxZ[lt + offset]);
        }
        __syncthreads();
    }

    if (lt == 0) {
        atomicMin(&boundsOrdered[0], floatToOrderedUint(sMinX[0]));
        atomicMin(&boundsOrdered[1], floatToOrderedUint(sMinY[0]));
        atomicMin(&boundsOrdered[2], floatToOrderedUint(sMinZ[0]));
        atomicMax(&boundsOrdered[3], floatToOrderedUint(sMaxX[0]));
        atomicMax(&boundsOrdered[4], floatToOrderedUint(sMaxY[0]));
        atomicMax(&boundsOrdered[5], floatToOrderedUint(sMaxZ[0]));
    }
}

__global__ void finalizeSceneBoundsKernel(
    const uint32_t* __restrict__ boundsOrdered,
    float* __restrict__ boundsFloat) {
    if (threadIdx.x < 6 && blockIdx.x == 0) {
        boundsFloat[threadIdx.x] = orderedUintToFloat(boundsOrdered[threadIdx.x]);
    }
}

__global__ void computeMortonCodesKernel(
    const float* __restrict__ posX,
    const float* __restrict__ posY,
    const float* __restrict__ posZ,
    const float* __restrict__ boundsFloat,
    uint32_t n,
    uint64_t* __restrict__ mortonCodes,
    uint32_t* __restrict__ indicesIn) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float minX = boundsFloat[0];
    const float minY = boundsFloat[1];
    const float minZ = boundsFloat[2];
    const float spanX = boundsFloat[3] - minX;
    const float spanY = boundsFloat[4] - minY;
    const float spanZ = boundsFloat[5] - minZ;
    const float invSpanX = (spanX > 1.0e-6f) ? (2097151.0f / spanX) : 0.0f;
    const float invSpanY = (spanY > 1.0e-6f) ? (2097151.0f / spanY) : 0.0f;
    const float invSpanZ = (spanZ > 1.0e-6f) ? (2097151.0f / spanZ) : 0.0f;

    mortonCodes[i] = morton3D64(
        __ldg(&posX[i]), __ldg(&posY[i]), __ldg(&posZ[i]),
        minX, minY, minZ, invSpanX, invSpanY, invSpanZ);
    indicesIn[i] = i;
}

__global__ void permuteParticlesKernel(
    ParticleBuffersDevice p,
    ParticleBuffersDevice alt,
    const uint32_t* __restrict__ indicesSorted,
    uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const uint32_t s = __ldg(&indicesSorted[i]);

    alt.posX[i]                = __ldg(&p.posX[s]);
    alt.posY[i]                = __ldg(&p.posY[s]);
    alt.posZ[i]                = __ldg(&p.posZ[s]);
    alt.velX[i]                = __ldg(&p.velX[s]);
    alt.velY[i]                = __ldg(&p.velY[s]);
    alt.velZ[i]                = __ldg(&p.velZ[s]);
    alt.velHalfX[i]            = __ldg(&p.velHalfX[s]);
    alt.velHalfY[i]            = __ldg(&p.velHalfY[s]);
    alt.velHalfZ[i]            = __ldg(&p.velHalfZ[s]);
    alt.accX[i]                = __ldg(&p.accX[s]);
    alt.accY[i]                = __ldg(&p.accY[s]);
    alt.accZ[i]                = __ldg(&p.accZ[s]);
    alt.mass[i]                = __ldg(&p.mass[s]);
    alt.h[i]                   = __ldg(&p.h[s]);
    alt.h0[i]                  = __ldg(&p.h0[s]);
    alt.rho[i]                 = __ldg(&p.rho[s]);
    alt.drhoDt[i]              = __ldg(&p.drhoDt[s]);
    alt.pressure[i]            = __ldg(&p.pressure[s]);
    alt.soundSpeed[i]          = __ldg(&p.soundSpeed[s]);
    alt.u[i]                   = __ldg(&p.u[s]);
    alt.duDt[i]                = __ldg(&p.duDt[s]);
    alt.tempK[i]               = __ldg(&p.tempK[s]);
    alt.radPowerW[i]           = __ldg(&p.radPowerW[s]);
    alt.sXX[i]                 = __ldg(&p.sXX[s]);
    alt.sYY[i]                 = __ldg(&p.sYY[s]);
    alt.sZZ[i]                 = __ldg(&p.sZZ[s]);
    alt.sXY[i]                 = __ldg(&p.sXY[s]);
    alt.sXZ[i]                 = __ldg(&p.sXZ[s]);
    alt.sYZ[i]                 = __ldg(&p.sYZ[s]);
    alt.dSXX[i]                = __ldg(&p.dSXX[s]);
    alt.dSYY[i]                = __ldg(&p.dSYY[s]);
    alt.dSZZ[i]                = __ldg(&p.dSZZ[s]);
    alt.dSXY[i]                = __ldg(&p.dSXY[s]);
    alt.dSXZ[i]                = __ldg(&p.dSXZ[s]);
    alt.dSYZ[i]                = __ldg(&p.dSYZ[s]);
    alt.damage[i]              = __ldg(&p.damage[s]);
    alt.damageCbrt[i]          = __ldg(&p.damageCbrt[s]);
    alt.dDamageCbrtDt[i]       = __ldg(&p.dDamageCbrtDt[s]);
    alt.epsFail[i]             = __ldg(&p.epsFail[s]);
    alt.divV[i]                = __ldg(&p.divV[s]);
    alt.curlVX[i]              = __ldg(&p.curlVX[s]);
    alt.curlVY[i]              = __ldg(&p.curlVY[s]);
    alt.curlVZ[i]              = __ldg(&p.curlVZ[s]);
    alt.shepardC[i]            = __ldg(&p.shepardC[s]);
    alt.surfFactor[i]          = __ldg(&p.surfFactor[s]);
    alt.phiGrav[i]             = __ldg(&p.phiGrav[s]);
    alt.materialId[i]          = __ldg(&p.materialId[s]);
    alt.phaseId[i]             = __ldg(&p.phaseId[s]);
    alt.bodyId[i]              = __ldg(&p.bodyId[s]);
    alt.originalId[i]          = __ldg(&p.originalId[s]);
    alt.tracerIndex[i]         = __ldg(&p.tracerIndex[s]);
    alt.tracerRestRadiusPsu[i] = __ldg(&p.tracerRestRadiusPsu[s]);
    alt.tracerRestNormalX[i]   = __ldg(&p.tracerRestNormalX[s]);
    alt.tracerRestNormalY[i]   = __ldg(&p.tracerRestNormalY[s]);
    alt.tracerRestNormalZ[i]   = __ldg(&p.tracerRestNormalZ[s]);
    alt.tracerIsOcean[i]       = __ldg(&p.tracerIsOcean[s]);
}

__global__ void initLeavesAndBuildKarrasKernel(
    ParticleBuffersDevice p,
    RadixBvhDevice bvh,
    uint32_t n,
    uint32_t* __restrict__ livingFlags) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize leaf [n - 1 .. 2n - 2]
    if (idx < n) {
        const uint32_t leafNode = (n - 1u) + idx;
        const float x = __ldg(&p.posX[idx]);
        const float y = __ldg(&p.posY[idx]);
        const float z = __ldg(&p.posZ[idx]);
        const float m = __ldg(&p.mass[idx]);
        const float H = 2.0f * __ldg(&p.h[idx]);

        bvh.aabbMinX[leafNode] = x;
        bvh.aabbMinY[leafNode] = y;
        bvh.aabbMinZ[leafNode] = z;
        bvh.aabbMaxX[leafNode] = x;
        bvh.aabbMaxY[leafNode] = y;
        bvh.aabbMaxZ[leafNode] = z;
        bvh.nodeMass[leafNode] = m;
        bvh.comX[leafNode]     = x;
        bvh.comY[leafNode]     = y;
        bvh.comZ[leafNode]     = z;
        bvh.nodeMaxH[leafNode] = H;

        if (bvh.nodeComMass != nullptr) {
            bvh.nodeComMass[leafNode]    = make_float4(x, y, z, m);
            bvh.nodeMinExtent[leafNode]  = make_float4(x, y, z, 0.0f);
            bvh.nodeMaxHPacked[leafNode] = make_float4(x, y, z, H);
        }

        if (__ldg(&p.tracerIndex[idx]) >= 0) {
            atomicOr(&livingFlags[0], 1u);
            if (__ldg(&p.damage[idx]) < 0.8f) {
                atomicOr(&livingFlags[1], 1u);
            }
        }
        if (n == 1u) {
            bvh.parent[0] = -1;
        }
    }

    // Build internal node idx in [0 .. n - 2]
    if (idx + 1u < n) {
        const int i = static_cast<int>(idx);
        const int nInt = static_cast<int>(n);
        const uint64_t* codes = bvh.mortonCodesSorted;

        const int dPos = karrasDelta(codes, nInt, i, i + 1);
        const int dNeg = karrasDelta(codes, nInt, i, i - 1);
        const int d = (dPos - dNeg >= 0) ? 1 : -1;
        const int deltaMin = karrasDelta(codes, nInt, i, i - d);

        int lmax = 2;
        while (karrasDelta(codes, nInt, i, i + lmax * d) > deltaMin) {
            lmax <<= 1;
        }

        int l = 0;
        for (int t = lmax >> 1; t >= 1; t >>= 1) {
            if (karrasDelta(codes, nInt, i, i + (l + t) * d) > deltaMin) {
                l += t;
            }
        }
        const int j = i + l * d;

        const int deltaNode = karrasDelta(codes, nInt, i, j);
        int s = 0;
        int t = l;
        do {
            t = (t + 1) >> 1;
            if (karrasDelta(codes, nInt, i, i + (s + t) * d) > deltaNode) {
                s += t;
            }
        } while (t > 1);

        const int gamma = i + s * d + ((d < 0) ? d : 0);
        const int first = (i < j) ? i : j;
        const int last  = (i > j) ? i : j;

        const int leftChildVal  = (first == gamma) ? ~gamma : gamma;
        const int rightChildVal = (last == gamma + 1) ? ~(gamma + 1) : (gamma + 1);
        const int leftNodeArr   = (first == gamma) ? (nInt - 1 + gamma) : gamma;
        const int rightNodeArr  = (last == gamma + 1) ? (nInt - 1 + gamma + 1) : (gamma + 1);

        bvh.leftChild[i]  = leftChildVal;
        bvh.rightChild[i] = rightChildVal;
        bvh.parent[leftNodeArr]  = i;
        bvh.parent[rightNodeArr] = i;
        bvh.atomicVisited[i] = 0u;
        if (i == 0) {
            bvh.parent[0] = -1;
        }
    }
}

__global__ void bottomUpBvhMomentsKernel(RadixBvhDevice bvh, uint32_t n) {
    const uint32_t leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= n || n < 2u) {
        return;
    }

    int curr = static_cast<int>((n - 1u) + leafIdx);
    int pNode = bvh.parent[curr];
    __threadfence();

    while (pNode >= 0) {
        const uint32_t arrived = atomicAdd(&bvh.atomicVisited[pNode], 1u);
        if (arrived == 0u) {
            return;
        }

        const int lc = __ldcg(&bvh.leftChild[pNode]);
        const int rc = __ldcg(&bvh.rightChild[pNode]);
        const int lArr = (lc < 0) ? (static_cast<int>(n - 1u) + (~lc)) : lc;
        const int rArr = (rc < 0) ? (static_cast<int>(n - 1u) + (~rc)) : rc;

        const float lMinX = __ldcg(&bvh.aabbMinX[lArr]);
        const float lMinY = __ldcg(&bvh.aabbMinY[lArr]);
        const float lMinZ = __ldcg(&bvh.aabbMinZ[lArr]);
        const float lMaxX = __ldcg(&bvh.aabbMaxX[lArr]);
        const float lMaxY = __ldcg(&bvh.aabbMaxY[lArr]);
        const float lMaxZ = __ldcg(&bvh.aabbMaxZ[lArr]);
        const float lMass = __ldcg(&bvh.nodeMass[lArr]);
        const float lComX = __ldcg(&bvh.comX[lArr]);
        const float lComY = __ldcg(&bvh.comY[lArr]);
        const float lComZ = __ldcg(&bvh.comZ[lArr]);
        const float lMaxH = __ldcg(&bvh.nodeMaxH[lArr]);

        const float rMinX = __ldcg(&bvh.aabbMinX[rArr]);
        const float rMinY = __ldcg(&bvh.aabbMinY[rArr]);
        const float rMinZ = __ldcg(&bvh.aabbMinZ[rArr]);
        const float rMaxX = __ldcg(&bvh.aabbMaxX[rArr]);
        const float rMaxY = __ldcg(&bvh.aabbMaxY[rArr]);
        const float rMaxZ = __ldcg(&bvh.aabbMaxZ[rArr]);
        const float rMass = __ldcg(&bvh.nodeMass[rArr]);
        const float rComX = __ldcg(&bvh.comX[rArr]);
        const float rComY = __ldcg(&bvh.comY[rArr]);
        const float rComZ = __ldcg(&bvh.comZ[rArr]);
        const float rMaxH = __ldcg(&bvh.nodeMaxH[rArr]);

        const float minX = fminf(lMinX, rMinX);
        const float minY = fminf(lMinY, rMinY);
        const float minZ = fminf(lMinZ, rMinZ);
        const float maxX = fmaxf(lMaxX, rMaxX);
        const float maxY = fmaxf(lMaxY, rMaxY);
        const float maxZ = fmaxf(lMaxZ, rMaxZ);
        const float maxH = fmaxf(lMaxH, rMaxH);
        const float mSum = lMass + rMass;

        float cx, cy, cz;
        if (mSum > 0.0f) {
            const float invM = 1.0f / mSum;
            cx = (lMass * lComX + rMass * rComX) * invM;
            cy = (lMass * lComY + rMass * rComY) * invM;
            cz = (lMass * lComZ + rMass * rComZ) * invM;
        } else {
            cx = 0.5f * (lComX + rComX);
            cy = 0.5f * (lComY + rComY);
            cz = 0.5f * (lComZ + rComZ);
        }

        bvh.aabbMinX[pNode] = minX;
        bvh.aabbMinY[pNode] = minY;
        bvh.aabbMinZ[pNode] = minZ;
        bvh.aabbMaxX[pNode] = maxX;
        bvh.aabbMaxY[pNode] = maxY;
        bvh.aabbMaxZ[pNode] = maxZ;
        bvh.nodeMaxH[pNode] = maxH;
        bvh.nodeMass[pNode] = mSum;
        bvh.comX[pNode]     = cx;
        bvh.comY[pNode]     = cy;
        bvh.comZ[pNode]     = cz;

        if (bvh.nodeComMass != nullptr) {
            const float dNode = fmaxf(maxX - minX, fmaxf(maxY - minY, maxZ - minZ));
            bvh.nodeComMass[pNode]    = make_float4(cx, cy, cz, mSum);
            bvh.nodeMinExtent[pNode]  = make_float4(minX, minY, minZ, dNode);
            bvh.nodeMaxHPacked[pNode] = make_float4(maxX, maxY, maxZ, maxH);
        }

        curr = pNode;
        pNode = __ldcg(&bvh.parent[curr]);
        __threadfence();
    }
}

// ---------- Utility Clear Stats Kernel ----------

__global__ void clearStatsKernel(ParticleBuffersDevice p, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    p.accX[i] = 0.0f; p.accY[i] = 0.0f; p.accZ[i] = 0.0f;
    p.drhoDt[i] = 0.0f;
    p.duDt[i] = 0.0f;
    p.radPowerW[i] = 0.0f;
    p.dSXX[i] = 0.0f; p.dSYY[i] = 0.0f; p.dSZZ[i] = 0.0f;
    p.dSXY[i] = 0.0f; p.dSXZ[i] = 0.0f; p.dSYZ[i] = 0.0f;
    p.dDamageCbrtDt[i] = 0.0f;
    p.divV[i] = 0.0f;
    p.curlVX[i] = 0.0f; p.curlVY[i] = 0.0f; p.curlVZ[i] = 0.0f;
    p.shepardC[i] = 1.0f;
    p.surfFactor[i] = 0.0f;
    p.phiGrav[i] = 0.0f;
}

// ---------- SPH Pass 1: Warp-Cooperative Shared-Stack BVH Traversal ----------

template <bool UsePacked>
__global__ void sphPass1Kernel(ParticleBuffersDevice p, RadixBvhDevice bvh, uint32_t n) {
    __shared__ int sStack[kWarpsPerBlock][kBvhStackSize];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (i < n);
    if (__ballot_sync(0xFFFFFFFFu, active) == 0u) {
        return;
    }

    const int warpId = static_cast<int>(threadIdx.x >> 5);
    const int lane   = static_cast<int>(threadIdx.x & 31);

    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    float vxi = 0.0f, vyi = 0.0f, vzi = 0.0f;
    float mi = 0.0f, hi = 1.0e-6f, h0i = 1.0e-6f, rhoI = 1.0f;
    float Hi = 0.0f, Hi2 = 0.0f;
    float shepardC = 0.0f;
    int ngbCount = 1;

    if (active) {
        xi  = __ldg(&p.posX[i]);
        yi  = __ldg(&p.posY[i]);
        zi  = __ldg(&p.posZ[i]);
        vxi = __ldg(&p.velX[i]);
        vyi = __ldg(&p.velY[i]);
        vzi = __ldg(&p.velZ[i]);
        mi  = __ldg(&p.mass[i]);
        hi  = fmaxf(__ldg(&p.h[i]), 1.0e-6f);
        h0i = fmaxf(__ldg(&p.h0[i]), 1.0e-6f);
        rhoI = fmaxf(__ldg(&p.rho[i]), 1.0e-9f);
        Hi  = 2.0f * hi;
        Hi2 = Hi * Hi;
        const float w0 = 21.0f / (2.0f * kPiF * Hi * Hi * Hi);
        shepardC = (mi / rhoI) * w0;
    }

    float nsX = 0.0f, nsY = 0.0f, nsZ = 0.0f;
    float gXX = 0.0f, gXY = 0.0f, gXZ = 0.0f;
    float gYX = 0.0f, gYY = 0.0f, gYZ = 0.0f;
    float gZX = 0.0f, gZY = 0.0f, gZZ = 0.0f;

    if (n >= 2u) {
        if (lane == 0) {
            sStack[warpId][0] = 0;
        }
        int sp = 1;
        __syncwarp();

        while (sp > 0) {
            const int node = sStack[warpId][--sp];
            const int children[2] = {
                __ldg(&bvh.leftChild[node]),
                __ldg(&bvh.rightChild[node])
            };

            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int c = children[k];
                if (c < 0) {
                    const uint32_t j = static_cast<uint32_t>(~c);
                    const float pxj = __ldg(&p.posX[j]);
                    const float pyj = __ldg(&p.posY[j]);
                    const float pzj = __ldg(&p.posZ[j]);
                    const float dx = xi - pxj;
                    const float dy = yi - pyj;
                    const float dz = zi - pzj;
                    const float r2 = dx * dx + dy * dy + dz * dz;
                    const bool inRange = active && (j != i) && (r2 < Hi2);
                    if (__any_sync(0xFFFFFFFFu, inRange)) {
                        const float mj   = __ldg(&p.mass[j]);
                        const float rhoj = fmaxf(__ldg(&p.rho[j]), 1.0e-9f);
                        const float vxj  = __ldg(&p.velX[j]);
                        const float vyj  = __ldg(&p.velY[j]);
                        const float vzj  = __ldg(&p.velZ[j]);
                        if (inRange) {
                            ++ngbCount;
                            const float r = sqrtf(r2);
                            const float volJ = mj / rhoj;
                            const float wij = wendlandC2(r, Hi);
                            shepardC += volJ * wij;

                            nsX -= volJ * dx * wij;
                            nsY -= volJ * dy * wij;
                            nsZ -= volJ * dz * wij;

                            const float gCoeff = wendlandC2GradCoeff(r, Hi);
                            const float gx = gCoeff * dx;
                            const float gy = gCoeff * dy;
                            const float gz = gCoeff * dz;
                            const float dvx = vxj - vxi;
                            const float dvy = vyj - vyi;
                            const float dvz = vzj - vzi;

                            gXX += volJ * dvx * gx; gXY += volJ * dvx * gy; gXZ += volJ * dvx * gz;
                            gYX += volJ * dvy * gx; gYY += volJ * dvy * gy; gYZ += volJ * dvy * gz;
                            gZX += volJ * dvz * gx; gZY += volJ * dvz * gy; gZZ += volJ * dvz * gz;
                        }
                    }
                } else {
                    float minX, minY, minZ, maxX, maxY, maxZ;
                    if constexpr (UsePacked) {
                        const float4 mn = __ldg(&bvh.nodeMinExtent[c]);
                        const float4 mx = __ldg(&bvh.nodeMaxHPacked[c]);
                        minX = mn.x; minY = mn.y; minZ = mn.z;
                        maxX = mx.x; maxY = mx.y; maxZ = mx.z;
                    } else {
                        minX = __ldg(&bvh.aabbMinX[c]); minY = __ldg(&bvh.aabbMinY[c]); minZ = __ldg(&bvh.aabbMinZ[c]);
                        maxX = __ldg(&bvh.aabbMaxX[c]); maxY = __ldg(&bvh.aabbMaxY[c]); maxZ = __ldg(&bvh.aabbMaxZ[c]);
                    }
                    const bool overlap = active && (pointAabbDistSq(xi, yi, zi, minX, minY, minZ, maxX, maxY, maxZ) <= Hi2);
                    if (__any_sync(0xFFFFFFFFu, overlap) && sp < kBvhStackSize) {
                        if (lane == 0) {
                            sStack[warpId][sp] = c;
                        }
                        ++sp;
                        __syncwarp();
                    }
                }
            }
        }
    }

    if (!active) {
        return;
    }

    p.shepardC[i] = shepardC;
    const float normNs = sqrtf(nsX * nsX + nsY * nsY + nsZ * nsZ);
    float fSurf = fminf(1.0f, fmaxf(0.0f, 1.6f * fmaxf(1.0f - shepardC, normNs / hi) - 0.15f));
    if (ngbCount <= 2) {
        fSurf = 1.0f;
    }
    p.surfFactor[i] = fSurf;

    p.divV[i]   = gXX + gYY + gZZ;
    p.curlVX[i] = gZY - gYZ;
    p.curlVY[i] = gXZ - gZX;
    p.curlVZ[i] = gYX - gXY;

    const float ratio = 56.0f / static_cast<float>(ngbCount);
    const float factor = fminf(1.08f, fmaxf(0.92f, cbrtf(ratio)));
    p.h[i] = fminf(6.0f * h0i, fmaxf(0.35f * h0i, hi * factor));
}

// ---------- EOS Kernel ----------

__global__ void evaluateEosKernel(ParticleBuffersDevice p, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const MaterialId mat = static_cast<MaterialId>(__ldg(&p.materialId[i]));
    const float rhoI = __ldg(&p.rho[i]);
    const float uI = __ldg(&p.u[i]);
    const float damI = __ldg(&p.damage[i]);
    const astraea::eos::EosEvalResult res = astraea::eos::evaluateEosPsu(mat, rhoI, uI, damI);
    p.pressure[i]   = res.pressurePsu;
    p.soundSpeed[i] = res.soundSpeedPsu;
    p.tempK[i]      = res.tempK;
    p.phaseId[i]    = res.phaseId;
}

// ---------- Barnes-Hut Self-Gravity Kernel (Warp-Cooperative Shared Stack) ----------

template <bool UsePacked>
__global__ void gravityBarnesHutKernel(
    ParticleBuffersDevice p,
    RadixBvhDevice bvh,
    uint32_t n,
    float theta2) {
    __shared__ int sStack[kWarpsPerBlock][kBvhStackSize];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (i < n);
    if (__ballot_sync(0xFFFFFFFFu, active) == 0u) {
        return;
    }

    if (n == 1u) {
        if (active) {
            p.accX[0] = 0.0f;
            p.accY[0] = 0.0f;
            p.accZ[0] = 0.0f;
            p.phiGrav[0] = 0.0f;
        }
        return;
    }

    const int warpId = static_cast<int>(threadIdx.x >> 5);
    const int lane   = static_cast<int>(threadIdx.x & 31);

    float xi = 0.0f, yi = 0.0f, zi = 0.0f, epsI = 1.0e-6f, epsI2 = 1.0e-12f;
    if (active) {
        xi = __ldg(&p.posX[i]);
        yi = __ldg(&p.posY[i]);
        zi = __ldg(&p.posZ[i]);
        epsI = 0.8f * fmaxf(__ldg(&p.h0[i]), 1.0e-6f);
        epsI2 = epsI * epsI;
    }

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float phi = 0.0f;

    if (lane == 0) {
        sStack[warpId][0] = 0;
    }
    int sp = 1;
    __syncwarp();

    const bool checkInside = (theta2 > 0.333333f);

    while (sp > 0) {
        const int node = sStack[warpId][--sp];
        const int children[2] = {
            __ldg(&bvh.leftChild[node]),
            __ldg(&bvh.rightChild[node])
        };

        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int c = children[k];
            if (c < 0) {
                const uint32_t j = static_cast<uint32_t>(~c);
                const float pxj  = __ldg(&p.posX[j]);
                const float pyj  = __ldg(&p.posY[j]);
                const float pzj  = __ldg(&p.posZ[j]);
                const float mj   = __ldg(&p.mass[j]);
                const float epsJ = 0.8f * fmaxf(__ldg(&p.h0[j]), 1.0e-6f);

                if (active && j != i) {
                    const float dx = xi - pxj;
                    const float dy = yi - pyj;
                    const float dz = zi - pzj;
                    const float r2 = dx * dx + dy * dy + dz * dz;
                    const float epsIJ = 0.5f * (epsI + epsJ);
                    const float denom2 = r2 + epsIJ * epsIJ + 1.0e-20f;
                    const float invR = rsqrtf(denom2);
                    const float invR3 = invR * invR * invR;
                    const float f = -astraea::units::G_PSU_F * mj * invR3;
                    ax += f * dx;
                    ay += f * dy;
                    az += f * dz;
                    phi -= astraea::units::G_PSU_F * mj * invR;
                }
            } else {
                float cx, cy, cz, mNode, dNode;
                if constexpr (UsePacked) {
                    const float4 cm = __ldg(&bvh.nodeComMass[c]);
                    const float4 me = __ldg(&bvh.nodeMinExtent[c]);
                    cx = cm.x; cy = cm.y; cz = cm.z; mNode = cm.w;
                    dNode = me.w;
                } else {
                    cx = __ldg(&bvh.comX[c]);
                    cy = __ldg(&bvh.comY[c]);
                    cz = __ldg(&bvh.comZ[c]);
                    mNode = __ldg(&bvh.nodeMass[c]);
                    const float minX = __ldg(&bvh.aabbMinX[c]);
                    const float minY = __ldg(&bvh.aabbMinY[c]);
                    const float minZ = __ldg(&bvh.aabbMinZ[c]);
                    const float maxX = __ldg(&bvh.aabbMaxX[c]);
                    const float maxY = __ldg(&bvh.aabbMaxY[c]);
                    const float maxZ = __ldg(&bvh.aabbMaxZ[c]);
                    dNode = fmaxf(maxX - minX, fmaxf(maxY - minY, maxZ - minZ));
                }

                float dx = 0.0f, dy = 0.0f, dz = 0.0f, r2 = 0.0f;
                bool canAccept = false;
                if (active) {
                    dx = xi - cx;
                    dy = yi - cy;
                    dz = zi - cz;
                    r2 = dx * dx + dy * dy + dz * dz;
                    canAccept = (dNode * dNode < theta2 * r2) && (r2 > 1.0e-20f);
                    if (canAccept && checkInside) {
                        const float minX = UsePacked ? __ldg(&bvh.nodeMinExtent[c]).x : __ldg(&bvh.aabbMinX[c]);
                        const float minY = UsePacked ? __ldg(&bvh.nodeMinExtent[c]).y : __ldg(&bvh.aabbMinY[c]);
                        const float minZ = UsePacked ? __ldg(&bvh.nodeMinExtent[c]).z : __ldg(&bvh.aabbMinZ[c]);
                        const float maxX = UsePacked ? __ldg(&bvh.nodeMaxHPacked[c]).x : __ldg(&bvh.aabbMaxX[c]);
                        const float maxY = UsePacked ? __ldg(&bvh.nodeMaxHPacked[c]).y : __ldg(&bvh.aabbMaxY[c]);
                        const float maxZ = UsePacked ? __ldg(&bvh.nodeMaxHPacked[c]).z : __ldg(&bvh.aabbMaxZ[c]);
                        if (xi >= minX && xi <= maxX && yi >= minY && yi <= maxY && zi >= minZ && zi <= maxZ) {
                            canAccept = false;
                        }
                    }
                }

                const bool anyOpen = __any_sync(0xFFFFFFFFu, active && !canAccept);
                if (!anyOpen || sp >= kBvhStackSize) {
                    if (active) {
                        const float denom2 = r2 + epsI2 + 1.0e-20f;
                        const float invR = rsqrtf(denom2);
                        const float invR3 = invR * invR * invR;
                        const float f = -astraea::units::G_PSU_F * mNode * invR3;
                        ax += f * dx;
                        ay += f * dy;
                        az += f * dz;
                        phi -= astraea::units::G_PSU_F * mNode * invR;
                    }
                } else {
                    if (lane == 0) {
                        sStack[warpId][sp] = c;
                    }
                    ++sp;
                    __syncwarp();
                }
            }
        }
    }

    if (active) {
        p.accX[i] = ax;
        p.accY[i] = ay;
        p.accZ[i] = az;
        p.phiGrav[i] = phi;
    }
}

// ---------- SPH Pass 2: Pressure, delta-SPH, AV, Strength, Damage, Radiative Cooling ----------

template <bool UsePacked>
__global__ void sphPass2Kernel(
    ParticleBuffersDevice p,
    RadixBvhDevice bvh,
    uint32_t n,
    const uint32_t* __restrict__ livingFlags) {
    __shared__ int sStack[kWarpsPerBlock][kBvhStackSize];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (i < n);
    if (__ballot_sync(0xFFFFFFFFu, active) == 0u) {
        return;
    }

    const int warpId = static_cast<int>(threadIdx.x >> 5);
    const int lane   = static_cast<int>(threadIdx.x & 31);

    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    float vxi = 0.0f, vyi = 0.0f, vzi = 0.0f;
    float mi = 0.0f, hi = 1.0e-6f, Hi = 0.0f;
    float rhoI = 1.0f, invRhoI2 = 1.0f, pI = 0.0f, csI = 0.05f;
    float tempI = 3.0f, damI = 0.0f, fSurf = 0.0f;
    uint8_t matI = 1u;
    float sXXi = 0.0f, sYYi = 0.0f, sZZi = 0.0f;
    float sXYi = 0.0f, sXZi = 0.0f, sYZi = 0.0f;
    float fBalI = 0.0f, Ri = 0.0f, pOverRho2I = 0.0f;

    if (active) {
        xi  = __ldg(&p.posX[i]);
        yi  = __ldg(&p.posY[i]);
        zi  = __ldg(&p.posZ[i]);
        vxi = __ldg(&p.velX[i]);
        vyi = __ldg(&p.velY[i]);
        vzi = __ldg(&p.velZ[i]);
        mi  = __ldg(&p.mass[i]);
        hi  = fmaxf(__ldg(&p.h[i]), 1.0e-6f);
        Hi  = 2.0f * hi;
        rhoI = fmaxf(__ldg(&p.rho[i]), 1.0e-9f);
        invRhoI2 = 1.0f / (rhoI * rhoI);
        pI = __ldg(&p.pressure[i]);
        csI = fmaxf(__ldg(&p.soundSpeed[i]), 0.05f);
        tempI = __ldg(&p.tempK[i]);
        damI = __ldg(&p.damage[i]);
        fSurf = __ldg(&p.surfFactor[i]);
        matI = __ldg(&p.materialId[i]);

        sXXi = __ldg(&p.sXX[i]);
        sYYi = __ldg(&p.sYY[i]);
        sZZi = __ldg(&p.sZZ[i]);
        sXYi = __ldg(&p.sXY[i]);
        sXZi = __ldg(&p.sXZ[i]);
        sYZi = __ldg(&p.sYZ[i]);

        fBalI = computeBalsaraSwitch(
            __ldg(&p.divV[i]), __ldg(&p.curlVX[i]), __ldg(&p.curlVY[i]), __ldg(&p.curlVZ[i]),
            csI, hi);
        Ri = (pI < 0.0f) ? (0.3f * (-pI) * invRhoI2) : 0.0f;
        pOverRho2I = pI * invRhoI2;
    }

    float axSph = 0.0f, aySph = 0.0f, azSph = 0.0f;
    float drhoCont = 0.0f;
    float drhoDiff = 0.0f;
    float duP = 0.0f;
    float duAv = 0.0f;
    float duTens = 0.0f;
    float duStressNumer = 0.0f;
    float gXX = 0.0f, gXY = 0.0f, gXZ = 0.0f;
    float gYX = 0.0f, gYY = 0.0f, gYZ = 0.0f;
    float gZX = 0.0f, gZY = 0.0f, gZZ = 0.0f;

    if (n >= 2u) {
        if (lane == 0) {
            sStack[warpId][0] = 0;
        }
        int sp = 1;
        __syncwarp();

        while (sp > 0) {
            const int node = sStack[warpId][--sp];
            const int children[2] = {
                __ldg(&bvh.leftChild[node]),
                __ldg(&bvh.rightChild[node])
            };

            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int c = children[k];
                if (c < 0) {
                    const uint32_t j = static_cast<uint32_t>(~c);
                    const float pxj = __ldg(&p.posX[j]);
                    const float pyj = __ldg(&p.posY[j]);
                    const float pzj = __ldg(&p.posZ[j]);
                    const float hj  = fmaxf(__ldg(&p.h[j]), 1.0e-6f);
                    const float Hj  = 2.0f * hj;
                    const float dx  = xi - pxj;
                    const float dy  = yi - pyj;
                    const float dz  = zi - pzj;
                    const float r2  = dx * dx + dy * dy + dz * dz;
                    const float Hmax = fmaxf(Hi, Hj);
                    const bool inRange = active && (j != i) && (r2 < Hmax * Hmax);

                    if (__any_sync(0xFFFFFFFFu, inRange)) {
                        const float mj   = __ldg(&p.mass[j]);
                        const float rhoJ = fmaxf(__ldg(&p.rho[j]), 1.0e-9f);
                        const float pJ   = __ldg(&p.pressure[j]);
                        const float csJ  = fmaxf(__ldg(&p.soundSpeed[j]), 0.05f);
                        const float vxj  = __ldg(&p.velX[j]);
                        const float vyj  = __ldg(&p.velY[j]);
                        const float vzj  = __ldg(&p.velZ[j]);
                        const uint8_t matJ = __ldg(&p.materialId[j]);
                        const float sXXj = __ldg(&p.sXX[j]);
                        const float sYYj = __ldg(&p.sYY[j]);
                        const float sZZj = __ldg(&p.sZZ[j]);
                        const float sXYj = __ldg(&p.sXY[j]);
                        const float sXZj = __ldg(&p.sXZ[j]);
                        const float sYZj = __ldg(&p.sYZ[j]);

                        const float dvx = vxi - vxj;
                        const float dvy = vyi - vyj;
                        const float dvz = vzi - vzj;
                        const float vDotR = dvx * dx + dvy * dy + dvz * dz;

                        float fBalJ = 0.0f;
                        if (__any_sync(0xFFFFFFFFu, inRange && (vDotR < 0.0f))) {
                            fBalJ = computeBalsaraSwitch(
                                __ldg(&p.divV[j]), __ldg(&p.curlVX[j]), __ldg(&p.curlVY[j]), __ldg(&p.curlVZ[j]),
                                csJ, hj);
                        }

                        if (inRange) {
                            const float r = sqrtf(r2);
                            const float gCoeffI = wendlandC2GradCoeff(r, Hi);
                            const float gCoeffJ = wendlandC2GradCoeff(r, Hj);
                            const float gBar = 0.5f * (gCoeffI + gCoeffJ);

                            const float gxBar = gBar * dx;
                            const float gyBar = gBar * dy;
                            const float gzBar = gBar * dz;

                            const float invRhoJ2 = 1.0f / (rhoJ * rhoJ);
                            const float vDotGradBar = dvx * gxBar + dvy * gyBar + dvz * gzBar;

                            drhoCont += (mj / rhoJ) * vDotGradBar;
                            if (matJ == matI) {
                                const float rDotGradBar = -(dx * gxBar + dy * gyBar + dz * gzBar);
                                const float diffDenom = r2 + 0.01f * Hi * Hi;
                                drhoDiff += (mj / rhoJ) * (2.0f * (rhoJ - rhoI) / diffDenom) * rDotGradBar;
                            }

                            float piIJ = 0.0f;
                            if (vDotR < 0.0f) {
                                const float hBar = 0.5f * (hi + hj);
                                const float cBar = 0.5f * (csI + csJ);
                                const float rhoBar = 0.5f * (rhoI + rhoJ);
                                const float muIJ = (hBar * vDotR) / (r2 + 0.01f * hBar * hBar);
                                piIJ = ((-1.0f * cBar * muIJ + 2.0f * muIJ * muIJ) / rhoBar) * (0.5f * (fBalI + fBalJ));
                            }

                            float tensTerm = 0.0f;
                            const float Rj = (pJ < 0.0f) ? (0.3f * (-pJ) * invRhoJ2) : 0.0f;
                            if (Ri + Rj > 0.0f) {
                                const float Hbar = 0.5f * (Hi + Hj);
                                if (r < Hbar) {
                                    const float qBar = r / Hbar;
                                    const float omq = 1.0f - qBar;
                                    const float omq2 = omq * omq;
                                    const float fRatio = (omq2 * omq2 * (1.0f + 4.0f * qBar)) * (1.0f / 0.6328125f);
                                    const float fRatio2 = fRatio * fRatio;
                                    tensTerm = (Ri + Rj) * (fRatio2 * fRatio2);
                                }
                            }

                            const float pTerm = pOverRho2I + pJ * invRhoJ2;
                            const float isoCoeff = -mj * (pTerm + piIJ + tensTerm);
                            axSph += isoCoeff * gxBar;
                            aySph += isoCoeff * gyBar;
                            azSph += isoCoeff * gzBar;

                            duP    += mj * vDotGradBar;
                            duAv   += 0.5f * mj * piIJ * vDotGradBar;
                            duTens += 0.5f * mj * tensTerm * vDotGradBar;

                            const float stXX = sXXi * invRhoI2 + sXXj * invRhoJ2;
                            const float stYY = sYYi * invRhoI2 + sYYj * invRhoJ2;
                            const float stZZ = sZZi * invRhoI2 + sZZj * invRhoJ2;
                            const float stXY = sXYi * invRhoI2 + sXYj * invRhoJ2;
                            const float stXZ = sXZi * invRhoI2 + sXZj * invRhoJ2;
                            const float stYZ = sYZi * invRhoI2 + sYZj * invRhoJ2;

                            axSph += mj * (stXX * gxBar + stXY * gyBar + stXZ * gzBar);
                            aySph += mj * (stXY * gxBar + stYY * gyBar + stYZ * gzBar);
                            azSph += mj * (stXZ * gxBar + stYZ * gyBar + stZZ * gzBar);

                            const float vjMinusViX = -dvx;
                            const float vjMinusViY = -dvy;
                            const float vjMinusViZ = -dvz;
                            duStressNumer += mj * (
                                (sXXi * vjMinusViX + sXYi * vjMinusViY + sXZi * vjMinusViZ) * gxBar +
                                (sXYi * vjMinusViX + sYYi * vjMinusViY + sYZi * vjMinusViZ) * gyBar +
                                (sXZi * vjMinusViX + sYZi * vjMinusViY + sZZi * vjMinusViZ) * gzBar);

                            const float volJ = mj / rhoJ;
                            const float gxi = gCoeffI * dx;
                            const float gyi = gCoeffI * dy;
                            const float gzi = gCoeffI * dz;
                            gXX += volJ * vjMinusViX * gxi; gXY += volJ * vjMinusViX * gyi; gXZ += volJ * vjMinusViX * gzi;
                            gYX += volJ * vjMinusViY * gxi; gYY += volJ * vjMinusViY * gyi; gYZ += volJ * vjMinusViY * gzi;
                            gZX += volJ * vjMinusViZ * gxi; gZY += volJ * vjMinusViZ * gyi; gZZ += volJ * vjMinusViZ * gzi;
                        }
                    }
                } else {
                    float minX, minY, minZ, maxX, maxY, maxZ, nodeH;
                    if constexpr (UsePacked) {
                        const float4 mn = __ldg(&bvh.nodeMinExtent[c]);
                        const float4 mx = __ldg(&bvh.nodeMaxHPacked[c]);
                        minX = mn.x; minY = mn.y; minZ = mn.z;
                        maxX = mx.x; maxY = mx.y; maxZ = mx.z;
                        nodeH = mx.w;
                    } else {
                        minX = __ldg(&bvh.aabbMinX[c]); minY = __ldg(&bvh.aabbMinY[c]); minZ = __ldg(&bvh.aabbMinZ[c]);
                        maxX = __ldg(&bvh.aabbMaxX[c]); maxY = __ldg(&bvh.aabbMaxY[c]); maxZ = __ldg(&bvh.aabbMaxZ[c]);
                        nodeH = __ldg(&bvh.nodeMaxH[c]);
                    }
                    const float rad = fmaxf(Hi, 1.09f * nodeH);
                    const bool overlap = active && (pointAabbDistSq(xi, yi, zi, minX, minY, minZ, maxX, maxY, maxZ) <= rad * rad);
                    if (__any_sync(0xFFFFFFFFu, overlap) && sp < kBvhStackSize) {
                        if (lane == 0) {
                            sStack[warpId][sp] = c;
                        }
                        ++sp;
                        __syncwarp();
                    }
                }
            }
        }
    }

    if (!active) {
        return;
    }

    p.accX[i] += axSph;
    p.accY[i] += aySph;
    p.accZ[i] += azSph;

    const float diffMask = (__ldg(&p.shepardC[i]) > 0.85f) ? (1.0f - fSurf) : 0.0f;
    p.drhoDt[i] = rhoI * drhoCont + 0.05f * Hi * csI * diffMask * drhoDiff;

    const float duStress = invRhoI2 * duStressNumer;

    // Surface radiative cooling
    const bool livingIntact = (livingFlags != nullptr) ? (livingFlags[1] != 0u) : false;
    const double tAmb = (livingIntact && __ldg(&p.bodyId[i]) == 0u && damI < 0.8f) ? 255.0 : 3.0;
    const double mSi = fmax(static_cast<double>(mi) * astraea::units::M_UNIT_KG, 1.0);
    const double rhoSi = fmax(static_cast<double>(rhoI) * astraea::units::RHO_UNIT_KGM3, 1.0e-3);
    const double rPartSi = cbrt((3.0 * mSi) / (4.0 * 3.14159265358979323846 * rhoSi));
    const double areaRadSi = 3.14159265358979323846 * rPartSi * rPartSi;
    const double T = static_cast<double>(tempI);
    const double T4 = T * T * T * T;
    const double Tamb4 = tAmb * tAmb * tAmb * tAmb;
    const double dT4 = fmax(0.0, T4 - Tamb4);
    const double pRadW = static_cast<double>(fSurf) * 0.9 * astraea::units::SIGMA_SB_SI * areaRadSi * dT4;
    p.radPowerW[i] = static_cast<float>(pRadW);
    const float duRadPsu = static_cast<float>(-pRadW / (fmax(static_cast<double>(mi), 1.0e-30) * 1.0e27));

    p.duDt[i] = pOverRho2I * duP + duAv + duTens + duStress + duRadPsu;

    // Strength: Jaumann stress rate + Grady-Kipp brittle damage evolution
    const uint8_t matIdx = (matI <= 4u) ? matI : 1u;
    const MatStrengthConstants st = kMatStrength[matIdx];
    if (st.muShearPsu <= 0.0f || st.tMelt0K <= 0.0f) {
        p.dSXX[i] = 0.0f; p.dSYY[i] = 0.0f; p.dSZZ[i] = 0.0f;
        p.dSXY[i] = 0.0f; p.dSXZ[i] = 0.0f; p.dSYZ[i] = 0.0f;
        p.dDamageCbrtDt[i] = 0.0f;
        return;
    }

    const float muShearPsu = st.muShearPsu;
    const float tMelt = st.tMelt0K;
    const float tNorm = tempI / tMelt;
    const float thermalSoft = fmaxf(0.0f, 1.0f - tNorm * tNorm);
    const float muEff = (1.0f - damI) * muShearPsu * thermalSoft;

    const float trG3 = (gXX + gYY + gZZ) * (1.0f / 3.0f);
    const float epsXX = gXX - trG3;
    const float epsYY = gYY - trG3;
    const float epsZZ = gZZ - trG3;
    const float epsXY = 0.5f * (gXY + gYX);
    const float epsXZ = 0.5f * (gXZ + gZX);
    const float epsYZ = 0.5f * (gYZ + gZY);

    const float rXY = 0.5f * (gXY - gYX);
    const float rXZ = 0.5f * (gXZ - gZX);
    const float rYZ = 0.5f * (gYZ - gZY);
    const float rYX = -rXY;
    const float rZX = -rXZ;
    const float rZY = -rYZ;

    p.dSXX[i] = 2.0f * muEff * epsXX + 2.0f * (sXYi * rXY + sXZi * rXZ);
    p.dSYY[i] = 2.0f * muEff * epsYY + 2.0f * (sXYi * rYX + sYZi * rYZ);
    p.dSZZ[i] = 2.0f * muEff * epsZZ + 2.0f * (sXZi * rZX + sYZi * rZY);
    p.dSXY[i] = 2.0f * muEff * epsXY + sXXi * rYX + sXZi * rYZ + sYYi * rXY + sYZi * rXZ;
    p.dSXZ[i] = 2.0f * muEff * epsXZ + sXXi * rZX + sXYi * rZY + sZZi * rXZ + sYZi * rXY;
    p.dSYZ[i] = 2.0f * muEff * epsYZ + sXYi * rZX + sYYi * rZY + sXZi * rYX + sZZi * rYZ;

    if (tempI < tMelt && damI < 1.0f) {
        const float sigmaMax = maxPrincipalStressFromTraceless(pI, sXXi, sYYi, sZZi, sXYi, sXZi, sYZi);
        const float denomStiff = fmaxf(st.bulkModPsu + (4.0f / 3.0f) * muShearPsu, 1.0e-12f);
        const float epsTens = fmaxf(0.0f, sigmaMax) / denomStiff;
        const float epsFailI = fmaxf(__ldg(&p.epsFail[i]), 1.0e-8f);
        const float cCrack = fmaxf(csI, sqrtf((st.bulkModPsu + (4.0f / 3.0f) * muShearPsu) / rhoI));
        p.dDamageCbrtDt[i] = (epsTens > epsFailI) ? (0.4f * cCrack / hi) : 0.0f;
    } else {
        p.dDamageCbrtDt[i] = 0.0f;
    }
}

// ---------- KDK Leapfrog Half-Kick & Drift Kernels ----------

__global__ void leapfrogKickHalfKernel(
    ParticleBuffersDevice p,
    uint32_t n,
    float halfDtPsu,
    bool includeXsph,
    bool clampThermal) {
    (void)includeXsph;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float vx = p.velX[i] + halfDtPsu * p.accX[i];
    const float vy = p.velY[i] + halfDtPsu * p.accY[i];
    const float vz = p.velZ[i] + halfDtPsu * p.accZ[i];
    p.velX[i] = vx;     p.velY[i] = vy;     p.velZ[i] = vz;
    p.velHalfX[i] = vx; p.velHalfY[i] = vy; p.velHalfZ[i] = vz;

    const float rhoNew = fmaxf(p.rho[i] + halfDtPsu * p.drhoDt[i], 1.0e-9f);
    p.rho[i] = rhoNew;

    const uint8_t matRaw = p.materialId[i];
    const uint8_t matIdx = (matRaw <= 4u) ? matRaw : 1u;
    const MaterialId mat = static_cast<MaterialId>(matIdx);
    const MatStrengthConstants st = kMatStrength[matIdx];

    float uNew = p.u[i] + halfDtPsu * p.duDt[i];
    if (clampThermal) {
        const astraea::eos::TillotsonParams& params = astraea::eos::getParams(mat);
        const double cvPsu = params.cv_JKgK / astraea::units::U_UNIT_JKG;
        const double uMin = astraea::eos::computeColdEnergyPsu(mat, static_cast<double>(rhoNew)) + 3.0 * cvPsu;
        uNew = fmaxf(uNew, static_cast<float>(uMin));
    } else {
        uNew = fmaxf(uNew, 0.0f);
    }
    p.u[i] = uNew;

    float dCbrt = p.damageCbrt[i] + halfDtPsu * p.dDamageCbrtDt[i];
    dCbrt = fminf(1.0f, fmaxf(p.damageCbrt[i], dCbrt));
    const float D = dCbrt * dCbrt * dCbrt;
    p.damageCbrt[i] = dCbrt;
    p.damage[i] = D;

    if (clampThermal) {
        const astraea::eos::EosEvalResult eosRes = astraea::eos::evaluateEosPsu(mat, rhoNew, uNew, D);
        p.pressure[i]   = eosRes.pressurePsu;
        p.soundSpeed[i] = eosRes.soundSpeedPsu;
        p.tempK[i]      = eosRes.tempK;
        p.phaseId[i]    = eosRes.phaseId;
    }

    if (st.muShearPsu <= 0.0f || st.tMelt0K <= 0.0f || p.tempK[i] >= st.tMelt0K) {
        p.sXX[i] = 0.0f; p.sYY[i] = 0.0f; p.sZZ[i] = 0.0f;
        p.sXY[i] = 0.0f; p.sXZ[i] = 0.0f; p.sYZ[i] = 0.0f;
        return;
    }

    float sXX = p.sXX[i] + halfDtPsu * p.dSXX[i];
    float sYY = p.sYY[i] + halfDtPsu * p.dSYY[i];
    float sZZ = p.sZZ[i] + halfDtPsu * p.dSZZ[i];
    float sXY = p.sXY[i] + halfDtPsu * p.dSXY[i];
    float sXZ = p.sXZ[i] + halfDtPsu * p.dSXZ[i];
    float sYZ = p.sYZ[i] + halfDtPsu * p.dSYZ[i];

    const float tr3 = (sXX + sYY + sZZ) * (1.0f / 3.0f);
    sXX -= tr3; sYY -= tr3; sZZ -= tr3;

    const float y0 = st.yieldY0Psu;
    const float tRatio = p.tempK[i] / st.tMelt0K;
    const float yIntact = (1.0f - D) * y0 * fmaxf(0.0f, 1.0f - tRatio);
    const float yFriction = D * fminf(y0, 0.6f * fmaxf(0.0f, p.pressure[i]));
    const float Y = fmaxf(yIntact, yFriction);

    const float j2 = 0.5f * (sXX * sXX + sYY * sYY + sZZ * sZZ + 2.0f * (sXY * sXY + sXZ * sXZ + sYZ * sYZ));
    const float eqStress = sqrtf(3.0f * j2 + 1.0e-30f);
    const float scale = (Y <= 0.0f) ? 0.0f : fminf(1.0f, Y / eqStress);

    p.sXX[i] = sXX * scale;
    p.sYY[i] = sYY * scale;
    p.sZZ[i] = sZZ * scale;
    p.sXY[i] = sXY * scale;
    p.sXZ[i] = sXZ * scale;
    p.sYZ[i] = sYZ * scale;
}

template <bool UsePacked>
__global__ void leapfrogDriftKernel(
    ParticleBuffersDevice p,
    RadixBvhDevice bvh,
    uint32_t n,
    float dtPsu,
    bool useBvhXsph) {
    __shared__ int sStack[kWarpsPerBlock][kBvhStackSize];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (i < n);
    if (__ballot_sync(0xFFFFFFFFu, active) == 0u) {
        return;
    }

    const int warpId = static_cast<int>(threadIdx.x >> 5);
    const int lane   = static_cast<int>(threadIdx.x & 31);

    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    float vxi = 0.0f, vyi = 0.0f, vzi = 0.0f;
    float hi = 1.0e-6f, Hi = 0.0f, rhoI = 1.0f;
    if (active) {
        xi  = __ldg(&p.posX[i]);
        yi  = __ldg(&p.posY[i]);
        zi  = __ldg(&p.posZ[i]);
        vxi = __ldg(&p.velX[i]);
        vyi = __ldg(&p.velY[i]);
        vzi = __ldg(&p.velZ[i]);
        hi  = fmaxf(__ldg(&p.h[i]), 1.0e-6f);
        Hi  = 2.0f * hi;
        rhoI = fmaxf(__ldg(&p.rho[i]), 1.0e-9f);
    }

    float xsphX = 0.0f, xsphY = 0.0f, xsphZ = 0.0f;

    if (useBvhXsph && n >= 2u) {
        if (lane == 0) {
            sStack[warpId][0] = 0;
        }
        int sp = 1;
        __syncwarp();

        while (sp > 0) {
            const int node = sStack[warpId][--sp];
            const int children[2] = {
                __ldg(&bvh.leftChild[node]),
                __ldg(&bvh.rightChild[node])
            };

            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int c = children[k];
                if (c < 0) {
                    const uint32_t j = static_cast<uint32_t>(~c);
                    const float pxj  = __ldg(&p.posX[j]);
                    const float pyj  = __ldg(&p.posY[j]);
                    const float pzj  = __ldg(&p.posZ[j]);
                    const float Hj   = 2.0f * fmaxf(__ldg(&p.h[j]), 1.0e-6f);
                    const float dx = xi - pxj;
                    const float dy = yi - pyj;
                    const float dz = zi - pzj;
                    const float r2 = dx * dx + dy * dy + dz * dz;
                    const float Hmax = fmaxf(Hi, Hj);
                    const bool inRange = active && (j != i) && (r2 < Hmax * Hmax);

                    if (__any_sync(0xFFFFFFFFu, inRange)) {
                        const float mj   = __ldg(&p.mass[j]);
                        const float rhoj = fmaxf(__ldg(&p.rho[j]), 1.0e-9f);
                        const float vxj  = __ldg(&p.velX[j]);
                        const float vyj  = __ldg(&p.velY[j]);
                        const float vzj  = __ldg(&p.velZ[j]);
                        if (inRange) {
                            const float r = sqrtf(r2);
                            const float wBar = 0.5f * (wendlandC2(r, Hi) + wendlandC2(r, Hj));
                            const float rhoBar = 0.5f * (rhoI + rhoj);
                            const float coeff = (mj / rhoBar) * wBar;
                            xsphX += coeff * (vxj - vxi);
                            xsphY += coeff * (vyj - vyi);
                            xsphZ += coeff * (vzj - vzi);
                        }
                    }
                } else {
                    float minX, minY, minZ, maxX, maxY, maxZ, nodeH;
                    if constexpr (UsePacked) {
                        const float4 mn = __ldg(&bvh.nodeMinExtent[c]);
                        const float4 mx = __ldg(&bvh.nodeMaxHPacked[c]);
                        minX = mn.x; minY = mn.y; minZ = mn.z;
                        maxX = mx.x; maxY = mx.y; maxZ = mx.z;
                        nodeH = mx.w;
                    } else {
                        minX = __ldg(&bvh.aabbMinX[c]); minY = __ldg(&bvh.aabbMinY[c]); minZ = __ldg(&bvh.aabbMinZ[c]);
                        maxX = __ldg(&bvh.aabbMaxX[c]); maxY = __ldg(&bvh.aabbMaxY[c]); maxZ = __ldg(&bvh.aabbMaxZ[c]);
                        nodeH = __ldg(&bvh.nodeMaxH[c]);
                    }
                    const float rad = fmaxf(Hi, 1.09f * nodeH);
                    const bool overlap = active && (pointAabbDistSq(xi, yi, zi, minX, minY, minZ, maxX, maxY, maxZ) <= rad * rad);
                    if (__any_sync(0xFFFFFFFFu, overlap) && sp < kBvhStackSize) {
                        if (lane == 0) {
                            sStack[warpId][sp] = c;
                        }
                        ++sp;
                        __syncwarp();
                    }
                }
            }
        }
    }

    if (active) {
        p.posX[i] = xi + dtPsu * (vxi + 0.25f * xsphX);
        p.posY[i] = yi + dtPsu * (vyi + 0.25f * xsphY);
        p.posZ[i] = zi + dtPsu * (vzi + 0.25f * xsphZ);
    }
}

// ---------- Adaptive Timestep Reduction Kernel ----------

__global__ void reduceDtMinKernel(
    ParticleBuffersDevice p,
    uint32_t n,
    uint32_t* __restrict__ outDtMinBits) {
    __shared__ float sMin[kBlockSize];

    float localMin = 1.0e30f;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = tid; i < n; i += stride) {
        const float hi  = fmaxf(__ldg(&p.h[i]), 1.0e-6f);
        const float csI = fmaxf(__ldg(&p.soundSpeed[i]), 0.05f);
        const float vx = __ldg(&p.velX[i]);
        const float vy = __ldg(&p.velY[i]);
        const float vz = __ldg(&p.velZ[i]);
        const float vMag = sqrtf(vx * vx + vy * vy + vz * vz);
        const float dtCfl = 0.25f * hi / (2.2f * csI + vMag + 1.0e-6f);

        const float ax = __ldg(&p.accX[i]);
        const float ay = __ldg(&p.accY[i]);
        const float az = __ldg(&p.accZ[i]);
        const float aMag = sqrtf(ax * ax + ay * ay + az * az);
        const float dtAcc = 0.25f * sqrtf(hi / (aMag + 1.0e-8f));

        uint8_t matIdx = __ldg(&p.materialId[i]);
        if (matIdx > 4u) {
            matIdx = 1u;
        }
        const float uI = fmaxf(__ldg(&p.u[i]), 0.0f);
        const float uScale = fmaxf(uI, kU0RefPsu[matIdx]);
        const float absDuDt = fabsf(__ldg(&p.duDt[i]));
        const float dtTherm = 0.20f * (uI + 0.01f * uScale) / (absDuDt + 1.0e-8f);

        const float dtI = fminf(dtCfl, fminf(dtAcc, dtTherm));
        if (isfinite(dtI) && dtI > 0.0f) {
            localMin = fminf(localMin, dtI);
        }
    }

    const int lt = threadIdx.x;
    sMin[lt] = localMin;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (lt < offset) {
            sMin[lt] = fminf(sMin[lt], sMin[lt + offset]);
        }
        __syncthreads();
    }

    if (lt == 0) {
        const float val = sMin[0];
        if (isfinite(val) && val > 0.0f) {
            atomicMin(outDtMinBits, __float_as_uint(val));
        }
    }
}

// ---------- Tracer Extraction & Binary Render Packing Kernels ----------

struct __align__(4) CrustTracerRecordPacked {
    uint16_t anchorIndex;
    uint16_t flags;
    float    xKm;
    float    yKm;
    float    zKm;
    float    dispKm;
    float    normalDelta;
    float    tempK;
    float    damage;
    float    overpressureKpa;
    float    tsunamiHeightM;
};
static_assert(sizeof(CrustTracerRecordPacked) == 40, "CrustTracerRecordPacked must be 40 bytes");

struct __align__(4) ParticleRecordPacked {
    float    xKm;
    float    yKm;
    float    zKm;
    uint16_t tempLogU16;
    uint16_t pressureLogU16;
    uint8_t  materialId;
    uint8_t  phaseAndBody;
    uint8_t  damageU8;
    uint8_t  radiusLogU8;
};
static_assert(sizeof(ParticleRecordPacked) == 20, "ParticleRecordPacked must be 20 bytes");

__device__ __forceinline__ float finiteOrZero(float v) {
    return isfinite(v) ? v : 0.0f;
}

__global__ void extractTracersKernel(
    ParticleBuffersDevice p,
    uint32_t n,
    uint8_t* __restrict__ d_tracerRecords40B,
    uint32_t tracerCount,
    float livingComX,
    float livingComY,
    float livingComZ,
    float sysComX,
    float sysComY,
    float sysComZ) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int32_t tIdx = __ldg(&p.tracerIndex[i]);
    if (tIdx < 0 || static_cast<uint32_t>(tIdx) >= tracerCount) {
        return;
    }

    const float px = __ldg(&p.posX[i]);
    const float py = __ldg(&p.posY[i]);
    const float pz = __ldg(&p.posZ[i]);

    const float dx = px - livingComX;
    const float dy = py - livingComY;
    const float dz = pz - livingComZ;
    const float rPsu = sqrtf(dx * dx + dy * dy + dz * dz);
    const float rRest = __ldg(&p.tracerRestRadiusPsu[i]);
    const float dispKm = (rPsu - rRest) * 1000.0f;

    float normalDelta = 0.0f;
    if (rPsu > 1.0e-9f) {
        const float invR = 1.0f / rPsu;
        const float nx = dx * invR;
        const float ny = dy * invR;
        const float nz = dz * invR;
        const float dotVal = fminf(1.0f, fmaxf(-1.0f,
            nx * __ldg(&p.tracerRestNormalX[i]) +
            ny * __ldg(&p.tracerRestNormalY[i]) +
            nz * __ldg(&p.tracerRestNormalZ[i])));
        normalDelta = acosf(dotVal);
    }

    const float dam = fminf(1.0f, fmaxf(0.0f, finiteOrZero(__ldg(&p.damage[i]))));
    const float tempK = fmaxf(3.0f, finiteOrZero(__ldg(&p.tempK[i])));
    const uint8_t phase = __ldg(&p.phaseId[i]);
    const bool isOcean = (__ldg(&p.tracerIsOcean[i]) != 0u);
    const bool isDisrupted = (dam >= 0.85f)
                          || (phase == static_cast<uint8_t>(PhaseId::Vapor))
                          || (phase == static_cast<uint8_t>(PhaseId::Plasma))
                          || (fabsf(rPsu - rRest) > 0.5f * fmaxf(rRest, 1.0e-3f));

    const float dvShockMs = fmaxf(0.0f, -finiteOrZero(__ldg(&p.divV[i])))
                          * fmaxf(__ldg(&p.h[i]), 0.0f) * 1000.0f;
    const float overpressureKpa = fmaxf(0.0f, 1.225f * 340.0f * dvShockMs * 1.0e-3f);
    const float tsunamiHeightM = isOcean ? fmaxf(0.0f, finiteOrZero(dispKm) * 1000.0f) : 0.0f;

    CrustTracerRecordPacked rec;
    rec.anchorIndex = static_cast<uint16_t>(tIdx);
    rec.flags = static_cast<uint16_t>((isOcean ? 1u : 0u) | (isDisrupted ? 2u : 0u));
    rec.xKm = finiteOrZero((px - sysComX) * 1000.0f);
    rec.yKm = finiteOrZero((py - sysComY) * 1000.0f);
    rec.zKm = finiteOrZero((pz - sysComZ) * 1000.0f);
    rec.dispKm = finiteOrZero(dispKm);
    rec.normalDelta = finiteOrZero(normalDelta);
    rec.tempK = tempK;
    rec.damage = dam;
    rec.overpressureKpa = finiteOrZero(overpressureKpa);
    rec.tsunamiHeightM = finiteOrZero(tsunamiHeightM);

    uint8_t* dst = d_tracerRecords40B + static_cast<size_t>(tIdx) * sizeof(CrustTracerRecordPacked);
    if ((reinterpret_cast<uintptr_t>(dst) & 3u) == 0u) {
        *reinterpret_cast<CrustTracerRecordPacked*>(dst) = rec;
    } else {
        const uint8_t* srcBytes = reinterpret_cast<const uint8_t*>(&rec);
        #pragma unroll
        for (int b = 0; b < 40; ++b) {
            dst[b] = srcBytes[b];
        }
    }
}

__global__ void packRenderRecordsKernel(
    ParticleBuffersDevice p,
    uint32_t n,
    uint8_t* __restrict__ d_particleRecords20B,
    uint8_t* __restrict__ d_speedBytes,
    float sysComX,
    float sysComY,
    float sysComZ) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    constexpr float kInvLog20000 = 1.0f / 9.9034875525361280f;
    constexpr float kInvLog1e7   = 1.0f / 16.118095650958320f;
    constexpr float kInvLog2000  = 1.0f / 7.6009024595420823f;

    const float xKm = finiteOrZero((__ldg(&p.posX[i]) - sysComX) * 1000.0f);
    const float yKm = finiteOrZero((__ldg(&p.posY[i]) - sysComY) * 1000.0f);
    const float zKm = finiteOrZero((__ldg(&p.posZ[i]) - sysComZ) * 1000.0f);

    const float T = fmaxf(10.0f, finiteOrZero(__ldg(&p.tempK[i])));
    const float tNorm = fminf(1.0f, fmaxf(0.0f, logf(T * 0.1f) * kInvLog20000));
    const uint16_t tempLogU16 = static_cast<uint16_t>(lrintf(65535.0f * tNorm));

    const float pGPa = finiteOrZero(__ldg(&p.pressure[i])) * 1000.0f;
    uint16_t pressureLogU16 = 0u;
    if (pGPa > 1.0e-3f) {
        const float pNorm = fminf(1.0f, fmaxf(0.0f, logf(pGPa * 1000.0f) * kInvLog1e7));
        pressureLogU16 = static_cast<uint16_t>(lrintf(1.0f + 65534.0f * pNorm));
    }

    const uint8_t matId = __ldg(&p.materialId[i]);
    const uint8_t phaseAndBody = static_cast<uint8_t>(
        ((__ldg(&p.bodyId[i]) & 0x0Fu) << 4) | (__ldg(&p.phaseId[i]) & 0x0Fu));
    const float dam = fminf(1.0f, fmaxf(0.0f, finiteOrZero(__ldg(&p.damage[i]))));
    const uint8_t damageU8 = static_cast<uint8_t>(lrintf(dam * 255.0f));

    const float volPsu = fmaxf(__ldg(&p.mass[i]), 0.0f) / fmaxf(__ldg(&p.rho[i]), 1.0e-9f);
    const float rKm = cbrtf((3.0f * volPsu) / (4.0f * kPiF)) * 1000.0f;
    const float rNorm = fminf(1.0f, fmaxf(0.0f, logf(fmaxf(rKm, 1.0f)) * kInvLog2000));
    const uint8_t radiusLogU8 = static_cast<uint8_t>(lrintf(255.0f * rNorm));

    ParticleRecordPacked rec;
    rec.xKm = xKm;
    rec.yKm = yKm;
    rec.zKm = zKm;
    rec.tempLogU16 = tempLogU16;
    rec.pressureLogU16 = pressureLogU16;
    rec.materialId = matId;
    rec.phaseAndBody = phaseAndBody;
    rec.damageU8 = damageU8;
    rec.radiusLogU8 = radiusLogU8;

    uint8_t* dst = d_particleRecords20B + static_cast<size_t>(i) * sizeof(ParticleRecordPacked);
    if ((reinterpret_cast<uintptr_t>(dst) & 3u) == 0u) {
        *reinterpret_cast<ParticleRecordPacked*>(dst) = rec;
    } else {
        const uint8_t* srcBytes = reinterpret_cast<const uint8_t*>(&rec);
        #pragma unroll
        for (int b = 0; b < 20; ++b) {
            dst[b] = srcBytes[b];
        }
    }

    if (d_speedBytes != nullptr) {
        const float vx = __ldg(&p.velX[i]);
        const float vy = __ldg(&p.velY[i]);
        const float vz = __ldg(&p.velZ[i]);
        const float speedKmS = finiteOrZero(sqrtf(vx * vx + vy * vy + vz * vz));
        const float vNorm = fminf(1.0f, fmaxf(0.0f, logf(fmaxf(speedKmS, 0.05f) * 20.0f) * kInvLog2000));
        d_speedBytes[i] = static_cast<uint8_t>(lrintf(255.0f * vNorm));
    }
}

void swapParticleBufferPointers(ParticleBuffersDevice& a, ParticleBuffersDevice& b) {
    std::swap(a.posX, b.posX); std::swap(a.posY, b.posY); std::swap(a.posZ, b.posZ);
    std::swap(a.velX, b.velX); std::swap(a.velY, b.velY); std::swap(a.velZ, b.velZ);
    std::swap(a.velHalfX, b.velHalfX); std::swap(a.velHalfY, b.velHalfY); std::swap(a.velHalfZ, b.velHalfZ);
    std::swap(a.accX, b.accX); std::swap(a.accY, b.accY); std::swap(a.accZ, b.accZ);
    std::swap(a.mass, b.mass);
    std::swap(a.h, b.h);
    std::swap(a.h0, b.h0);
    std::swap(a.rho, b.rho);
    std::swap(a.drhoDt, b.drhoDt);
    std::swap(a.pressure, b.pressure);
    std::swap(a.soundSpeed, b.soundSpeed);
    std::swap(a.u, b.u);
    std::swap(a.duDt, b.duDt);
    std::swap(a.tempK, b.tempK);
    std::swap(a.radPowerW, b.radPowerW);
    std::swap(a.sXX, b.sXX); std::swap(a.sYY, b.sYY); std::swap(a.sZZ, b.sZZ);
    std::swap(a.sXY, b.sXY); std::swap(a.sXZ, b.sXZ); std::swap(a.sYZ, b.sYZ);
    std::swap(a.dSXX, b.dSXX); std::swap(a.dSYY, b.dSYY); std::swap(a.dSZZ, b.dSZZ);
    std::swap(a.dSXY, b.dSXY); std::swap(a.dSXZ, b.dSXZ); std::swap(a.dSYZ, b.dSYZ);
    std::swap(a.damage, b.damage);
    std::swap(a.damageCbrt, b.damageCbrt);
    std::swap(a.dDamageCbrtDt, b.dDamageCbrtDt);
    std::swap(a.epsFail, b.epsFail);
    std::swap(a.divV, b.divV);
    std::swap(a.curlVX, b.curlVX); std::swap(a.curlVY, b.curlVY); std::swap(a.curlVZ, b.curlVZ);
    std::swap(a.shepardC, b.shepardC);
    std::swap(a.surfFactor, b.surfFactor);
    std::swap(a.phiGrav, b.phiGrav);
    std::swap(a.materialId, b.materialId);
    std::swap(a.phaseId, b.phaseId);
    std::swap(a.bodyId, b.bodyId);
    std::swap(a.originalId, b.originalId);
    std::swap(a.tracerIndex, b.tracerIndex);
    std::swap(a.tracerRestRadiusPsu, b.tracerRestRadiusPsu);
    std::swap(a.tracerRestNormalX, b.tracerRestNormalX);
    std::swap(a.tracerRestNormalY, b.tracerRestNormalY);
    std::swap(a.tracerRestNormalZ, b.tracerRestNormalZ);
    std::swap(a.tracerIsOcean, b.tracerIsOcean);
}

inline int gridForCount(uint32_t n) {
    return static_cast<int>((n + kBlockSize - 1u) / kBlockSize);
}

inline int warpGridForCount(uint32_t n) {
    return static_cast<int>((n + kWarpBlockSize - 1u) / kWarpBlockSize);
}

} // namespace

// ---------- Allocation & Lifecycle Helpers ----------

void RadixBvhDevice::allocate(uint32_t maxCapacity) {
    allocateRadixBvh(*this, maxCapacity);
}

void RadixBvhDevice::free() {
    freeRadixBvh(*this);
}

void allocateRadixBvh(RadixBvhDevice& bvh, uint32_t capacity) {
    freeRadixBvh(bvh);
    if (capacity == 0u) {
        return;
    }

    const size_t numNodes = 2u * static_cast<size_t>(capacity) - 1u;
    const size_t numInternal = (capacity > 1u) ? static_cast<size_t>(capacity - 1u) : 1u;

    allocArray(bvh.mortonCodes, capacity);
    allocArray(bvh.mortonCodesSorted, capacity);
    allocArray(bvh.indicesIn, capacity);
    allocArray(bvh.indicesSorted, capacity);

    allocArray(bvh.leftChild, numInternal);
    allocArray(bvh.rightChild, numInternal);
    allocArray(bvh.parent, numNodes);
    allocArray(bvh.atomicVisited, numInternal);

    allocArray(bvh.aabbMinX, numNodes);
    allocArray(bvh.aabbMinY, numNodes);
    allocArray(bvh.aabbMinZ, numNodes);
    allocArray(bvh.aabbMaxX, numNodes);
    allocArray(bvh.aabbMaxY, numNodes);
    allocArray(bvh.aabbMaxZ, numNodes);
    allocArray(bvh.nodeMass, numNodes);
    allocArray(bvh.comX, numNodes);
    allocArray(bvh.comY, numNodes);
    allocArray(bvh.comZ, numNodes);
    allocArray(bvh.nodeMaxH, numNodes);

    allocArray(bvh.sceneBoundsOrdered, 6);
    allocArray(bvh.sceneBoundsFloat, 6);
    allocArray(bvh.livingFlags, 2);
    allocArray(bvh.nodeComMass, numNodes);
    allocArray(bvh.nodeMinExtent, numNodes);
    allocArray(bvh.nodeMaxHPacked, numNodes);

    size_t tempBytes = 0;
    ASTRAEA_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, tempBytes,
        bvh.mortonCodes, bvh.mortonCodesSorted,
        bvh.indicesIn, bvh.indicesSorted,
        static_cast<int>(capacity), 0, 63));
    if (tempBytes > 0) {
        ASTRAEA_CUDA_CHECK(cudaMalloc(&bvh.cubTempStorage, tempBytes));
    }
    bvh.cubTempBytes = tempBytes;
    bvh.capacity = capacity;
}

void freeRadixBvh(RadixBvhDevice& bvh) {
    freeArray(bvh.mortonCodes);
    freeArray(bvh.mortonCodesSorted);
    freeArray(bvh.indicesIn);
    freeArray(bvh.indicesSorted);

    freeArray(bvh.leftChild);
    freeArray(bvh.rightChild);
    freeArray(bvh.parent);
    freeArray(bvh.atomicVisited);

    freeArray(bvh.aabbMinX);
    freeArray(bvh.aabbMinY);
    freeArray(bvh.aabbMinZ);
    freeArray(bvh.aabbMaxX);
    freeArray(bvh.aabbMaxY);
    freeArray(bvh.aabbMaxZ);
    freeArray(bvh.nodeMass);
    freeArray(bvh.comX);
    freeArray(bvh.comY);
    freeArray(bvh.comZ);
    freeArray(bvh.nodeMaxH);

    freeArray(bvh.sceneBoundsOrdered);
    freeArray(bvh.sceneBoundsFloat);
    freeArray(bvh.livingFlags);
    freeArray(bvh.nodeComMass);
    freeArray(bvh.nodeMinExtent);
    freeArray(bvh.nodeMaxHPacked);

    if (bvh.cubTempStorage != nullptr) {
        cudaFree(bvh.cubTempStorage);
        bvh.cubTempStorage = nullptr;
    }
    bvh.cubTempBytes = 0;
    bvh.capacity = 0;
}

void allocateParticleBuffers(ParticleBuffersDevice& buffers, uint32_t capacity) {
    freeParticleBuffers(buffers);
    if (capacity == 0u) {
        return;
    }
    const size_t cap = capacity;
    allocArray(buffers.posX, cap);     allocArray(buffers.posY, cap);     allocArray(buffers.posZ, cap);
    allocArray(buffers.velX, cap);     allocArray(buffers.velY, cap);     allocArray(buffers.velZ, cap);
    allocArray(buffers.velHalfX, cap); allocArray(buffers.velHalfY, cap); allocArray(buffers.velHalfZ, cap);
    allocArray(buffers.accX, cap);     allocArray(buffers.accY, cap);     allocArray(buffers.accZ, cap);
    allocArray(buffers.mass, cap);
    allocArray(buffers.h, cap);
    allocArray(buffers.h0, cap);
    allocArray(buffers.rho, cap);
    allocArray(buffers.drhoDt, cap);
    allocArray(buffers.pressure, cap);
    allocArray(buffers.soundSpeed, cap);
    allocArray(buffers.u, cap);
    allocArray(buffers.duDt, cap);
    allocArray(buffers.tempK, cap);
    allocArray(buffers.radPowerW, cap);
    allocArray(buffers.sXX, cap);  allocArray(buffers.sYY, cap);  allocArray(buffers.sZZ, cap);
    allocArray(buffers.sXY, cap);  allocArray(buffers.sXZ, cap);  allocArray(buffers.sYZ, cap);
    allocArray(buffers.dSXX, cap); allocArray(buffers.dSYY, cap); allocArray(buffers.dSZZ, cap);
    allocArray(buffers.dSXY, cap); allocArray(buffers.dSXZ, cap); allocArray(buffers.dSYZ, cap);
    allocArray(buffers.damage, cap);
    allocArray(buffers.damageCbrt, cap);
    allocArray(buffers.dDamageCbrtDt, cap);
    allocArray(buffers.epsFail, cap);
    allocArray(buffers.divV, cap);
    allocArray(buffers.curlVX, cap); allocArray(buffers.curlVY, cap); allocArray(buffers.curlVZ, cap);
    allocArray(buffers.shepardC, cap);
    allocArray(buffers.surfFactor, cap);
    allocArray(buffers.phiGrav, cap);
    allocArray(buffers.materialId, cap);
    allocArray(buffers.phaseId, cap);
    allocArray(buffers.bodyId, cap);
    allocArray(buffers.originalId, cap);
    allocArray(buffers.tracerIndex, cap);
    allocArray(buffers.tracerRestRadiusPsu, cap);
    allocArray(buffers.tracerRestNormalX, cap);
    allocArray(buffers.tracerRestNormalY, cap);
    allocArray(buffers.tracerRestNormalZ, cap);
    allocArray(buffers.tracerIsOcean, cap);
    buffers.capacity = capacity;
    buffers.count = 0;
}

void freeParticleBuffers(ParticleBuffersDevice& buffers) {
    freeArray(buffers.posX);     freeArray(buffers.posY);     freeArray(buffers.posZ);
    freeArray(buffers.velX);     freeArray(buffers.velY);     freeArray(buffers.velZ);
    freeArray(buffers.velHalfX); freeArray(buffers.velHalfY); freeArray(buffers.velHalfZ);
    freeArray(buffers.accX);     freeArray(buffers.accY);     freeArray(buffers.accZ);
    freeArray(buffers.mass);
    freeArray(buffers.h);
    freeArray(buffers.h0);
    freeArray(buffers.rho);
    freeArray(buffers.drhoDt);
    freeArray(buffers.pressure);
    freeArray(buffers.soundSpeed);
    freeArray(buffers.u);
    freeArray(buffers.duDt);
    freeArray(buffers.tempK);
    freeArray(buffers.radPowerW);
    freeArray(buffers.sXX);  freeArray(buffers.sYY);  freeArray(buffers.sZZ);
    freeArray(buffers.sXY);  freeArray(buffers.sXZ);  freeArray(buffers.sYZ);
    freeArray(buffers.dSXX); freeArray(buffers.dSYY); freeArray(buffers.dSZZ);
    freeArray(buffers.dSXY); freeArray(buffers.dSXZ); freeArray(buffers.dSYZ);
    freeArray(buffers.damage);
    freeArray(buffers.damageCbrt);
    freeArray(buffers.dDamageCbrtDt);
    freeArray(buffers.epsFail);
    freeArray(buffers.divV);
    freeArray(buffers.curlVX); freeArray(buffers.curlVY); freeArray(buffers.curlVZ);
    freeArray(buffers.shepardC);
    freeArray(buffers.surfFactor);
    freeArray(buffers.phiGrav);
    freeArray(buffers.materialId);
    freeArray(buffers.phaseId);
    freeArray(buffers.bodyId);
    freeArray(buffers.originalId);
    freeArray(buffers.tracerIndex);
    freeArray(buffers.tracerRestRadiusPsu);
    freeArray(buffers.tracerRestNormalX);
    freeArray(buffers.tracerRestNormalY);
    freeArray(buffers.tracerRestNormalZ);
    freeArray(buffers.tracerIsOcean);
    buffers.capacity = 0;
    buffers.count = 0;
}

// ---------- Public Kernel Launch Wrappers ----------

void launchClearStats(ParticleBuffersDevice& p) {
    if (p.count == 0u) {
        return;
    }
    clearStatsKernel<<<gridForCount(p.count), kBlockSize>>>(p, p.count);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

void launchSortAndBuildBvh(ParticleBuffersDevice& p, ParticleBuffersDevice& alt, RadixBvhDevice& bvh) {
    const uint32_t n = p.count;
    if (n == 0u) {
        return;
    }
    if (bvh.capacity < n || alt.capacity < n) {
        throw std::runtime_error("launchSortAndBuildBvh: device buffers capacity < particle count");
    }

    uint32_t* boundsOrdered = bvh.sceneBoundsOrdered;
    float*    boundsFloat   = bvh.sceneBoundsFloat;
    uint32_t* livingFlags   = bvh.livingFlags;
    if (boundsOrdered == nullptr || boundsFloat == nullptr || livingFlags == nullptr) {
        ensureFallbackBvhScratch(boundsOrdered, boundsFloat, livingFlags);
        bvh.livingFlags = livingFlags;
    }

    initSceneBoundsKernel<<<1, 1>>>(boundsOrdered, livingFlags);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());

    const int redBlocks = std::min(gridForCount(n), 512);
    reduceSceneBoundsKernel<<<redBlocks, kBlockSize>>>(p.posX, p.posY, p.posZ, n, boundsOrdered);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());

    finalizeSceneBoundsKernel<<<1, 8>>>(boundsOrdered, boundsFloat);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());

    const int blocks = gridForCount(n);
    computeMortonCodesKernel<<<blocks, kBlockSize>>>(
        p.posX, p.posY, p.posZ, boundsFloat, n, bvh.mortonCodes, bvh.indicesIn);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());

    size_t reqBytes = 0;
    ASTRAEA_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, reqBytes,
        bvh.mortonCodes, bvh.mortonCodesSorted,
        bvh.indicesIn, bvh.indicesSorted,
        static_cast<int>(n), 0, 63));
    if (bvh.cubTempStorage == nullptr || bvh.cubTempBytes < reqBytes) {
        if (bvh.cubTempStorage != nullptr) {
            cudaFree(bvh.cubTempStorage);
        }
        ASTRAEA_CUDA_CHECK(cudaMalloc(&bvh.cubTempStorage, reqBytes));
        bvh.cubTempBytes = reqBytes;
    }

    ASTRAEA_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        bvh.cubTempStorage, reqBytes,
        bvh.mortonCodes, bvh.mortonCodesSorted,
        bvh.indicesIn, bvh.indicesSorted,
        static_cast<int>(n), 0, 63));

    permuteParticlesKernel<<<blocks, kBlockSize>>>(p, alt, bvh.indicesSorted, n);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());

    swapParticleBufferPointers(p, alt);
    alt.count = p.count;

    initLeavesAndBuildKarrasKernel<<<blocks, kBlockSize>>>(p, bvh, n, livingFlags);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());

    if (n >= 2u) {
        bottomUpBvhMomentsKernel<<<blocks, kBlockSize>>>(bvh, n);
        ASTRAEA_CUDA_CHECK(cudaGetLastError());
    }
}

void launchSphPass1Neighbors(ParticleBuffersDevice& p, const RadixBvhDevice& bvh) {
    const uint32_t n = p.count;
    if (n == 0u) {
        return;
    }
    if (bvh.nodeMinExtent != nullptr && bvh.nodeMaxHPacked != nullptr) {
        sphPass1Kernel<true><<<warpGridForCount(n), kWarpBlockSize>>>(p, bvh, n);
    } else {
        sphPass1Kernel<false><<<warpGridForCount(n), kWarpBlockSize>>>(p, bvh, n);
    }
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

void launchEvaluateEos(ParticleBuffersDevice& p) {
    const uint32_t n = p.count;
    if (n == 0u) {
        return;
    }
    evaluateEosKernel<<<gridForCount(n), kBlockSize>>>(p, n);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

void launchGravityBarnesHut(ParticleBuffersDevice& p, const RadixBvhDevice& bvh, float theta) {
    const uint32_t n = p.count;
    if (n == 0u) {
        return;
    }
    const float theta2 = fmaxf(0.0f, theta) * fmaxf(0.0f, theta);
    if (bvh.nodeComMass != nullptr && bvh.nodeMinExtent != nullptr && bvh.nodeMaxHPacked != nullptr) {
        gravityBarnesHutKernel<true><<<warpGridForCount(n), kWarpBlockSize>>>(p, bvh, n, theta2);
    } else {
        gravityBarnesHutKernel<false><<<warpGridForCount(n), kWarpBlockSize>>>(p, bvh, n, theta2);
    }
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

void launchSphPass2Forces(ParticleBuffersDevice& p, const RadixBvhDevice& bvh) {
    const uint32_t n = p.count;
    if (n == 0u) {
        return;
    }
    const uint32_t* livingFlags = (bvh.livingFlags != nullptr) ? bvh.livingFlags : globalScratch().livingFlags;
    if (bvh.nodeMinExtent != nullptr && bvh.nodeMaxHPacked != nullptr) {
        sphPass2Kernel<true><<<warpGridForCount(n), kWarpBlockSize>>>(p, bvh, n, livingFlags);
    } else {
        sphPass2Kernel<false><<<warpGridForCount(n), kWarpBlockSize>>>(p, bvh, n, livingFlags);
    }
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

void launchLeapfrogKickHalf(ParticleBuffersDevice& p, float halfDtPsu, bool includeXsph, bool clampThermal) {
    const uint32_t n = p.count;
    if (n == 0u) {
        return;
    }
    leapfrogKickHalfKernel<<<gridForCount(n), kBlockSize>>>(p, n, halfDtPsu, includeXsph, clampThermal);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

void launchLeapfrogDrift(ParticleBuffersDevice& p, float dtPsu, const RadixBvhDevice& bvh) {
    const uint32_t n = p.count;
    if (n == 0u) {
        return;
    }
    const bool useBvhXsph = (bvh.capacity >= n && bvh.leftChild != nullptr && bvh.rightChild != nullptr && n >= 2u);
    if (bvh.nodeMinExtent != nullptr && bvh.nodeMaxHPacked != nullptr) {
        leapfrogDriftKernel<true><<<warpGridForCount(n), kWarpBlockSize>>>(p, bvh, n, dtPsu, useBvhXsph);
    } else {
        leapfrogDriftKernel<false><<<warpGridForCount(n), kWarpBlockSize>>>(p, bvh, n, dtPsu, useBvhXsph);
    }
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

float launchReduceDtPsu(const ParticleBuffersDevice& p, float prevDtPsu, float maxDtPsu) {
    const float minDtFloor = fminf(1.0e-5f, maxDtPsu);
    const float upperBound = (prevDtPsu > 0.0f) ? fminf(prevDtPsu * 1.15f, maxDtPsu) : maxDtPsu;
    const uint32_t n = p.count;
    if (n == 0u) {
        return upperBound;
    }

    uint32_t* d_dtMinBits = ensureDtMinBits();
    // 0x7f7f7f7f == ~3.396e38f (positive IEEE-754 float, larger than localMin cap 1e30f)
    ASTRAEA_CUDA_CHECK(cudaMemsetAsync(d_dtMinBits, 0x7f, sizeof(uint32_t)));

    const int redBlocks = std::min(gridForCount(n), 512);
    reduceDtMinKernel<<<redBlocks, kBlockSize>>>(p, n, d_dtMinBits);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());

    uint32_t resultBits = 0x7f7f7f7fu;
    ASTRAEA_CUDA_CHECK(cudaMemcpy(&resultBits, d_dtMinBits, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    float rawMin = 0.0f;
    std::memcpy(&rawMin, &resultBits, sizeof(float));
    if (resultBits == 0x7f7f7f7fu || !std::isfinite(rawMin) || rawMin <= 0.0f) {
        rawMin = upperBound;
    }
    return fminf(upperBound, fmaxf(minDtFloor, rawMin));
}

void launchExtractTracers(
    const ParticleBuffersDevice& p,
    uint8_t* d_tracerRecords40B,
    uint32_t tracerCount,
    uint32_t tracerGridRes,
    float livingComX,
    float livingComY,
    float livingComZ,
    float sysComX,
    float sysComY,
    float sysComZ) {
    (void)tracerGridRes;
    if (tracerCount == 0u || d_tracerRecords40B == nullptr) {
        return;
    }
    ASTRAEA_CUDA_CHECK(cudaMemsetAsync(d_tracerRecords40B, 0, static_cast<size_t>(tracerCount) * 40u));
    if (p.count == 0u) {
        return;
    }
    extractTracersKernel<<<gridForCount(p.count), kBlockSize>>>(
        p, p.count, d_tracerRecords40B, tracerCount,
        livingComX, livingComY, livingComZ,
        sysComX, sysComY, sysComZ);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

void launchPackRenderRecords(
    const ParticleBuffersDevice& p,
    uint8_t* d_particleRecords20B,
    uint8_t* d_speedBytes,
    float sysComX,
    float sysComY,
    float sysComZ) {
    if (p.count == 0u || d_particleRecords20B == nullptr) {
        return;
    }
    packRenderRecordsKernel<<<gridForCount(p.count), kBlockSize>>>(
        p, p.count, d_particleRecords20B, d_speedBytes, sysComX, sysComY, sysComZ);
    ASTRAEA_CUDA_CHECK(cudaGetLastError());
}

} // namespace astraea::cuda
