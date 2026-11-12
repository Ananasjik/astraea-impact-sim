#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "types.hpp"

#ifndef ASTRAEA_CUDA_CHECK
#define ASTRAEA_CUDA_CHECK(expr)                                                               \
    do {                                                                                       \
        const cudaError_t astraeaCudaErr__ = (expr);                                           \
        if (astraeaCudaErr__ != cudaSuccess) {                                                 \
            throw std::runtime_error(                                                          \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) +    \
                " (" #expr "): " + cudaGetErrorString(astraeaCudaErr__));                      \
        }                                                                                      \
    } while (false)
#endif

namespace astraea::cuda {

struct RadixBvhDevice {
    uint32_t  capacity = 0;
    uint64_t* mortonCodes = nullptr;       // [N]
    uint64_t* mortonCodesSorted = nullptr; // [N]
    uint32_t* indicesIn = nullptr;         // [N]
    uint32_t* indicesSorted = nullptr;     // [N]

    // Internal nodes [0 .. N-2], Leaf nodes conceptually [N-1 .. 2N-2]
    int32_t*  leftChild = nullptr;         // >= 0: internal node, < 0: encoded leaf (~idx)
    int32_t*  rightChild = nullptr;
    int32_t*  parent = nullptr;            // [2N - 1]
    uint32_t* atomicVisited = nullptr;     // [N - 1]

    // Node bounding boxes & Barnes-Hut multipole moments (size 2N - 1)
    float*    aabbMinX = nullptr; float* aabbMinY = nullptr; float* aabbMinZ = nullptr;
    float*    aabbMaxX = nullptr; float* aabbMaxY = nullptr; float* aabbMaxZ = nullptr;
    float*    nodeMass = nullptr;
    float*    comX = nullptr;     float* comY = nullptr;     float* comZ = nullptr;
    float*    nodeMaxH = nullptr; // max smoothing support H_i in subtree (for SPH BVH query)

    void*     cubTempStorage = nullptr;
    size_t    cubTempBytes = 0;

    // Reusable per-step device scratch buffers (zero per-step allocations)
    uint32_t* sceneBoundsOrdered = nullptr; // [6]: minX, minY, minZ, maxX, maxY, maxZ (ordered uint)
    float*    sceneBoundsFloat = nullptr;   // [6]: decoded scene AABB bounds
    uint32_t* livingFlags = nullptr;        // [2]: [0] hasLivingTracers, [1] livingIntact
    float4*   nodeComMass = nullptr;        // [2N - 1]: {comX, comY, comZ, nodeMass}
    float4*   nodeMinExtent = nullptr;      // [2N - 1]: {aabbMinX, aabbMinY, aabbMinZ, dNode}
    float4*   nodeMaxHPacked = nullptr;     // [2N - 1]: {aabbMaxX, aabbMaxY, aabbMaxZ, nodeMaxH}

    void allocate(uint32_t maxCapacity);
    void free();
};

void allocateRadixBvh(RadixBvhDevice& bvh, uint32_t capacity);
void freeRadixBvh(RadixBvhDevice& bvh);
void allocateParticleBuffers(ParticleBuffersDevice& buffers, uint32_t capacity);
void freeParticleBuffers(ParticleBuffersDevice& buffers);
void launchClearStats(ParticleBuffersDevice& p);

void launchSortAndBuildBvh(ParticleBuffersDevice& p, ParticleBuffersDevice& alt, RadixBvhDevice& bvh);
void launchSphPass1Neighbors(ParticleBuffersDevice& p, const RadixBvhDevice& bvh);
void launchEvaluateEos(ParticleBuffersDevice& p);
void launchGravityBarnesHut(ParticleBuffersDevice& p, const RadixBvhDevice& bvh, float theta = 0.5f);
void launchSphPass2Forces(ParticleBuffersDevice& p, const RadixBvhDevice& bvh);
void launchLeapfrogKickHalf(ParticleBuffersDevice& p, float halfDtPsu, bool includeXsph, bool clampThermal);
void launchLeapfrogDrift(ParticleBuffersDevice& p, float dtPsu, const RadixBvhDevice& bvh);
float launchReduceDtPsu(const ParticleBuffersDevice& p, float prevDtPsu, float maxDtPsu);
void launchExtractTracers(
    const ParticleBuffersDevice& p,
    uint8_t* d_tracerRecords40B,
    uint32_t tracerCount,
    uint32_t tracerGridRes,
    float livingComX,
    float livingComY,
    float livingComZ);
void launchPackRenderRecords(
    const ParticleBuffersDevice& p,
    uint8_t* d_particleRecords20B,
    uint8_t* d_speedBytes,
    float sysComX,
    float sysComY,
    float sysComZ);

} // namespace astraea::cuda
