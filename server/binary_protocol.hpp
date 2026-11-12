#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.hpp"

namespace astraea::server {

constexpr uint32_t kPclsMagic = 0x50434C53u;
constexpr uint16_t kPclsVersion = 1u;
constexpr std::size_t kPclsHeaderBytes = 40u;
constexpr std::size_t kParticleRecordBytes = 20u;
constexpr std::size_t kTracerRecordBytes = 40u;

enum FrameFlags : uint16_t {
    kFlagZlib          = 1u << 0,
    kFlagHasTracers    = 1u << 1,
    kFlagHasLivingCom  = 1u << 2,
    kFlagHasSpeedArray = 1u << 3
};

struct BinaryFrameHeader {
    uint32_t magic = kPclsMagic;
    uint16_t version = kPclsVersion;
    uint16_t flags = 0;
    double   simTimeSec = 0.0;
    uint32_t frameSeq = 0;
    float    dtSec = 0.0f;
    uint32_t particleCount = 0;
    uint32_t tracerCount = 0;
    uint32_t tracerGridRes = 0;
    uint32_t payloadBytes = 0;
};

/// Encode a 40-byte little-endian PCLS binary frame header.
void encodeHeader40(const BinaryFrameHeader& hdr, uint8_t out[kPclsHeaderBytes]);

/// Decode and validate magic/version of a 40-byte little-endian PCLS binary frame header.
bool decodeHeader40(const uint8_t* data, std::size_t len, BinaryFrameHeader& outHdr);

/// Compress payload with zlib (Z_BEST_SPEED).
bool zlibCompress(const uint8_t* src, std::size_t srcLen, std::vector<uint8_t>& outCompressed);

/// Decompress zlib-compressed payload into an output buffer of expectedSize bytes.
bool zlibDecompress(
    const uint8_t* src,
    std::size_t srcLen,
    std::size_t expectedSize,
    std::vector<uint8_t>& outDecompressed);

/// Package Simulation::extractRenderFrame output into a 40B header + payload frame.
std::vector<uint8_t> buildBinaryFrame(
    const TelemetrySnapshot& snapshot,
    const std::vector<uint8_t>& particleBytes,
    const std::vector<uint8_t>& tracerBytes,
    const std::vector<uint8_t>* speedBytes,
    bool compressZlib);

} // namespace astraea::server
