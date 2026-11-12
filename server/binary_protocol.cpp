#include "binary_protocol.hpp"

#include <cstring>
#include <limits>
#include <zlib.h>

namespace astraea::server {
namespace {

inline void writeU16LE(uint8_t* dst, uint16_t v) {
    dst[0] = static_cast<uint8_t>(v & 0xFFu);
    dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
}

inline void writeU32LE(uint8_t* dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v & 0xFFu);
    dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    dst[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    dst[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

inline void writeU64LE(uint8_t* dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFFULL);
    }
}

inline void writeF32LE(uint8_t* dst, float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    writeU32LE(dst, bits);
}

inline void writeF64LE(uint8_t* dst, double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    writeU64LE(dst, bits);
}

inline uint16_t readU16LE(const uint8_t* src) {
    return static_cast<uint16_t>(src[0]) |
           (static_cast<uint16_t>(src[1]) << 8);
}

inline uint32_t readU32LE(const uint8_t* src) {
    return static_cast<uint32_t>(src[0]) |
           (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) |
           (static_cast<uint32_t>(src[3]) << 24);
}

inline uint64_t readU64LE(const uint8_t* src) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(src[i]) << (8 * i);
    }
    return v;
}

inline float readF32LE(const uint8_t* src) {
    const uint32_t bits = readU32LE(src);
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

inline double readF64LE(const uint8_t* src) {
    const uint64_t bits = readU64LE(src);
    double v = 0.0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

} // namespace

void encodeHeader40(const BinaryFrameHeader& hdr, uint8_t out[kPclsHeaderBytes]) {
    writeU32LE(out + 0, hdr.magic);
    writeU16LE(out + 4, hdr.version);
    writeU16LE(out + 6, hdr.flags);
    writeF64LE(out + 8, hdr.simTimeSec);
    writeU32LE(out + 16, hdr.frameSeq);
    writeF32LE(out + 20, hdr.dtSec);
    writeU32LE(out + 24, hdr.particleCount);
    writeU32LE(out + 28, hdr.tracerCount);
    writeU32LE(out + 32, hdr.tracerGridRes);
    writeU32LE(out + 36, hdr.payloadBytes);
}

bool decodeHeader40(const uint8_t* data, std::size_t len, BinaryFrameHeader& outHdr) {
    if (data == nullptr || len < kPclsHeaderBytes) {
        return false;
    }
    outHdr.magic = readU32LE(data + 0);
    outHdr.version = readU16LE(data + 4);
    outHdr.flags = readU16LE(data + 6);
    outHdr.simTimeSec = readF64LE(data + 8);
    outHdr.frameSeq = readU32LE(data + 16);
    outHdr.dtSec = readF32LE(data + 20);
    outHdr.particleCount = readU32LE(data + 24);
    outHdr.tracerCount = readU32LE(data + 28);
    outHdr.tracerGridRes = readU32LE(data + 32);
    outHdr.payloadBytes = readU32LE(data + 36);
    return outHdr.magic == kPclsMagic && outHdr.version == kPclsVersion;
}

bool zlibCompress(const uint8_t* src, std::size_t srcLen, std::vector<uint8_t>& outCompressed) {
    if (srcLen > static_cast<std::size_t>(std::numeric_limits<uLong>::max())) {
        return false;
    }
    uLongf destLen = compressBound(static_cast<uLong>(srcLen));
    outCompressed.resize(static_cast<std::size_t>(destLen));
    const uint8_t emptyByte = 0;
    const Bytef* srcPtr = (src != nullptr && srcLen > 0u) ? src : &emptyByte;
    const int rc = compress2(
        outCompressed.data(),
        &destLen,
        srcPtr,
        static_cast<uLong>(srcLen),
        Z_BEST_SPEED);
    if (rc != Z_OK) {
        outCompressed.clear();
        return false;
    }
    outCompressed.resize(static_cast<std::size_t>(destLen));
    return true;
}

bool zlibDecompress(
    const uint8_t* src,
    std::size_t srcLen,
    std::size_t expectedSize,
    std::vector<uint8_t>& outDecompressed) {
    if (src == nullptr && srcLen > 0u) {
        return false;
    }
    if (expectedSize > static_cast<std::size_t>(std::numeric_limits<uLongf>::max())) {
        return false;
    }
    outDecompressed.resize(expectedSize);
    if (expectedSize == 0u && srcLen == 0u) {
        return true;
    }
    uLongf destLen = static_cast<uLongf>(expectedSize);
    uint8_t dummy = 0;
    Bytef* dstPtr = expectedSize > 0u ? outDecompressed.data() : &dummy;
    const Bytef* srcPtr = srcLen > 0u ? src : &dummy;
    const int rc = uncompress(dstPtr, &destLen, srcPtr, static_cast<uLong>(srcLen));
    if (rc != Z_OK || static_cast<std::size_t>(destLen) != expectedSize) {
        outDecompressed.clear();
        return false;
    }
    return true;
}

std::vector<uint8_t> buildBinaryFrame(
    const TelemetrySnapshot& snapshot,
    const std::vector<uint8_t>& particleBytes,
    const std::vector<uint8_t>& tracerBytes,
    const std::vector<uint8_t>* speedBytes,
    bool compressZlib) {
    const uint32_t particleCount = static_cast<uint32_t>(particleBytes.size() / kParticleRecordBytes);
    const uint32_t tracerCount = static_cast<uint32_t>(tracerBytes.size() / kTracerRecordBytes);
    const bool hasSpeed = (speedBytes != nullptr && speedBytes->size() == particleCount && particleCount > 0u);

    const std::size_t rawPayloadSize =
        particleBytes.size() + tracerBytes.size() + (hasSpeed ? speedBytes->size() : 0u);

    BinaryFrameHeader hdr;
    hdr.magic = kPclsMagic;
    hdr.version = kPclsVersion;
    hdr.flags = 0u;
    if (tracerCount > 0u) {
        hdr.flags |= kFlagHasTracers;
    }
    if (snapshot.livingBodyPresent) {
        hdr.flags |= kFlagHasLivingCom;
    }
    if (hasSpeed) {
        hdr.flags |= kFlagHasSpeedArray;
    }
    hdr.simTimeSec = snapshot.simTimeSec;
    hdr.frameSeq = static_cast<uint32_t>(snapshot.frameSeq & 0xFFFFFFFFULL);
    hdr.dtSec = snapshot.dtSec;
    hdr.particleCount = particleCount;
    hdr.tracerCount = tracerCount;
    hdr.tracerGridRes = (tracerCount > 0u) ? snapshot.tracerGridRes : 0u;
    hdr.payloadBytes = static_cast<uint32_t>(rawPayloadSize);

    std::vector<uint8_t> rawPayload;
    rawPayload.reserve(rawPayloadSize);
    rawPayload.insert(rawPayload.end(), particleBytes.begin(), particleBytes.end());
    rawPayload.insert(rawPayload.end(), tracerBytes.begin(), tracerBytes.end());
    if (hasSpeed) {
        rawPayload.insert(rawPayload.end(), speedBytes->begin(), speedBytes->end());
    }

    if (compressZlib) {
        std::vector<uint8_t> compressed;
        if (zlibCompress(rawPayload.data(), rawPayload.size(), compressed)) {
            hdr.flags |= kFlagZlib;
            std::vector<uint8_t> frame(kPclsHeaderBytes + compressed.size());
            encodeHeader40(hdr, frame.data());
            if (!compressed.empty()) {
                std::memcpy(frame.data() + kPclsHeaderBytes, compressed.data(), compressed.size());
            }
            return frame;
        }
    }

    hdr.flags &= static_cast<uint16_t>(~kFlagZlib);
    std::vector<uint8_t> frame(kPclsHeaderBytes + rawPayload.size());
    encodeHeader40(hdr, frame.data());
    if (!rawPayload.empty()) {
        std::memcpy(frame.data() + kPclsHeaderBytes, rawPayload.data(), rawPayload.size());
    }
    return frame;
}

} // namespace astraea::server
