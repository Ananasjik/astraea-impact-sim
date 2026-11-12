#include "ws_crypto.hpp"

#include <cstring>

namespace astraea::server {
namespace {

constexpr const char* kWsGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
constexpr char kBase64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

inline uint32_t rotl32(uint32_t value, unsigned int bits) {
    return (value << bits) | (value >> (32u - bits));
}

inline uint32_t loadBigEndian32(const uint8_t* ptr) {
    return (static_cast<uint32_t>(ptr[0]) << 24) |
           (static_cast<uint32_t>(ptr[1]) << 16) |
           (static_cast<uint32_t>(ptr[2]) << 8) |
           static_cast<uint32_t>(ptr[3]);
}

inline void storeBigEndian32(uint8_t* ptr, uint32_t value) {
    ptr[0] = static_cast<uint8_t>((value >> 24) & 0xFFu);
    ptr[1] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    ptr[2] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    ptr[3] = static_cast<uint8_t>(value & 0xFFu);
}

std::string trimAsciiWhitespace(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() &&
           (s[start] == ' ' || s[start] == '\t' || s[start] == '\r' || s[start] == '\n')) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start &&
           (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\r' || s[end - 1] == '\n')) {
        --end;
    }
    return s.substr(start, end - start);
}

} // namespace

std::array<uint8_t, 20> sha1Digest(const uint8_t* data, std::size_t len) {
    std::vector<uint8_t> padded;
    padded.reserve(((len + 8u) / 64u + 1u) * 64u);
    if (data != nullptr && len > 0u) {
        padded.insert(padded.end(), data, data + len);
    }

    const uint64_t bitLen = static_cast<uint64_t>(len) * 8ULL;
    padded.push_back(0x80u);
    while ((padded.size() % 64u) != 56u) {
        padded.push_back(0x00u);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        padded.push_back(static_cast<uint8_t>((bitLen >> shift) & 0xFFULL));
    }

    uint32_t h0 = 0x67452301u;
    uint32_t h1 = 0xEFCDAB89u;
    uint32_t h2 = 0x98BADCFEu;
    uint32_t h3 = 0x10325476u;
    uint32_t h4 = 0xC3D2E1F0u;

    uint32_t w[80]{};
    for (std::size_t chunk = 0; chunk < padded.size(); chunk += 64u) {
        const uint8_t* block = padded.data() + chunk;
        for (std::size_t i = 0; i < 16u; ++i) {
            w[i] = loadBigEndian32(block + i * 4u);
        }
        for (std::size_t i = 16; i < 80u; ++i) {
            w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1u);
        }

        uint32_t a = h0;
        uint32_t b = h1;
        uint32_t c = h2;
        uint32_t d = h3;
        uint32_t e = h4;

        for (std::size_t i = 0; i < 80u; ++i) {
            uint32_t f = 0;
            uint32_t k = 0;
            if (i < 20u) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999u;
            } else if (i < 40u) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1u;
            } else if (i < 60u) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDCu;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6u;
            }

            const uint32_t temp = rotl32(a, 5u) + f + e + k + w[i];
            e = d;
            d = c;
            c = rotl32(b, 30u);
            b = a;
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    std::array<uint8_t, 20> out{};
    storeBigEndian32(out.data() + 0u, h0);
    storeBigEndian32(out.data() + 4u, h1);
    storeBigEndian32(out.data() + 8u, h2);
    storeBigEndian32(out.data() + 12u, h3);
    storeBigEndian32(out.data() + 16u, h4);
    return out;
}

std::array<uint8_t, 20> sha1Digest(const std::string& input) {
    return sha1Digest(reinterpret_cast<const uint8_t*>(input.data()), input.size());
}

std::string base64Encode(const uint8_t* data, std::size_t len) {
    if (data == nullptr || len == 0u) {
        return {};
    }

    std::string out;
    out.reserve(((len + 2u) / 3u) * 4u);

    for (std::size_t i = 0; i < len; i += 3u) {
        const std::size_t remaining = len - i;
        const uint32_t b0 = data[i];
        const uint32_t b1 = (remaining > 1u) ? data[i + 1u] : 0u;
        const uint32_t b2 = (remaining > 2u) ? data[i + 2u] : 0u;
        const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out.push_back(kBase64Alphabet[(triple >> 18) & 0x3Fu]);
        out.push_back(kBase64Alphabet[(triple >> 12) & 0x3Fu]);
        out.push_back((remaining > 1u) ? kBase64Alphabet[(triple >> 6) & 0x3Fu] : '=');
        out.push_back((remaining > 2u) ? kBase64Alphabet[triple & 0x3Fu] : '=');
    }

    return out;
}

std::string base64Encode(const std::vector<uint8_t>& data) {
    return base64Encode(data.data(), data.size());
}

std::string sha1Base64AcceptKey(const std::string& clientWebSocketKey) {
    const std::string combined = trimAsciiWhitespace(clientWebSocketKey) + kWsGuid;
    const auto digest = sha1Digest(combined);
    return base64Encode(digest.data(), digest.size());
}

} // namespace astraea::server
