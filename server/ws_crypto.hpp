#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace astraea::server {

/// Compute RFC 3174 SHA-1 160-bit digest over raw bytes.
std::array<uint8_t, 20> sha1Digest(const uint8_t* data, std::size_t len);

/// Compute RFC 3174 SHA-1 160-bit digest over string bytes.
std::array<uint8_t, 20> sha1Digest(const std::string& input);

/// Encode raw bytes using RFC 4648 standard Base64 alphabet with '=' padding.
std::string base64Encode(const uint8_t* data, std::size_t len);

/// Encode a byte vector using RFC 4648 standard Base64 alphabet with '=' padding.
std::string base64Encode(const std::vector<uint8_t>& data);

/// Compute RFC 6455 Sec-WebSocket-Accept value for a client Sec-WebSocket-Key.
std::string sha1Base64AcceptKey(const std::string& clientWebSocketKey);

} // namespace astraea::server
