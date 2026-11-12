#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <zlib.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "engine/presets.hpp"
#include "engine/simulation.hpp"
#include "engine/types.hpp"
#include "server/http_ws_server.hpp"
#include "server/ws_crypto.hpp"

#ifndef ASTRAEA_SERVER_BIN
#define ASTRAEA_SERVER_BIN ""
#endif

namespace {

using astraea::CollisionScenarioConfig;
using astraea::ScenarioPresetId;
using astraea::Simulation;
using astraea::makePresetConfig;
using astraea::server::HttpWsServer;
using astraea::server::ServerConfig;
using astraea::server::sha1Base64AcceptKey;

// -----------------------------------------------------------------------------
// Little-endian binary readers
// -----------------------------------------------------------------------------

uint16_t readU16Le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t readU32Le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

float readF32Le(const uint8_t* p) {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(float));
    return v;
}

double readF64Le(const uint8_t* p) {
    double v = 0.0;
    std::memcpy(&v, p, sizeof(double));
    return v;
}

// -----------------------------------------------------------------------------
// String / Base64 helpers
// -----------------------------------------------------------------------------

std::string toLowerAscii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

std::string trimAscii(std::string s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

bool isBase64BodyChar(char c) {
    const unsigned char uc = static_cast<unsigned char>(c);
    return std::isalnum(uc) != 0 || c == '+' || c == '/';
}

// -----------------------------------------------------------------------------
// Minimal RFC 8259 JSON parser (zero external dependencies)
// -----------------------------------------------------------------------------

struct JsonValue {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolVal = false;
    double numVal = 0.0;
    std::string strVal;
    std::vector<JsonValue> arrVal;
    std::map<std::string, JsonValue> objVal;

    bool isNull() const { return kind == Kind::Null; }
    bool isBool() const { return kind == Kind::Bool; }
    bool isNumber() const { return kind == Kind::Number; }
    bool isString() const { return kind == Kind::String; }
    bool isArray() const { return kind == Kind::Array; }
    bool isObject() const { return kind == Kind::Object; }

    bool has(const std::string& key) const {
        return kind == Kind::Object && objVal.find(key) != objVal.end();
    }

    const JsonValue* find(const std::string& key) const {
        if (kind != Kind::Object) return nullptr;
        auto it = objVal.find(key);
        return (it != objVal.end()) ? &it->second : nullptr;
    }

    std::string getString(const std::string& key, const std::string& fallback = "") const {
        const JsonValue* v = find(key);
        return (v && v->isString()) ? v->strVal : fallback;
    }

    double getDouble(const std::string& key, double fallback = 0.0) const {
        const JsonValue* v = find(key);
        return (v && v->isNumber()) ? v->numVal : fallback;
    }

    uint64_t getUint64(const std::string& key, uint64_t fallback = 0) const {
        const JsonValue* v = find(key);
        if (!v || !v->isNumber() || !std::isfinite(v->numVal) || v->numVal < 0.0) {
            return fallback;
        }
        return static_cast<uint64_t>(std::llround(v->numVal));
    }

    bool getBool(const std::string& key, bool fallback = false) const {
        const JsonValue* v = find(key);
        return (v && v->isBool()) ? v->boolVal : fallback;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : src_(src) {}

    std::optional<JsonValue> parse() {
        skipWs();
        JsonValue val;
        if (!parseValue(val)) {
            return std::nullopt;
        }
        skipWs();
        if (pos_ != src_.size()) {
            return std::nullopt;
        }
        return val;
    }

private:
    const std::string& src_;
    size_t pos_ = 0;

    void skipWs() {
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool parseValue(JsonValue& out) {
        skipWs();
        if (pos_ >= src_.size()) return false;
        const char c = src_[pos_];
        if (c == '{') return parseObject(out);
        if (c == '[') return parseArray(out);
        if (c == '"') {
            out = JsonValue{};
            out.kind = JsonValue::Kind::String;
            return parseString(out.strVal);
        }
        if (c == 't') return matchLiteral("true", out, true);
        if (c == 'f') return matchLiteral("false", out, false);
        if (c == 'n') {
            if (src_.compare(pos_, 4, "null") == 0) {
                pos_ += 4;
                out = JsonValue{};
                out.kind = JsonValue::Kind::Null;
                return true;
            }
            return false;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parseNumber(out);
        }
        return false;
    }

    bool matchLiteral(const char* lit, JsonValue& out, bool bVal) {
        const size_t len = std::strlen(lit);
        if (src_.compare(pos_, len, lit) != 0) return false;
        pos_ += len;
        out = JsonValue{};
        out.kind = JsonValue::Kind::Bool;
        out.boolVal = bVal;
        return true;
    }

    bool parseString(std::string& out) {
        if (pos_ >= src_.size() || src_[pos_] != '"') return false;
        ++pos_;
        out.clear();
        while (pos_ < src_.size()) {
            const char c = src_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= src_.size()) return false;
                const char esc = src_[pos_++];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > src_.size()) return false;
                        unsigned int code = 0;
                        for (int i = 0; i < 4; ++i) {
                            const char h = src_[pos_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= static_cast<unsigned int>(h - '0');
                            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned int>(10 + h - 'a');
                            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned int>(10 + h - 'A');
                            else return false;
                        }
                        if (code <= 0x7Fu) {
                            out.push_back(static_cast<char>(code));
                        } else if (code <= 0x7FFu) {
                            out.push_back(static_cast<char>(0xC0u | (code >> 6)));
                            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                        } else {
                            out.push_back(static_cast<char>(0xE0u | (code >> 12)));
                            out.push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
                            out.push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
                        }
                        break;
                    }
                    default:
                        return false;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool parseNumber(JsonValue& out) {
        const size_t start = pos_;
        if (src_[pos_] == '-') ++pos_;
        if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) return false;
        while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) ++pos_;
        if (pos_ < src_.size() && src_[pos_] == '.') {
            ++pos_;
            if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) return false;
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) ++pos_;
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) ++pos_;
            if (pos_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[pos_]))) return false;
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) ++pos_;
        }
        const std::string token = src_.substr(start, pos_ - start);
        char* endPtr = nullptr;
        const double val = std::strtod(token.c_str(), &endPtr);
        if (endPtr != token.c_str() + token.size()) return false;
        out = JsonValue{};
        out.kind = JsonValue::Kind::Number;
        out.numVal = val;
        return true;
    }

    bool parseArray(JsonValue& out) {
        ++pos_; // '['
        out = JsonValue{};
        out.kind = JsonValue::Kind::Array;
        skipWs();
        if (pos_ < src_.size() && src_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            JsonValue elem;
            if (!parseValue(elem)) return false;
            out.arrVal.push_back(std::move(elem));
            skipWs();
            if (pos_ >= src_.size()) return false;
            if (src_[pos_] == ']') {
                ++pos_;
                return true;
            }
            if (src_[pos_] != ',') return false;
            ++pos_;
        }
    }

    bool parseObject(JsonValue& out) {
        ++pos_; // '{'
        out = JsonValue{};
        out.kind = JsonValue::Kind::Object;
        skipWs();
        if (pos_ < src_.size() && src_[pos_] == '}') {
            ++pos_;
            return true;
        }
        while (true) {
            skipWs();
            std::string key;
            if (!parseString(key)) return false;
            skipWs();
            if (pos_ >= src_.size() || src_[pos_] != ':') return false;
            ++pos_;
            JsonValue val;
            if (!parseValue(val)) return false;
            out.objVal[std::move(key)] = std::move(val);
            skipWs();
            if (pos_ >= src_.size()) return false;
            if (src_[pos_] == '}') {
                ++pos_;
                return true;
            }
            if (src_[pos_] != ',') return false;
            ++pos_;
        }
    }
};

std::optional<JsonValue> parseJson(const std::string& text) {
    return JsonParser(text).parse();
}

// -----------------------------------------------------------------------------
// RAII temp webRoot and Server lifecycle guards
// -----------------------------------------------------------------------------

class TempWebRoot {
public:
    TempWebRoot() {
        const std::string pidStr = std::to_string(static_cast<long>(::getpid()));
        rootDir_ = std::filesystem::path("/tmp") / ("astraea_e3_webroot_" + pidStr);
        secretFile_ = std::filesystem::path("/tmp") / ("astraea_e3_secret_" + pidStr + ".txt");
        cleanup();

        std::filesystem::create_directories(rootDir_ / "assets");

        indexHtml_ = "<!doctype html><html><head><title>Astraea</title></head><body>OK_E3</body></html>\n";
        appJs_ = "export const astraeaVersion = 'e3-test';\n";
        secretContent_ = "TOP_SECRET_OUTSIDE_WEBROOT_E3\n";

        writeTextFile(rootDir_ / "index.html", indexHtml_);
        writeTextFile(rootDir_ / "assets" / "app.js", appJs_);
        writeTextFile(secretFile_, secretContent_);
    }

    ~TempWebRoot() {
        cleanup();
    }

    TempWebRoot(const TempWebRoot&) = delete;
    TempWebRoot& operator=(const TempWebRoot&) = delete;

    std::string root() const { return rootDir_.string(); }
    std::string secretFilename() const { return secretFile_.filename().string(); }
    const std::string& indexHtml() const { return indexHtml_; }
    const std::string& appJs() const { return appJs_; }
    const std::string& secretContent() const { return secretContent_; }

private:
    std::filesystem::path rootDir_;
    std::filesystem::path secretFile_;
    std::string indexHtml_;
    std::string appJs_;
    std::string secretContent_;

    static void writeTextFile(const std::filesystem::path& p, const std::string& data) {
        std::ofstream ofs(p, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!ofs) {
            throw std::runtime_error("Failed to create temp file: " + p.string());
        }
        ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
    }

    void cleanup() noexcept {
        std::error_code ec;
        std::filesystem::remove_all(rootDir_, ec);
        std::filesystem::remove(secretFile_, ec);
    }
};

class ServerGuard {
public:
    explicit ServerGuard(HttpWsServer& srv) : srv_(srv) {}

    ~ServerGuard() {
        stop();
    }

    ServerGuard(const ServerGuard&) = delete;
    ServerGuard& operator=(const ServerGuard&) = delete;

    bool start() {
        if (!srv_.start()) {
            return false;
        }
        started_ = true;
        loopThread_ = std::thread([this]() { srv_.runEventLoop(); });
        return true;
    }

    void stop() {
        if (started_) {
            srv_.stop();
            started_ = false;
        }
        if (loopThread_.joinable()) {
            loopThread_.join();
        }
    }

private:
    HttpWsServer& srv_;
    bool started_ = false;
    std::thread loopThread_;
};

// -----------------------------------------------------------------------------
// POSIX TCP + HTTP/1.1 + RFC 6455 WebSocket Client
// -----------------------------------------------------------------------------

class TcpSocket {
public:
    TcpSocket() = default;
    ~TcpSocket() { close(); }

    TcpSocket(const TcpSocket&) = delete;
    TcpSocket& operator=(const TcpSocket&) = delete;

    TcpSocket(TcpSocket&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    TcpSocket& operator=(TcpSocket&& other) noexcept {
        if (this != &other) {
            close();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    bool connectLoopback(uint16_t port) {
        close();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return false;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) != 1) {
            close();
            return false;
        }
        while (::connect(fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
            if (errno == EINTR) continue;
            close();
            return false;
        }
        return true;
    }

    void close() {
        if (fd_ >= 0) {
            ::shutdown(fd_, SHUT_RDWR);
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool isOpen() const { return fd_ >= 0; }
    int fd() const { return fd_; }

    bool sendAll(const uint8_t* data, size_t len) {
        size_t sent = 0;
        while (sent < len) {
            const ssize_t n = ::send(fd_, data + sent, len - sent, MSG_NOSIGNAL);
            if (n > 0) {
                sent += static_cast<size_t>(n);
                continue;
            }
            if (n < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        return true;
    }

    bool sendTextRaw(const std::string& s) {
        return sendAll(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

    // Returns >0 bytes appended, 0 on clean EOF, -1 on timeout or error.
    ssize_t recvSome(std::vector<uint8_t>& outBuf, int timeoutMs) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            const int remMs = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            if (remMs <= 0) {
                return -1;
            }
            pollfd pfd{fd_, POLLIN, 0};
            const int pr = ::poll(&pfd, 1, remMs);
            if (pr < 0) {
                if (errno == EINTR) continue;
                return -1;
            }
            if (pr == 0) {
                return -1;
            }
            uint8_t tmp[8192];
            const ssize_t n = ::recv(fd_, tmp, sizeof(tmp), 0);
            if (n < 0) {
                if (errno == EINTR) continue;
                return -1;
            }
            if (n == 0) {
                return 0;
            }
            outBuf.insert(outBuf.end(), tmp, tmp + n);
            return n;
        }
    }

private:
    int fd_ = -1;
};

struct HttpResponse {
    int statusCode = 0;
    std::map<std::string, std::string> headers;
    std::string body;
};

bool parseHttpHeaders(const std::string& headerBlock, int& outStatus, std::map<std::string, std::string>& outHeaders) {
    outHeaders.clear();
    std::istringstream iss(headerBlock);
    std::string statusLine;
    if (!std::getline(iss, statusLine)) return false;
    if (!statusLine.empty() && statusLine.back() == '\r') statusLine.pop_back();

    // HTTP/1.1 200 OK
    const size_t sp1 = statusLine.find(' ');
    if (sp1 == std::string::npos) return false;
    size_t codeStart = sp1 + 1;
    while (codeStart < statusLine.size() && statusLine[codeStart] == ' ') ++codeStart;
    if (codeStart + 3 > statusLine.size()) return false;
    outStatus = std::atoi(statusLine.substr(codeStart, 3).c_str());

    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = toLowerAscii(trimAscii(line.substr(0, colon)));
        std::string val = trimAscii(line.substr(colon + 1));
        outHeaders[std::move(key)] = std::move(val);
    }
    return outStatus > 0;
}

std::optional<HttpResponse> httpGet(uint16_t port, const std::string& rawTarget, int timeoutMs = 5000) {
    TcpSocket sock;
    if (!sock.connectLoopback(port)) {
        return std::nullopt;
    }

    std::ostringstream req;
    req << "GET " << rawTarget << " HTTP/1.1\r\n"
        << "Host: 127.0.0.1:" << port << "\r\n"
        << "Connection: close\r\n\r\n";
    if (!sock.sendTextRaw(req.str())) {
        return std::nullopt;
    }

    std::vector<uint8_t> buf;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

    size_t headerEnd = std::string::npos;
    while (true) {
        const std::string view(reinterpret_cast<const char*>(buf.data()), buf.size());
        headerEnd = view.find("\r\n\r\n");
        if (headerEnd != std::string::npos) {
            break;
        }
        const int remMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count());
        if (remMs <= 0) return std::nullopt;
        const ssize_t n = sock.recvSome(buf, remMs);
        if (n <= 0) return std::nullopt;
    }

    HttpResponse resp;
    const std::string headerBlock(reinterpret_cast<const char*>(buf.data()), headerEnd);
    if (!parseHttpHeaders(headerBlock, resp.statusCode, resp.headers)) {
        return std::nullopt;
    }

    const size_t bodyStart = headerEnd + 4;
    const auto clIt = resp.headers.find("content-length");
    if (clIt != resp.headers.end()) {
        const size_t expectedLen = static_cast<size_t>(std::strtoull(clIt->second.c_str(), nullptr, 10));
        while (buf.size() - bodyStart < expectedLen) {
            const int remMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count());
            if (remMs <= 0) return std::nullopt;
            const ssize_t n = sock.recvSome(buf, remMs);
            if (n <= 0) return std::nullopt;
        }
        resp.body.assign(reinterpret_cast<const char*>(buf.data() + bodyStart), expectedLen);
        return resp;
    }

    // No Content-Length: read until EOF
    while (true) {
        const int remMs = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count());
        if (remMs <= 0) break;
        const ssize_t n = sock.recvSome(buf, remMs);
        if (n <= 0) break;
    }
    resp.body.assign(reinterpret_cast<const char*>(buf.data() + bodyStart), buf.size() - bodyStart);
    return resp;
}

struct WsMessage {
    uint8_t opcode = 0; // 0x1 text, 0x2 binary, 0x8 close
    std::vector<uint8_t> payload;
};

class WsClient {
public:
    bool connect(uint16_t port,
                 const std::string& path = "/ws",
                 const std::string& wsKey = "dGhlIHNhbXBsZSBub25jZQ==",
                 int timeoutMs = 5000) {
        rxBuf_.clear();
        if (!sock_.connectLoopback(port)) {
            return false;
        }

        std::ostringstream req;
        req << "GET " << path << " HTTP/1.1\r\n"
            << "Host: 127.0.0.1:" << port << "\r\n"
            << "Upgrade: websocket\r\n"
            << "Connection: Upgrade\r\n"
            << "Sec-WebSocket-Key: " << wsKey << "\r\n"
            << "Sec-WebSocket-Version: 13\r\n\r\n";
        if (!sock_.sendTextRaw(req.str())) {
            return false;
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        size_t headerEnd = std::string::npos;
        while (true) {
            const std::string view(reinterpret_cast<const char*>(rxBuf_.data()), rxBuf_.size());
            headerEnd = view.find("\r\n\r\n");
            if (headerEnd != std::string::npos) {
                break;
            }
            const int remMs = remainingMs(deadline);
            if (remMs <= 0 || sock_.recvSome(rxBuf_, remMs) <= 0) {
                return false;
            }
        }

        int status = 0;
        std::map<std::string, std::string> headers;
        const std::string headerBlock(reinterpret_cast<const char*>(rxBuf_.data()), headerEnd);
        if (!parseHttpHeaders(headerBlock, status, headers)) {
            return false;
        }

        // Preserve any pipelined WebSocket frames arriving in the same segment as 101 headers
        rxBuf_.erase(rxBuf_.begin(), rxBuf_.begin() + static_cast<std::ptrdiff_t>(headerEnd + 4));

        EXPECT_EQ(status, 101);
        if (status != 101) return false;

        auto upIt = headers.find("upgrade");
        EXPECT_TRUE(upIt != headers.end() && toLowerAscii(upIt->second) == "websocket");

        auto connIt = headers.find("connection");
        EXPECT_TRUE(connIt != headers.end() && toLowerAscii(connIt->second).find("upgrade") != std::string::npos);

        auto accIt = headers.find("sec-websocket-accept");
        const std::string expectedAccept = sha1Base64AcceptKey(wsKey);
        EXPECT_TRUE(accIt != headers.end());
        if (accIt == headers.end()) return false;
        EXPECT_EQ(accIt->second, expectedAccept);

        return accIt->second == expectedAccept;
    }

    bool sendText(const std::string& json) {
        std::vector<uint8_t> bytes(json.begin(), json.end());
        return sendFrame(0x1, bytes);
    }

    bool sendFrame(uint8_t opcode, const std::vector<uint8_t>& payload) {
        std::vector<uint8_t> frame;
        frame.reserve(payload.size() + 14);
        frame.push_back(static_cast<uint8_t>(0x80u | (opcode & 0x0Fu))); // FIN=1

        const size_t len = payload.size();
        if (len <= 125u) {
            frame.push_back(static_cast<uint8_t>(0x80u | len)); // MASK=1
        } else if (len <= 65535u) {
            frame.push_back(static_cast<uint8_t>(0x80u | 126u));
            frame.push_back(static_cast<uint8_t>((len >> 8) & 0xFFu));
            frame.push_back(static_cast<uint8_t>(len & 0xFFu));
        } else {
            frame.push_back(static_cast<uint8_t>(0x80u | 127u));
            for (int shift = 56; shift >= 0; shift -= 8) {
                frame.push_back(static_cast<uint8_t>((static_cast<uint64_t>(len) >> shift) & 0xFFu));
            }
        }

        const std::array<uint8_t, 4> maskKey{0x37u, 0xFAu, 0x21u, 0x3Du};
        frame.insert(frame.end(), maskKey.begin(), maskKey.end());
        for (size_t i = 0; i < len; ++i) {
            frame.push_back(static_cast<uint8_t>(payload[i] ^ maskKey[i & 3u]));
        }
        return sock_.sendAll(frame.data(), frame.size());
    }

    void close() {
        if (sock_.isOpen()) {
            sendFrame(0x8, {});
            sock_.close();
        }
    }

    bool recvMessage(WsMessage& outMsg, int timeoutMs = 5000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        bool assembling = false;
        uint8_t msgOpcode = 0;
        std::vector<uint8_t> assembledPayload;

        while (true) {
            if (!ensureBytes(2, deadline)) {
                return false;
            }
            const uint8_t b0 = rxBuf_[0];
            const uint8_t b1 = rxBuf_[1];
            const bool fin = (b0 & 0x80u) != 0;
            const uint8_t opcode = b0 & 0x0Fu;
            const bool masked = (b1 & 0x80u) != 0;
            uint64_t payloadLen = b1 & 0x7Fu;

            size_t headerLen = 2;
            if (payloadLen == 126u) {
                if (!ensureBytes(4, deadline)) return false;
                payloadLen = (static_cast<uint64_t>(rxBuf_[2]) << 8) | static_cast<uint64_t>(rxBuf_[3]);
                headerLen = 4;
            } else if (payloadLen == 127u) {
                if (!ensureBytes(10, deadline)) return false;
                payloadLen = 0;
                for (int i = 0; i < 8; ++i) {
                    payloadLen = (payloadLen << 8) | static_cast<uint64_t>(rxBuf_[2 + i]);
                }
                headerLen = 10;
            }

            std::array<uint8_t, 4> maskKey{};
            if (masked) {
                ADD_FAILURE() << "RFC 6455 violation: server-to-client frame must not be masked";
                if (!ensureBytes(headerLen + 4, deadline)) return false;
                for (int i = 0; i < 4; ++i) maskKey[i] = rxBuf_[headerLen + i];
                headerLen += 4;
            }

            if (payloadLen > 256ull * 1024ull * 1024ull) {
                return false;
            }

            if (!ensureBytes(headerLen + static_cast<size_t>(payloadLen), deadline)) {
                return false;
            }

            std::vector<uint8_t> framePayload(
                rxBuf_.begin() + static_cast<std::ptrdiff_t>(headerLen),
                rxBuf_.begin() + static_cast<std::ptrdiff_t>(headerLen + static_cast<size_t>(payloadLen)));
            rxBuf_.erase(rxBuf_.begin(), rxBuf_.begin() + static_cast<std::ptrdiff_t>(headerLen + static_cast<size_t>(payloadLen)));

            if (masked) {
                for (size_t i = 0; i < framePayload.size(); ++i) {
                    framePayload[i] ^= maskKey[i & 3u];
                }
            }

            // Control frames may arrive interleaved inside fragmented data frames
            if (opcode == 0x8u) {
                outMsg.opcode = 0x8u;
                outMsg.payload = std::move(framePayload);
                return true;
            }
            if (opcode == 0x9u) { // Ping -> Pong
                sendFrame(0xAu, framePayload);
                continue;
            }
            if (opcode == 0xAu) { // Pong
                continue;
            }

            if (opcode == 0x0u) {
                if (!assembling) {
                    ADD_FAILURE() << "Unexpected continuation frame without active fragmented message";
                    return false;
                }
                assembledPayload.insert(assembledPayload.end(), framePayload.begin(), framePayload.end());
            } else {
                if (assembling) {
                    ADD_FAILURE() << "Received new data frame before previous fragmented message finished";
                    return false;
                }
                msgOpcode = opcode;
                assembledPayload = std::move(framePayload);
                assembling = true;
            }

            if (fin) {
                outMsg.opcode = msgOpcode;
                outMsg.payload = std::move(assembledPayload);
                return true;
            }
        }
    }

    template <typename Pred>
    std::optional<JsonValue> recvTextMatching(Pred&& pred, int timeoutMs = 8000, bool failOnError = true) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (true) {
            const int remMs = remainingMs(deadline);
            if (remMs <= 0) return std::nullopt;
            WsMessage msg;
            if (!recvMessage(msg, remMs)) return std::nullopt;
            if (msg.opcode == 0x8u) return std::nullopt;
            if (msg.opcode != 0x1u) continue; // skip binary frames

            const std::string text(msg.payload.begin(), msg.payload.end());
            const auto parsed = parseJson(text);
            if (!parsed.has_value()) {
                ADD_FAILURE() << "Server sent invalid JSON text frame: " << text;
                return std::nullopt;
            }
            if (failOnError && parsed->getString("type") == "error") {
                if (!pred(*parsed)) {
                    ADD_FAILURE() << "Unexpected server error frame: " << text;
                    return std::nullopt;
                }
            }
            if (pred(*parsed)) {
                return parsed;
            }
        }
    }

    template <typename Pred>
    std::optional<std::vector<uint8_t>> recvBinaryMatching(Pred&& pred, int timeoutMs = 8000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (true) {
            const int remMs = remainingMs(deadline);
            if (remMs <= 0) return std::nullopt;
            WsMessage msg;
            if (!recvMessage(msg, remMs)) return std::nullopt;
            if (msg.opcode == 0x8u) return std::nullopt;
            if (msg.opcode != 0x2u) continue; // skip text frames
            if (pred(msg.payload)) {
                return msg.payload;
            }
        }
    }

private:
    TcpSocket sock_;
    std::vector<uint8_t> rxBuf_;

    static int remainingMs(std::chrono::steady_clock::time_point deadline) {
        return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count());
    }

    bool ensureBytes(size_t needed, std::chrono::steady_clock::time_point deadline) {
        while (rxBuf_.size() < needed) {
            const int remMs = remainingMs(deadline);
            if (remMs <= 0) return false;
            const ssize_t n = sock_.recvSome(rxBuf_, remMs);
            if (n <= 0) return false;
        }
        return true;
    }
};

// -----------------------------------------------------------------------------
// Validation helpers
// -----------------------------------------------------------------------------

void validateTelemetryDtoSchema(const JsonValue& t) {
    EXPECT_EQ(t.getString("type"), "telemetry");
    EXPECT_TRUE(t.find("frameSeq") && t.find("frameSeq")->isNumber());
    EXPECT_TRUE(t.find("simTimeSec") && t.find("simTimeSec")->isNumber());
    EXPECT_TRUE(t.find("dtSec") && t.find("dtSec")->isNumber());

    const std::string state = t.getString("state");
    EXPECT_TRUE(state == "paused" || state == "playing") << "state=" << state;

    EXPECT_FALSE(t.getString("preset").empty());
    const std::string outcome = t.getString("outcome");
    EXPECT_TRUE(outcome == "Approaching" || outcome == "Merging" ||
                outcome == "GrazeAndMerge" || outcome == "HitAndRun" ||
                outcome == "CatastrophicDisruption") << "outcome=" << outcome;

    const std::array<const char*, 17> scalarFields = {
        "kineticEnergyJ", "potentialEnergyJ", "internalEnergyJ", "radiatedEnergyJ",
        "totalEnergyJ", "energyDriftRel", "virialRatio", "maxTempK",
        "maxPressureGPa", "boundMassFrac", "diskMassFrac", "ejectaMassFrac", "qRatioQrd",
        "gpuStepMs", "livingBodyRadiusKm", "livingBodySpinRadS", "tracerGridRes"
    };
    for (const char* f : scalarFields) {
        const JsonValue* v = t.find(f);
        ASSERT_NE(v, nullptr) << "Missing telemetry field: " << f;
        EXPECT_TRUE(v->isNumber()) << "Field not numeric: " << f;
        EXPECT_TRUE(std::isfinite(v->numVal)) << "Field not finite: " << f;
    }

    EXPECT_TRUE(t.find("livingBodyPresent") && t.find("livingBodyPresent")->isBool());

    const std::array<const char*, 4> vec3Fields = {
        "momentumKgMS", "angularMomentumKgM2S", "livingBodyComKm", "impactorComKm"
    };
    for (const char* vf : vec3Fields) {
        const JsonValue* v = t.find(vf);
        ASSERT_NE(v, nullptr) << "Missing vec3 field: " << vf;
        ASSERT_TRUE(v->isArray()) << "Not array: " << vf;
        ASSERT_EQ(v->arrVal.size(), 3u) << "Not len 3: " << vf;
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_TRUE(v->arrVal[i].isNumber() && std::isfinite(v->arrVal[i].numVal));
        }
    }
}

void validateBinaryFramePayload(const std::vector<uint8_t>& frame,
                                uint32_t expectedN,
                                uint32_t expectedT,
                                uint32_t expectedGridRes,
                                bool expectZlib,
                                bool expectSpeedArray) {
    ASSERT_GE(frame.size(), 40u);

    const uint32_t magic = readU32Le(frame.data() + 0);
    const uint16_t version = readU16Le(frame.data() + 4);
    const uint16_t flags = readU16Le(frame.data() + 6);
    const double simTimeSec = readF64Le(frame.data() + 8);
    const uint32_t frameSeq = readU32Le(frame.data() + 16);
    const float dtSec = readF32Le(frame.data() + 20);
    const uint32_t particleCount = readU32Le(frame.data() + 24);
    const uint32_t tracerCount = readU32Le(frame.data() + 28);
    const uint32_t tracerGridRes = readU32Le(frame.data() + 32);
    const uint32_t payloadBytes = readU32Le(frame.data() + 36);

    (void)frameSeq;
    EXPECT_EQ(magic, 0x50434C53u);
    EXPECT_EQ(version, 1u);
    EXPECT_EQ((flags & 0x01u) != 0, expectZlib);
    EXPECT_EQ((flags & 0x02u) != 0, expectedT > 0u);
    EXPECT_EQ((flags & 0x08u) != 0, expectSpeedArray);
    EXPECT_TRUE(std::isfinite(simTimeSec));
    EXPECT_GE(simTimeSec, 0.0);
    EXPECT_TRUE(std::isfinite(dtSec));
    EXPECT_GE(dtSec, 0.0f);
    EXPECT_EQ(particleCount, expectedN);
    EXPECT_EQ(tracerCount, expectedT);
    EXPECT_EQ(tracerGridRes, expectedGridRes);

    const uint32_t expectedPayload = 20u * expectedN + 40u * expectedT + (expectSpeedArray ? expectedN : 0u);
    EXPECT_EQ(payloadBytes, expectedPayload);

    std::vector<uint8_t> inflated;
    const uint8_t* payloadPtr = nullptr;
    if (expectZlib) {
        ASSERT_GT(frame.size(), 40u);
        inflated.resize(expectedPayload);
        uLongf destLen = static_cast<uLongf>(expectedPayload);
        const int zret = ::uncompress(
            inflated.data(),
            &destLen,
            frame.data() + 40,
            static_cast<uLong>(frame.size() - 40));
        ASSERT_EQ(zret, Z_OK);
        ASSERT_EQ(destLen, static_cast<uLongf>(expectedPayload));
        payloadPtr = inflated.data();
    } else {
        ASSERT_EQ(frame.size(), 40u + expectedPayload);
        payloadPtr = frame.data() + 40;
    }

    for (uint32_t i = 0; i < expectedN; ++i) {
        const uint8_t* rec = payloadPtr + 20u * i;
        const float xKm = readF32Le(rec + 0);
        const float yKm = readF32Le(rec + 4);
        const float zKm = readF32Le(rec + 8);
        const uint8_t materialId = rec[16];
        const uint8_t phaseAndBody = rec[17];
        const uint8_t bodyId = phaseAndBody >> 4;
        const uint8_t phaseId = phaseAndBody & 0x0Fu;

        ASSERT_TRUE(std::isfinite(xKm));
        ASSERT_TRUE(std::isfinite(yKm));
        ASSERT_TRUE(std::isfinite(zKm));
        EXPECT_LE(materialId, 4u);
        EXPECT_LE(phaseId, 5u);
        EXPECT_LE(bodyId, 1u);
    }

    if (expectedT > 0u) {
        const uint8_t* tracerBase = payloadPtr + 20u * expectedN;
        std::vector<uint32_t> seen(expectedT, 0u);
        for (uint32_t t = 0; t < expectedT; ++t) {
            const uint8_t* rec = tracerBase + 40u * t;
            const uint16_t anchorIdx = readU16Le(rec + 0);
            const uint16_t tFlags = readU16Le(rec + 2);
            ASSERT_LT(anchorIdx, expectedT);
            seen[anchorIdx]++;
            EXPECT_LE(tFlags, 3u);
            for (size_t off = 4; off <= 36; off += 4) {
                EXPECT_TRUE(std::isfinite(readF32Le(rec + off)));
            }
        }
        for (uint32_t idx = 0; idx < expectedT; ++idx) {
            EXPECT_EQ(seen[idx], 1u) << "Missing/duplicate tracer anchor " << idx;
        }
    }
}

template <typename Pred>
bool waitForCondition(Pred&& pred, int timeoutMs = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

int runProcessCapture(const std::string& exe, const std::vector<std::string>& args, std::string& outOutput) {
    int pipefd[2]{-1, -1};
    if (::pipe(pipefd) != 0) return -1;

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        ::close(pipefd[0]);
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);

        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(exe.c_str()));
        for (const auto& a : args) {
            argv.push_back(const_cast<char*>(a.c_str()));
        }
        argv.push_back(nullptr);
        ::execv(exe.c_str(), argv.data());
        ::_exit(127);
    }

    ::close(pipefd[1]);
    outOutput.clear();
    char buf[4096];
    while (true) {
        const ssize_t n = ::read(pipefd[0], buf, sizeof(buf));
        if (n > 0) {
            outOutput.append(buf, static_cast<size_t>(n));
        } else if (n == 0) {
            break;
        } else if (errno != EINTR) {
            break;
        }
    }
    ::close(pipefd[0]);

    int status = 0;
    pid_t w = -1;
    do {
        w = ::waitpid(pid, &status, 0);
    } while (w < 0 && errno == EINTR);
    if (w < 0) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::string resolveSimServerBin() {
    const std::string configured = ASTRAEA_SERVER_BIN;
    if (!configured.empty()) {
        return configured;
    }
    const std::array<const char*, 4> fallbacks = {
        "./sim_server",
        "./server/sim_server",
        "../server/sim_server",
        "../../server/sim_server"
    };
    for (const char* c : fallbacks) {
        if (::access(c, X_OK) == 0) {
            return c;
        }
    }
    return {};
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

TEST(ServerProtocolTest, Rfc6455AcceptKeyExactness_AC11) {
    EXPECT_EQ(sha1Base64AcceptKey("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");

    const std::string second = sha1Base64AcceptKey("x3JJHMbDL1EzLkh9GBhXDw==");
    EXPECT_EQ(second, "HSmrc0sMlYUkAGmm5OPpG2HaGWk=");
    EXPECT_EQ(second.size(), 28u);
    EXPECT_EQ(second.back(), '=');
    EXPECT_NE(second, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    for (size_t i = 0; i < 27; ++i) {
        EXPECT_TRUE(isBase64BodyChar(second[i])) << "Invalid base64 char at " << i;
    }
}

TEST(ServerProtocolTest, HttpStaticServingAndPathTraversalGuards_AC11) {
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    TempWebRoot webRoot;

    Simulation sim(8192);
    sim.loadScenario(makePresetConfig(ScenarioPresetId::MercuryMantleStrip, 4096));

    // Invalid bind address must fail start() cleanly
    {
        ServerConfig badCfg;
        badCfg.bindHost = "999.999.999.999";
        badCfg.port = 0;
        badCfg.webRoot = webRoot.root();
        HttpWsServer badServer(sim, badCfg);
        EXPECT_FALSE(badServer.start());
    }

    ServerConfig cfg;
    cfg.bindHost = "127.0.0.1";
    cfg.port = 0;
    cfg.webRoot = webRoot.root();
    cfg.targetFps = 30;
    cfg.compressBinaryZlib = false;

    HttpWsServer server(sim, cfg);
    ServerGuard guard(server);
    ASSERT_TRUE(guard.start());
    const uint16_t port = server.boundPort();
    ASSERT_NE(port, 0u);

    // 1. GET / -> 200 OK, text/html, index.html body
    {
        const auto resp = httpGet(port, "/");
        ASSERT_TRUE(resp.has_value());
        EXPECT_EQ(resp->statusCode, 200);
        const auto ctIt = resp->headers.find("content-type");
        ASSERT_NE(ctIt, resp->headers.end());
        EXPECT_NE(toLowerAscii(ctIt->second).find("text/html"), std::string::npos);
        EXPECT_EQ(resp->body, webRoot.indexHtml());
    }

    // 2. GET /assets/app.js -> 200 OK, JS MIME type, exact body
    {
        const auto resp = httpGet(port, "/assets/app.js");
        ASSERT_TRUE(resp.has_value());
        EXPECT_EQ(resp->statusCode, 200);
        const auto ctIt = resp->headers.find("content-type");
        ASSERT_NE(ctIt, resp->headers.end());
        EXPECT_NE(toLowerAscii(ctIt->second).find("javascript"), std::string::npos);
        EXPECT_EQ(resp->body, webRoot.appJs());
    }

    // 3. GET /missing_file.html -> 404
    {
        const auto resp = httpGet(port, "/missing_file.html");
        ASSERT_TRUE(resp.has_value());
        EXPECT_EQ(resp->statusCode, 404);
    }

    // 4. GET /../etc/passwd -> 403 Forbidden
    {
        const auto resp = httpGet(port, "/../etc/passwd");
        ASSERT_TRUE(resp.has_value());
        EXPECT_EQ(resp->statusCode, 403);
    }

    // 5. GET /%2e%2e/astraea_e3_secret_<pid>.txt -> 403 Forbidden
    {
        const auto resp = httpGet(port, "/%2e%2e/" + webRoot.secretFilename());
        ASSERT_TRUE(resp.has_value());
        EXPECT_EQ(resp->statusCode, 403);
        EXPECT_EQ(resp->body.find("TOP_SECRET_OUTSIDE_WEBROOT_E3"), std::string::npos);
    }
}

TEST(ServerProtocolTest, BinaryFrameStreamAndVelocityMode_AC12) {
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    TempWebRoot webRoot;

    Simulation sim(8192);
    sim.loadScenario(makePresetConfig(ScenarioPresetId::LivingGaiaAsteroidCataclysm, 4096));

    ServerConfig cfg;
    cfg.bindHost = "127.0.0.1";
    cfg.port = 0;
    cfg.webRoot = webRoot.root();
    cfg.targetFps = 30;
    cfg.compressBinaryZlib = false;

    HttpWsServer server(sim, cfg);
    ServerGuard guard(server);
    ASSERT_TRUE(guard.start());
    const uint16_t port = server.boundPort();
    ASSERT_NE(port, 0u);

    EXPECT_EQ(server.connectedClientCount(), 0u);

    WsClient ws;
    ASSERT_TRUE(ws.connect(port, "/ws", "dGhlIHNhbXBsZSBub25jZQ=="));
    EXPECT_TRUE(waitForCondition([&]() { return server.connectedClientCount() == 1u; }));
    EXPECT_EQ(server.connectedClientCount(), 1u);

    ASSERT_TRUE(ws.sendText(R"({"type":"step","count":1})"));
    const auto binFrame = ws.recvBinaryMatching([](const std::vector<uint8_t>& frame) {
        return frame.size() >= 40u && readU32Le(frame.data() + 16) >= 1u;
    });
    ASSERT_TRUE(binFrame.has_value()) << "Timed out waiting for uncompressed binary frame";
    validateBinaryFramePayload(*binFrame, 4096u, 1536u, 16u, /*expectZlib=*/false, /*expectSpeedArray=*/false);

    // Enable Velocity visualization mode -> sets flags & 0x08 and appends N speed bytes
    ASSERT_TRUE(ws.sendText(R"({"type":"set_vis_mode","mode":"Velocity"})"));
    ASSERT_TRUE(ws.sendText(R"({"type":"step","count":1})"));

    const auto velFrame = ws.recvBinaryMatching([](const std::vector<uint8_t>& frame) {
        return frame.size() >= 40u && (readU16Le(frame.data() + 6) & 0x08u) != 0u;
    });
    ASSERT_TRUE(velFrame.has_value()) << "Timed out waiting for Velocity mode binary frame with flags & 0x08";
    validateBinaryFramePayload(*velFrame, 4096u, 1536u, 16u, /*expectZlib=*/false, /*expectSpeedArray=*/true);

    ws.close();
    EXPECT_TRUE(waitForCondition([&]() { return server.connectedClientCount() == 0u; }));
    EXPECT_EQ(server.connectedClientCount(), 0u);
}

TEST(ServerProtocolTest, BinaryFrameZlibCompression_AC12) {
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    TempWebRoot webRoot;

    Simulation sim(8192);
    sim.loadScenario(makePresetConfig(ScenarioPresetId::LivingGaiaAsteroidCataclysm, 4096));

    ServerConfig cfg;
    cfg.bindHost = "127.0.0.1";
    cfg.port = 0;
    cfg.webRoot = webRoot.root();
    cfg.targetFps = 30;
    cfg.compressBinaryZlib = true;

    HttpWsServer server(sim, cfg);
    ServerGuard guard(server);
    ASSERT_TRUE(guard.start());
    const uint16_t port = server.boundPort();
    ASSERT_NE(port, 0u);

    WsClient ws;
    ASSERT_TRUE(ws.connect(port, "/ws", "x3JJHMbDL1EzLkh9GBhXDw=="));

    ASSERT_TRUE(ws.sendText(R"({"type":"step","count":1})"));
    const auto zFrame = ws.recvBinaryMatching([](const std::vector<uint8_t>& frame) {
        return frame.size() >= 40u && readU32Le(frame.data() + 16) >= 1u;
    });
    ASSERT_TRUE(zFrame.has_value()) << "Timed out waiting for zlib-compressed binary frame";
    validateBinaryFramePayload(*zFrame, 4096u, 1536u, 16u, /*expectZlib=*/true, /*expectSpeedArray=*/false);

    ws.close();
    EXPECT_TRUE(waitForCondition([&]() { return server.connectedClientCount() == 0u; }));
}

TEST(ServerProtocolTest, WebSocketJsonControlRpcRoundtrip_AC13) {
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    TempWebRoot webRoot;

    Simulation sim(16384);
    sim.loadScenario(makePresetConfig(ScenarioPresetId::LivingGaiaAsteroidCataclysm, 4096));

    ServerConfig cfg;
    cfg.bindHost = "127.0.0.1";
    cfg.port = 0;
    cfg.webRoot = webRoot.root();
    cfg.targetFps = 30;
    cfg.compressBinaryZlib = false;

    HttpWsServer server(sim, cfg);
    ServerGuard guard(server);
    ASSERT_TRUE(guard.start());
    const uint16_t port = server.boundPort();
    ASSERT_NE(port, 0u);
    EXPECT_EQ(server.connectedClientCount(), 0u);

    WsClient ws;
    ASSERT_TRUE(ws.connect(port, "/ws", "dGhlIHNhbXBsZSBub25jZQ=="));
    EXPECT_TRUE(waitForCondition([&]() { return server.connectedClientCount() == 1u; }));
    EXPECT_EQ(server.connectedClientCount(), 1u);

    // 1. ping -> pong
    ASSERT_TRUE(ws.sendText(R"({"type":"ping","ts":1700000123})"));
    const auto pong1 = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "pong" && v.getUint64("ts") == 1700000123ull;
    });
    ASSERT_TRUE(pong1.has_value());

    // 2a. set_preset (MercuryMantleStrip -> non-living body, tracerCount == 0)
    ASSERT_TRUE(ws.sendText(R"({"type":"set_preset","preset":"MercuryMantleStrip","particles":4096})"));
    const auto loadedMercury = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "preset_loaded" && v.getString("preset") == "MercuryMantleStrip";
    });
    ASSERT_TRUE(loadedMercury.has_value());
    EXPECT_EQ(loadedMercury->getUint64("particleCount"), 4096ull);
    EXPECT_EQ(loadedMercury->getUint64("tracerCount"), 0ull);
    EXPECT_FALSE(loadedMercury->getBool("livingBodyPresent", true));

    auto telMercury = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "telemetry" && v.getString("preset") == "MercuryMantleStrip";
    }, 1000);
    if (!telMercury.has_value()) {
        ASSERT_TRUE(ws.sendText(R"({"type":"step","count":1})"));
        telMercury = ws.recvTextMatching([](const JsonValue& v) {
            return v.getString("type") == "telemetry" && v.getString("preset") == "MercuryMantleStrip";
        });
    }
    ASSERT_TRUE(telMercury.has_value());
    validateTelemetryDtoSchema(*telMercury);
    EXPECT_FALSE(telMercury->getBool("livingBodyPresent", true));

    // 2b. set_preset (EarthTheiaMoonForming -> LivingTerra target)
    ASSERT_TRUE(ws.sendText(R"({"type":"set_preset","preset":"EarthTheiaMoonForming","particles":4096})"));
    const auto loadedEarth = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "preset_loaded" && v.getString("preset") == "EarthTheiaMoonForming";
    });
    ASSERT_TRUE(loadedEarth.has_value());
    EXPECT_EQ(loadedEarth->getUint64("particleCount"), 4096ull);
    EXPECT_TRUE(loadedEarth->getBool("livingBodyPresent", false));

    auto telEarth = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "telemetry" && v.getString("preset") == "EarthTheiaMoonForming";
    }, 1000);
    if (!telEarth.has_value()) {
        ASSERT_TRUE(ws.sendText(R"({"type":"step","count":1})"));
        telEarth = ws.recvTextMatching([](const JsonValue& v) {
            return v.getString("type") == "telemetry" && v.getString("preset") == "EarthTheiaMoonForming";
        });
    }
    ASSERT_TRUE(telEarth.has_value());
    EXPECT_EQ(telEarth->getString("preset"), "EarthTheiaMoonForming");

    // 3. launch_custom (LivingTerra + IronCore)
    ASSERT_TRUE(ws.sendText(
        R"({"type":"launch_custom","config":{"targetArchetype":"LivingTerra","impactorArchetype":"IronCore","targetMassKg":5.97219e24,"impactorMassKg":6.417e23,"impactVelocityRatioVesc":1.2,"impactParamB":0.3,"initialSepRadii":2.5,"targetSpinRadS":7.2921e-5,"particles":4096}})"));
    const auto loadedCustom = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "preset_loaded";
    });
    ASSERT_TRUE(loadedCustom.has_value());
    EXPECT_TRUE(loadedCustom->getBool("livingBodyPresent", false));
    EXPECT_EQ(loadedCustom->getUint64("particleCount"), 4096ull);
    EXPECT_EQ(loadedCustom->getUint64("tracerCount"), 1536ull);
    EXPECT_EQ(loadedCustom->getUint64("tracerGridRes"), 16ull);

    // 4. pause + step(count=3)
    ASSERT_TRUE(ws.sendText(R"({"type":"pause"})"));
    ASSERT_TRUE(ws.sendText(R"({"type":"step","count":3})"));
    const auto telStepped = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "telemetry" &&
               v.getUint64("frameSeq") >= 3ull &&
               v.getDouble("simTimeSec") > 0.0;
    });
    ASSERT_TRUE(telStepped.has_value());
    validateTelemetryDtoSchema(*telStepped);
    EXPECT_EQ(telStepped->getString("state"), "paused");
    const uint64_t seqAfterStep = telStepped->getUint64("frameSeq");
    EXPECT_GE(seqAfterStep, 3ull);

    // 5. set_timescale + play -> continuous frameSeq advance -> pause
    ASSERT_TRUE(ws.sendText(R"({"type":"set_timescale","scale":2.5})"));
    ASSERT_TRUE(ws.sendText(R"({"type":"play"})"));

    const auto telPlay0 = ws.recvTextMatching([&](const JsonValue& v) {
        return v.getString("type") == "telemetry" &&
               v.getString("state") == "playing" &&
               v.getUint64("frameSeq") > seqAfterStep;
    });
    ASSERT_TRUE(telPlay0.has_value());
    const uint64_t playSeq0 = telPlay0->getUint64("frameSeq");

    const auto telPlay1 = ws.recvTextMatching([&](const JsonValue& v) {
        return v.getString("type") == "telemetry" &&
               v.getString("state") == "playing" &&
               v.getUint64("frameSeq") >= playSeq0 + 2ull;
    });
    ASSERT_TRUE(telPlay1.has_value());

    ASSERT_TRUE(ws.sendText(R"({"type":"pause"})"));
    auto telPaused = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "telemetry" && v.getString("state") == "paused";
    }, 1000);
    if (!telPaused.has_value()) {
        ASSERT_TRUE(ws.sendText(R"({"type":"step","count":1})"));
        telPaused = ws.recvTextMatching([](const JsonValue& v) {
            return v.getString("type") == "telemetry" && v.getString("state") == "paused";
        });
    }
    ASSERT_TRUE(telPaused.has_value());
    const uint64_t pausedSeq = telPaused->getUint64("frameSeq");

    // Confirm paused state stays stable after roundtrip + brief settle
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    ASSERT_TRUE(ws.sendText(R"({"type":"ping","ts":111222})"));
    const auto pongPause = ws.recvTextMatching([&](const JsonValue& v) {
        if (v.getString("type") == "telemetry") {
            EXPECT_EQ(v.getString("state"), "paused");
            EXPECT_EQ(v.getUint64("frameSeq"), pausedSeq);
        }
        return v.getString("type") == "pong" && v.getUint64("ts") == 111222ull;
    });
    ASSERT_TRUE(pongPause.has_value());

    // 6. set_slice_plane -> accepted without error
    ASSERT_TRUE(ws.sendText(R"({"type":"set_slice_plane","enabled":true,"normal":[0,1,0],"offsetKm":100.0})"));
    ASSERT_TRUE(ws.sendText(R"({"type":"ping","ts":777888})"));
    const auto pongSlice = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "pong" && v.getUint64("ts") == 777888ull;
    });
    ASSERT_TRUE(pongSlice.has_value());

    // 7. request_export csv & json
    ASSERT_TRUE(ws.sendText(R"({"type":"request_export","format":"csv"})"));
    const auto csvExport = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "export_data" && v.getString("format") == "csv";
    });
    ASSERT_TRUE(csvExport.has_value());
    EXPECT_FALSE(csvExport->getString("filename").empty());
    const std::string csvContent = csvExport->getString("content");
    EXPECT_FALSE(csvContent.empty());
    EXPECT_NE(csvContent.find("frameSeq"), std::string::npos);
    EXPECT_NE(csvContent.find("simTimeSec"), std::string::npos);
    EXPECT_NE(csvContent.find("kineticEnergyJ"), std::string::npos);
    EXPECT_NE(csvContent.find("energyDriftRel"), std::string::npos);
    EXPECT_GE(std::count(csvContent.begin(), csvContent.end(), '\n'), 2);

    ASSERT_TRUE(ws.sendText(R"({"type":"request_export","format":"json"})"));
    const auto jsonExport = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "export_data" && v.getString("format") == "json";
    });
    ASSERT_TRUE(jsonExport.has_value());
    EXPECT_FALSE(jsonExport->getString("filename").empty());
    const std::string jsonContent = jsonExport->getString("content");
    ASSERT_FALSE(jsonContent.empty());
    const auto parsedExportJson = parseJson(jsonContent);
    ASSERT_TRUE(parsedExportJson.has_value()) << "Exported JSON content failed to parse";
    EXPECT_TRUE(parsedExportJson->isObject());
    EXPECT_TRUE(parsedExportJson->has("kineticEnergyJ"));
    EXPECT_TRUE(parsedExportJson->has("potentialEnergyJ"));
    EXPECT_TRUE(parsedExportJson->has("internalEnergyJ"));
    EXPECT_TRUE(parsedExportJson->has("radiatedEnergyJ"));
    EXPECT_TRUE(parsedExportJson->has("totalEnergyJ"));
    EXPECT_TRUE(parsedExportJson->has("energyDriftRel"));
    EXPECT_TRUE(parsedExportJson->has("frameSeq"));
    EXPECT_TRUE(parsedExportJson->has("simTimeSec"));

    // 8. Malformed JSON, unknown command, and unknown preset -> error response without crash
    ASSERT_TRUE(ws.sendText("{not valid json"));
    const auto errMalformed = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "error";
    }, 5000, /*failOnError=*/false);
    ASSERT_TRUE(errMalformed.has_value());
    EXPECT_FALSE(errMalformed->getString("code").empty());
    EXPECT_FALSE(errMalformed->getString("message").empty());

    ASSERT_TRUE(ws.sendText(R"({"type":"unknown_command_e3"})"));
    const auto errCmd = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "error";
    }, 5000, /*failOnError=*/false);
    ASSERT_TRUE(errCmd.has_value());

    ASSERT_TRUE(ws.sendText(R"({"type":"set_preset","preset":"NonExistentPreset_E3"})"));
    const auto errPreset = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "error";
    }, 5000, /*failOnError=*/false);
    ASSERT_TRUE(errPreset.has_value());

    // Server remains healthy after error inputs
    ASSERT_TRUE(ws.sendText(R"({"type":"ping","ts":999000})"));
    const auto pongFinal = ws.recvTextMatching([](const JsonValue& v) {
        return v.getString("type") == "pong" && v.getUint64("ts") == 999000ull;
    });
    ASSERT_TRUE(pongFinal.has_value());

    ws.close();
    EXPECT_TRUE(waitForCondition([&]() { return server.connectedClientCount() == 0u; }));
    EXPECT_EQ(server.connectedClientCount(), 0u);
}

TEST(ServerProtocolTest, SimServerCliHelpFlagParity) {
    const std::string binPath = resolveSimServerBin();
    if (binPath.empty()) {
        GTEST_SKIP() << "sim_server target not defined in this build";
    }
    ASSERT_EQ(::access(binPath.c_str(), X_OK), 0) << "sim_server binary not executable: " << binPath;

    std::string out;
    const int exitCode = runProcessCapture(binPath, {"--help"}, out);
    EXPECT_EQ(exitCode, 0) << out;
    EXPECT_NE(out.find("--port"), std::string::npos);
    EXPECT_NE(out.find("--bind"), std::string::npos);
    EXPECT_NE(out.find("--web-root"), std::string::npos);
    EXPECT_NE(out.find("--particles"), std::string::npos);
    EXPECT_NE(out.find("--preset"), std::string::npos);
    EXPECT_NE(out.find("--target-fps"), std::string::npos);
}

} // namespace
