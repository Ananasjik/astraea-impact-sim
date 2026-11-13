#include "http_ws_server.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/sockios.h>
#include <sys/ioctl.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "binary_protocol.hpp"
#include "json_codec.hpp"
#include "presets.hpp"
#include "ws_crypto.hpp"

namespace astraea::server {
namespace {

constexpr std::size_t kMaxHttpHeaderBytes = 65536u;
constexpr std::size_t kMaxWsFrameBytes = 16u * 1024u * 1024u;
constexpr std::size_t kMaxStaticFileBytes = 64u * 1024u * 1024u;
constexpr std::size_t kStreamBackpressureBytes = 256u * 1024u;
constexpr std::size_t kMaxBroadcastQueueBytes = 64u * 1024u * 1024u;

enum class ConnState {
    HttpRequest,
    WebSocket,
    Closing
};

struct ClientConn {
    int fd = -1;
    ConnState state = ConnState::HttpRequest;
    bool countedAsWs = false;

    std::vector<uint8_t> recvBuf;
    std::deque<std::vector<uint8_t>> sendQueue;
    std::size_t sendOffset = 0;
    std::size_t totalQueuedBytes = 0;

    bool fragActive = false;
    uint8_t fragOpcode = 0;
    std::vector<uint8_t> fragBuf;
};

bool setNonBlockingCloexec(int fd) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        return false;
    }
    if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        return false;
    }
    const int fdFlags = ::fcntl(fd, F_GETFD, 0);
    if (fdFlags >= 0) {
        (void)::fcntl(fd, F_SETFD, fdFlags | FD_CLOEXEC);
    }
    return true;
}

std::string toLowerAscii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

std::string trimAscii(const std::string& s) {
    std::size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) {
        ++a;
    }
    std::size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) {
        --b;
    }
    return s.substr(a, b - a);
}

bool headerContainsToken(const std::string& headerValue, const std::string& needleLower) {
    std::size_t start = 0;
    while (start <= headerValue.size()) {
        std::size_t comma = headerValue.find(',', start);
        if (comma == std::string::npos) {
            comma = headerValue.size();
        }
        const std::string tok = toLowerAscii(trimAscii(headerValue.substr(start, comma - start)));
        if (tok == needleLower) {
            return true;
        }
        start = comma + 1u;
    }
    return false;
}

bool hasTraversalCharsOrDotDot(const std::string& path) {
    if (path.find('\0') != std::string::npos || path.find('\\') != std::string::npos) {
        return true;
    }
    std::size_t i = 0;
    while (i <= path.size()) {
        std::size_t slash = path.find('/', i);
        if (slash == std::string::npos) {
            slash = path.size();
        }
        const std::string seg = path.substr(i, slash - i);
        if (seg == "..") {
            return true;
        }
        if (slash == path.size()) {
            break;
        }
        i = slash + 1u;
    }
    return false;
}

bool percentDecode(const std::string& raw, std::string& out) {
    out.clear();
    out.reserve(raw.size());
    auto hexDigit = [](char c, uint8_t& val) -> bool {
        if (c >= '0' && c <= '9') {
            val = static_cast<uint8_t>(c - '0');
            return true;
        }
        if (c >= 'a' && c <= 'f') {
            val = static_cast<uint8_t>(10 + (c - 'a'));
            return true;
        }
        if (c >= 'A' && c <= 'F') {
            val = static_cast<uint8_t>(10 + (c - 'A'));
            return true;
        }
        return false;
    };

    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '%') {
            if (i + 2u >= raw.size()) {
                return false;
            }
            uint8_t hi = 0;
            uint8_t lo = 0;
            if (!hexDigit(raw[i + 1u], hi) || !hexDigit(raw[i + 2u], lo)) {
                return false;
            }
            out.push_back(static_cast<char>((hi << 4) | lo));
            i += 2u;
        } else {
            out.push_back(raw[i]);
        }
    }
    return true;
}

bool isPathWithinRoot(const std::filesystem::path& rootCanon, const std::filesystem::path& targetCanon) {
    auto rNorm = rootCanon.lexically_normal();
    auto tNorm = targetCanon.lexically_normal();

    auto stripTrailingEmpty = [](std::filesystem::path& p) {
        std::vector<std::filesystem::path> parts;
        for (const auto& part : p) {
            parts.push_back(part);
        }
        while (parts.size() > 1u && parts.back().empty()) {
            parts.pop_back();
        }
        std::filesystem::path rebuilt;
        for (const auto& part : parts) {
            rebuilt /= part;
        }
        p = rebuilt;
    };

    stripTrailingEmpty(rNorm);
    stripTrailingEmpty(tNorm);

    auto rIt = rNorm.begin();
    auto tIt = tNorm.begin();
    for (; rIt != rNorm.end() && tIt != tNorm.end(); ++rIt, ++tIt) {
        if (*rIt != *tIt) {
            return false;
        }
    }
    return rIt == rNorm.end();
}

const char* mimeTypeForExtension(const std::filesystem::path& path) {
    const std::string ext = toLowerAscii(path.extension().string());
    if (ext == ".html" || ext == ".htm") {
        return "text/html; charset=utf-8";
    }
    if (ext == ".js" || ext == ".mjs") {
        return "application/javascript; charset=utf-8";
    }
    if (ext == ".css") {
        return "text/css; charset=utf-8";
    }
    if (ext == ".json") {
        return "application/json; charset=utf-8";
    }
    if (ext == ".svg") {
        return "image/svg+xml";
    }
    if (ext == ".png") {
        return "image/png";
    }
    if (ext == ".jpg" || ext == ".jpeg") {
        return "image/jpeg";
    }
    if (ext == ".wasm") {
        return "application/wasm";
    }
    if (ext == ".ico") {
        return "image/x-icon";
    }
    if (ext == ".txt") {
        return "text/plain; charset=utf-8";
    }
    return "application/octet-stream";
}

std::vector<uint8_t> buildHttpResponse(
    int statusCode,
    const char* statusText,
    const char* contentType,
    const std::string& body,
    bool headOnly = false,
    const std::string& extraHeaders = "") {
    std::ostringstream os;
    os << "HTTP/1.1 " << statusCode << ' ' << statusText << "\r\n"
       << "Content-Type: " << contentType << "\r\n"
       << "Content-Length: " << body.size() << "\r\n"
       << "Connection: close\r\n"
       << "X-Content-Type-Options: nosniff\r\n"
       << "Cache-Control: no-cache\r\n"
       << extraHeaders
       << "\r\n";
    if (!headOnly && !body.empty()) {
        os.write(body.data(), static_cast<std::streamsize>(body.size()));
    }
    const std::string raw = os.str();
    return std::vector<uint8_t>(raw.begin(), raw.end());
}

std::vector<uint8_t> encodeWsFrame(uint8_t opcode, const uint8_t* payload, std::size_t len) {
    std::vector<uint8_t> frame;
    frame.reserve(10u + len);
    frame.push_back(static_cast<uint8_t>(0x80u | (opcode & 0x0Fu)));

    if (len <= 125u) {
        frame.push_back(static_cast<uint8_t>(len));
    } else if (len <= 65535u) {
        frame.push_back(126u);
        frame.push_back(static_cast<uint8_t>((len >> 8) & 0xFFu));
        frame.push_back(static_cast<uint8_t>(len & 0xFFu));
    } else {
        frame.push_back(127u);
        const uint64_t len64 = static_cast<uint64_t>(len);
        for (int shift = 56; shift >= 0; shift -= 8) {
            frame.push_back(static_cast<uint8_t>((len64 >> shift) & 0xFFULL));
        }
    }

    if (payload != nullptr && len > 0u) {
        frame.insert(frame.end(), payload, payload + len);
    }
    return frame;
}

std::vector<uint8_t> encodeWsTextFrame(const std::string& text) {
    return encodeWsFrame(
        0x01u,
        reinterpret_cast<const uint8_t*>(text.data()),
        text.size());
}

std::vector<uint8_t> encodeWsBinaryFrame(const std::vector<uint8_t>& data) {
    return encodeWsFrame(0x02u, data.data(), data.size());
}

std::vector<uint8_t> encodeWsCloseFrame(uint16_t code) {
    uint8_t body[2] = {
        static_cast<uint8_t>((code >> 8) & 0xFFu),
        static_cast<uint8_t>(code & 0xFFu)
    };
    return encodeWsFrame(0x08u, body, 2u);
}

int bindAndListenAddress(const struct addrinfo* ai, uint16_t& outPort) {
    const int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd < 0) {
        return -1;
    }

    const int yes = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    if (!setNonBlockingCloexec(fd)) {
        ::close(fd);
        return -1;
    }

    if (::bind(fd, ai->ai_addr, ai->ai_addrlen) < 0) {
        ::close(fd);
        return -1;
    }

    if (::listen(fd, 128) < 0) {
        ::close(fd);
        return -1;
    }

    struct sockaddr_storage ss{};
    socklen_t ssLen = sizeof(ss);
    if (::getsockname(fd, reinterpret_cast<struct sockaddr*>(&ss), &ssLen) < 0) {
        ::close(fd);
        return -1;
    }

    if (ss.ss_family == AF_INET) {
        const auto* s4 = reinterpret_cast<const struct sockaddr_in*>(&ss);
        outPort = ntohs(s4->sin_port);
    } else if (ss.ss_family == AF_INET6) {
        const auto* s6 = reinterpret_cast<const struct sockaddr_in6*>(&ss);
        outPort = ntohs(s6->sin6_port);
    } else {
        ::close(fd);
        return -1;
    }

    return fd;
}

} // namespace

struct HttpWsServer::Impl {
    Simulation& sim;
    ServerConfig cfg;

    std::mutex simMutex;
    std::atomic<bool> stopRequested{false};
    std::atomic<bool> loopRunning{false};
    std::atomic<uint16_t> boundPort{0};
    std::atomic<std::size_t> wsClientCount{0};

    int listenFd = -1;
    int wakeReadFd = -1;
    std::atomic<int> wakeWriteFd{-1};

    bool playing = false;
    bool includeSpeedArray = false;
    std::string visMode = "NaturalBlackbody";
    std::chrono::steady_clock::time_point nextStepTime{};

    std::vector<ClientConn> clients;

    Impl(Simulation& s, const ServerConfig& c) : sim(s), cfg(c) {}

    ~Impl() {
        cleanupNetworkSockets();
        closeWakePipe();
    }

    void closeWakePipe() {
        const int wfd = wakeWriteFd.exchange(-1, std::memory_order_acq_rel);
        if (wfd >= 0) {
            ::close(wfd);
        }
        if (wakeReadFd >= 0) {
            ::close(wakeReadFd);
            wakeReadFd = -1;
        }
    }

    void decrementWsCount(ClientConn& c) {
        if (c.countedAsWs) {
            c.countedAsWs = false;
            wsClientCount.fetch_sub(1u, std::memory_order_relaxed);
        }
    }

    void cleanupNetworkSockets() {
        for (ClientConn& c : clients) {
            decrementWsCount(c);
            if (c.fd >= 0) {
                ::close(c.fd);
                c.fd = -1;
            }
        }
        clients.clear();
        wsClientCount.store(0u, std::memory_order_relaxed);

        if (listenFd >= 0) {
            ::close(listenFd);
            listenFd = -1;
        }
    }

    static void queueBytes(ClientConn& c, std::vector<uint8_t>&& bytes) {
        if (bytes.empty()) {
            return;
        }
        c.totalQueuedBytes += bytes.size();
        c.sendQueue.push_back(std::move(bytes));
    }

    static bool flushClient(ClientConn& c) {
        while (!c.sendQueue.empty()) {
            const std::vector<uint8_t>& front = c.sendQueue.front();
            const std::size_t rem = front.size() - c.sendOffset;
            const ssize_t sent = ::send(
                c.fd,
                front.data() + c.sendOffset,
                rem,
                MSG_NOSIGNAL);
            if (sent > 0) {
                c.sendOffset += static_cast<std::size_t>(sent);
                c.totalQueuedBytes -= static_cast<std::size_t>(sent);
                if (c.sendOffset >= front.size()) {
                    c.sendQueue.pop_front();
                    c.sendOffset = 0;
                }
            } else if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                return true;
            } else if (sent < 0 && errno == EINTR) {
                continue;
            } else {
                return false;
            }
        }
        return true;
    }

    void beginClosingWsClient(ClientConn& c, uint16_t closeCode) {
        decrementWsCount(c);
        c.state = ConnState::Closing;
        queueBytes(c, encodeWsCloseFrame(closeCode));
    }

    void extractCurrentFrameAndTelemetry(
        TelemetrySnapshot& outSnap,
        std::string& outPresetName,
        std::vector<uint8_t>& outBinaryFrame) {
        std::vector<uint8_t> particleBytes;
        std::vector<uint8_t> tracerBytes;
        std::vector<uint8_t> speedBytes;
        {
            std::lock_guard<std::mutex> lock(simMutex);
            outSnap = sim.computeTelemetry();
            outPresetName = sim.activeConfig().presetName;
            sim.extractRenderFrame(
                particleBytes,
                tracerBytes,
                includeSpeedArray ? &speedBytes : nullptr);
        }
        outBinaryFrame = buildBinaryFrame(
            outSnap,
            particleBytes,
            tracerBytes,
            includeSpeedArray ? &speedBytes : nullptr,
            cfg.compressBinaryZlib);
    }

    void broadcastWsText(const std::string& json) {
        const std::vector<uint8_t> frame = encodeWsTextFrame(json);
        for (ClientConn& c : clients) {
            if (c.state == ConnState::WebSocket && c.totalQueuedBytes <= kMaxBroadcastQueueBytes) {
                std::vector<uint8_t> copy = frame;
                queueBytes(c, std::move(copy));
            }
        }
    }

    static std::size_t effectiveQueuedBytes(const ClientConn& c) {
        std::size_t queued = c.totalQueuedBytes;
#ifdef __linux__
        if (c.fd >= 0) {
            int outq = 0;
            if (::ioctl(c.fd, TIOCOUTQ, &outq) == 0 && outq > 0) {
                queued += static_cast<std::size_t>(outq);
            }
        }
#endif
        return queued;
    }

    void broadcastWsBinary(const std::vector<uint8_t>& rawPclsFrame, std::size_t maxQueueLimit = kMaxBroadcastQueueBytes) {
        const std::vector<uint8_t> frame = encodeWsBinaryFrame(rawPclsFrame);
        for (ClientConn& c : clients) {
            if (c.state == ConnState::WebSocket && effectiveQueuedBytes(c) <= maxQueueLimit) {
                std::vector<uint8_t> copy = frame;
                queueBytes(c, std::move(copy));
            }
        }
    }

    void pushInitialSyncToClient(ClientConn& c) {
        TelemetrySnapshot snap{};
        std::string presetName;
        std::vector<uint8_t> binFrame;
        extractCurrentFrameAndTelemetry(snap, presetName, binFrame);

        queueBytes(c, encodeWsTextFrame(serializePresetLoaded(
            presetName,
            snap.particleCount,
            snap.tracerCount,
            snap.tracerGridRes,
            snap.livingBodyPresent)));
        queueBytes(c, encodeWsTextFrame(serializeTelemetryDto(snap, playing, presetName)));
        queueBytes(c, encodeWsBinaryFrame(binFrame));
    }

    void handleHttpRequest(ClientConn& c) {
        static const char kHeaderTerm[] = "\r\n\r\n";
        auto it = std::search(
            c.recvBuf.begin(),
            c.recvBuf.end(),
            kHeaderTerm,
            kHeaderTerm + 4);
        if (it == c.recvBuf.end()) {
            if (c.recvBuf.size() > kMaxHttpHeaderBytes) {
                queueBytes(c, buildHttpResponse(400, "Bad Request", "text/plain; charset=utf-8", "Header too large"));
                c.state = ConnState::Closing;
            }
            return;
        }

        const std::size_t hdrEnd = static_cast<std::size_t>(std::distance(c.recvBuf.begin(), it));
        const std::string headerBlock(
            reinterpret_cast<const char*>(c.recvBuf.data()),
            hdrEnd);
        c.recvBuf.erase(c.recvBuf.begin(), c.recvBuf.begin() + static_cast<std::ptrdiff_t>(hdrEnd + 4u));

        std::istringstream iss(headerBlock);
        std::string requestLine;
        if (!std::getline(iss, requestLine)) {
            queueBytes(c, buildHttpResponse(400, "Bad Request", "text/plain; charset=utf-8", "Malformed request"));
            c.state = ConnState::Closing;
            return;
        }
        if (!requestLine.empty() && requestLine.back() == '\r') {
            requestLine.pop_back();
        }

        std::istringstream lineStream(requestLine);
        std::string method;
        std::string rawTarget;
        std::string httpVersion;
        if (!(lineStream >> method >> rawTarget >> httpVersion)) {
            queueBytes(c, buildHttpResponse(400, "Bad Request", "text/plain; charset=utf-8", "Invalid request line"));
            c.state = ConnState::Closing;
            return;
        }

        std::unordered_map<std::string, std::string> headers;
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            const std::size_t colon = line.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            const std::string key = toLowerAscii(trimAscii(line.substr(0, colon)));
            const std::string val = trimAscii(line.substr(colon + 1u));
            headers[key] = val;
        }

        if (method != "GET" && method != "HEAD") {
            queueBytes(c, buildHttpResponse(405, "Method Not Allowed", "text/plain; charset=utf-8", "Method Not Allowed"));
            c.state = ConnState::Closing;
            return;
        }

        // Strip query and fragment
        std::string rawPath = rawTarget;
        const std::size_t fragPos = rawPath.find('#');
        if (fragPos != std::string::npos) {
            rawPath.resize(fragPos);
        }
        const std::size_t queryPos = rawPath.find('?');
        if (queryPos != std::string::npos) {
            rawPath.resize(queryPos);
        }

        if (hasTraversalCharsOrDotDot(rawPath)) {
            queueBytes(c, buildHttpResponse(403, "Forbidden", "text/plain; charset=utf-8", "Forbidden"));
            c.state = ConnState::Closing;
            return;
        }

        std::string decodedPath;
        if (!percentDecode(rawPath, decodedPath)) {
            queueBytes(c, buildHttpResponse(400, "Bad Request", "text/plain; charset=utf-8", "Invalid URL encoding"));
            c.state = ConnState::Closing;
            return;
        }

        if (hasTraversalCharsOrDotDot(decodedPath)) {
            queueBytes(c, buildHttpResponse(403, "Forbidden", "text/plain; charset=utf-8", "Forbidden"));
            c.state = ConnState::Closing;
            return;
        }

        // Check WebSocket upgrade
        const auto upgIt = headers.find("upgrade");
        const auto connIt = headers.find("connection");
        if (upgIt != headers.end() && connIt != headers.end() &&
            toLowerAscii(trimAscii(upgIt->second)) == "websocket" &&
            headerContainsToken(connIt->second, "upgrade")) {
            if (method != "GET") {
                queueBytes(c, buildHttpResponse(405, "Method Not Allowed", "text/plain; charset=utf-8", "GET required"));
                c.state = ConnState::Closing;
                return;
            }
            const auto verIt = headers.find("sec-websocket-version");
            if (verIt != headers.end() && trimAscii(verIt->second) != "13") {
                queueBytes(c, buildHttpResponse(
                    400,
                    "Bad Request",
                    "text/plain; charset=utf-8",
                    "Unsupported WebSocket version",
                    false,
                    "Sec-WebSocket-Version: 13\r\n"));
                c.state = ConnState::Closing;
                return;
            }
            const auto keyIt = headers.find("sec-websocket-key");
            if (keyIt == headers.end() || trimAscii(keyIt->second).empty()) {
                queueBytes(c, buildHttpResponse(400, "Bad Request", "text/plain; charset=utf-8", "Missing Sec-WebSocket-Key"));
                c.state = ConnState::Closing;
                return;
            }

            const std::string acceptKey = sha1Base64AcceptKey(keyIt->second);
            std::ostringstream wsResp;
            wsResp << "HTTP/1.1 101 Switching Protocols\r\n"
                   << "Upgrade: websocket\r\n"
                   << "Connection: Upgrade\r\n"
                   << "Sec-WebSocket-Accept: " << acceptKey << "\r\n\r\n";
            const std::string wsHdr = wsResp.str();
            queueBytes(c, std::vector<uint8_t>(wsHdr.begin(), wsHdr.end()));

            c.state = ConnState::WebSocket;
            if (!c.countedAsWs) {
                c.countedAsWs = true;
                wsClientCount.fetch_add(1u, std::memory_order_relaxed);
            }

            pushInitialSyncToClient(c);
            if (!c.recvBuf.empty()) {
                processWsFrames(c);
            }
            return;
        }

        // Static file serving
        const bool headOnly = (method == "HEAD");
        std::string relStr = decodedPath;
        while (!relStr.empty() && relStr.front() == '/') {
            relStr.erase(relStr.begin());
        }
        if (relStr.empty()) {
            relStr = "index.html";
        }

        std::error_code ec;
        const std::filesystem::path rootPath(cfg.webRoot);
        const std::filesystem::path rootCanon = std::filesystem::weakly_canonical(rootPath, ec);
        if (ec || !std::filesystem::exists(rootCanon, ec) || !std::filesystem::is_directory(rootCanon, ec)) {
            queueBytes(c, buildHttpResponse(404, "Not Found", "text/plain; charset=utf-8", "Not Found", headOnly));
            c.state = ConnState::Closing;
            return;
        }

        std::filesystem::path candidate = rootCanon / std::filesystem::path(relStr);
        std::filesystem::path targetCanon = std::filesystem::weakly_canonical(candidate, ec);
        if (ec) {
            queueBytes(c, buildHttpResponse(403, "Forbidden", "text/plain; charset=utf-8", "Forbidden", headOnly));
            c.state = ConnState::Closing;
            return;
        }
        if (!isPathWithinRoot(rootCanon, targetCanon)) {
            queueBytes(c, buildHttpResponse(403, "Forbidden", "text/plain; charset=utf-8", "Forbidden", headOnly));
            c.state = ConnState::Closing;
            return;
        }

        if (std::filesystem::is_directory(targetCanon, ec)) {
            targetCanon = std::filesystem::weakly_canonical(targetCanon / "index.html", ec);
            if (ec || !isPathWithinRoot(rootCanon, targetCanon)) {
                queueBytes(c, buildHttpResponse(403, "Forbidden", "text/plain; charset=utf-8", "Forbidden", headOnly));
                c.state = ConnState::Closing;
                return;
            }
        }

        if (!std::filesystem::exists(targetCanon, ec) || !std::filesystem::is_regular_file(targetCanon, ec)) {
            queueBytes(c, buildHttpResponse(404, "Not Found", "text/plain; charset=utf-8", "Not Found", headOnly));
            c.state = ConnState::Closing;
            return;
        }

        const uintmax_t fileSize = std::filesystem::file_size(targetCanon, ec);
        if (ec || fileSize > kMaxStaticFileBytes) {
            queueBytes(c, buildHttpResponse(500, "Internal Server Error", "text/plain; charset=utf-8", "File read error", headOnly));
            c.state = ConnState::Closing;
            return;
        }

        std::ifstream ifs(targetCanon, std::ios::binary);
        if (!ifs) {
            queueBytes(c, buildHttpResponse(404, "Not Found", "text/plain; charset=utf-8", "Not Found", headOnly));
            c.state = ConnState::Closing;
            return;
        }

        std::string body(static_cast<std::size_t>(fileSize), '\0');
        if (fileSize > 0u) {
            ifs.read(body.data(), static_cast<std::streamsize>(fileSize));
            if (!ifs) {
                queueBytes(c, buildHttpResponse(500, "Internal Server Error", "text/plain; charset=utf-8", "File read failed", headOnly));
                c.state = ConnState::Closing;
                return;
            }
        }

        queueBytes(c, buildHttpResponse(200, "OK", mimeTypeForExtension(targetCanon), body, headOnly));
        c.state = ConnState::Closing;
    }

    void handleWsCommand(ClientConn& c, const std::string& jsonText) {
        ClientCommand cmd;
        std::string errCode;
        std::string errMsg;
        if (!parseClientCommand(jsonText, cmd, errCode, errMsg)) {
            queueBytes(c, encodeWsTextFrame(serializeError(errCode, errMsg)));
            return;
        }

        switch (cmd.type) {
            case ClientCommandType::SetPreset: {
                try {
                    const ScenarioPresetId presetId = parsePresetName(cmd.preset);
                    TelemetrySnapshot snap{};
                    std::string presetName;
                    std::vector<uint8_t> binFrame;
                    {
                        std::lock_guard<std::mutex> lock(simMutex);
                        const uint32_t activeParticles = sim.activeConfig().totalParticles > 0u
                            ? sim.activeConfig().totalParticles
                            : 100000u;
                        const uint32_t particles = cmd.hasParticles ? cmd.particles : activeParticles;
                        const CollisionScenarioConfig scenarioCfg = makePresetConfig(presetId, particles);
                        sim.loadScenario(scenarioCfg);
                        snap = sim.computeTelemetry();
                        presetName = sim.activeConfig().presetName;

                        std::vector<uint8_t> particleBytes;
                        std::vector<uint8_t> tracerBytes;
                        std::vector<uint8_t> speedBytes;
                        sim.extractRenderFrame(
                            particleBytes,
                            tracerBytes,
                            includeSpeedArray ? &speedBytes : nullptr);
                        binFrame = buildBinaryFrame(
                            snap,
                            particleBytes,
                            tracerBytes,
                            includeSpeedArray ? &speedBytes : nullptr,
                            cfg.compressBinaryZlib);
                    }
                    playing = false;
                    broadcastWsText(serializePresetLoaded(
                        presetName,
                        snap.particleCount,
                        snap.tracerCount,
                        snap.tracerGridRes,
                        snap.livingBodyPresent));
                    broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
                    broadcastWsBinary(binFrame);
                } catch (const std::exception& ex) {
                    queueBytes(c, encodeWsTextFrame(serializeError("PRESET_ERROR", ex.what())));
                }
                break;
            }

            case ClientCommandType::LaunchCustom: {
                CollisionScenarioConfig scenarioCfg;
                std::string customErr;
                if (!buildCustomScenarioConfig(cmd.customConfig, scenarioCfg, customErr)) {
                    queueBytes(c, encodeWsTextFrame(serializeError("INVALID_PARAMS", customErr)));
                    return;
                }
                try {
                    TelemetrySnapshot snap{};
                    std::string presetName;
                    std::vector<uint8_t> binFrame;
                    {
                        std::lock_guard<std::mutex> lock(simMutex);
                        sim.loadScenario(scenarioCfg);
                        snap = sim.computeTelemetry();
                        presetName = sim.activeConfig().presetName;

                        std::vector<uint8_t> particleBytes;
                        std::vector<uint8_t> tracerBytes;
                        std::vector<uint8_t> speedBytes;
                        sim.extractRenderFrame(
                            particleBytes,
                            tracerBytes,
                            includeSpeedArray ? &speedBytes : nullptr);
                        binFrame = buildBinaryFrame(
                            snap,
                            particleBytes,
                            tracerBytes,
                            includeSpeedArray ? &speedBytes : nullptr,
                            cfg.compressBinaryZlib);
                    }
                    playing = false;
                    broadcastWsText(serializePresetLoaded(
                        presetName,
                        snap.particleCount,
                        snap.tracerCount,
                        snap.tracerGridRes,
                        snap.livingBodyPresent));
                    broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
                    broadcastWsBinary(binFrame);
                } catch (const std::exception& ex) {
                    queueBytes(c, encodeWsTextFrame(serializeError("SCENARIO_ERROR", ex.what())));
                }
                break;
            }

            case ClientCommandType::Play: {
                playing = true;
                nextStepTime = std::chrono::steady_clock::now();
                TelemetrySnapshot snap{};
                std::string presetName;
                {
                    std::lock_guard<std::mutex> lock(simMutex);
                    snap = sim.computeTelemetry();
                    presetName = sim.activeConfig().presetName;
                }
                broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
                break;
            }

            case ClientCommandType::Pause: {
                playing = false;
                TelemetrySnapshot snap{};
                std::string presetName;
                {
                    std::lock_guard<std::mutex> lock(simMutex);
                    snap = sim.computeTelemetry();
                    presetName = sim.activeConfig().presetName;
                }
                broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
                break;
            }

            case ClientCommandType::Step: {
                try {
                    TelemetrySnapshot snap{};
                    std::string presetName;
                    std::vector<uint8_t> binFrame;
                    {
                        std::lock_guard<std::mutex> lock(simMutex);
                        for (uint32_t i = 0; i < cmd.stepCount; ++i) {
                            snap = sim.step(5.0);
                        }
                        presetName = sim.activeConfig().presetName;

                        std::vector<uint8_t> particleBytes;
                        std::vector<uint8_t> tracerBytes;
                        std::vector<uint8_t> speedBytes;
                        sim.extractRenderFrame(
                            particleBytes,
                            tracerBytes,
                            includeSpeedArray ? &speedBytes : nullptr);
                        binFrame = buildBinaryFrame(
                            snap,
                            particleBytes,
                            tracerBytes,
                            includeSpeedArray ? &speedBytes : nullptr,
                            cfg.compressBinaryZlib);
                    }
                    broadcastWsBinary(binFrame);
                    broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
                } catch (const std::exception& ex) {
                    queueBytes(c, encodeWsTextFrame(serializeError("STEP_ERROR", ex.what())));
                }
                break;
            }

            case ClientCommandType::SetTimescale: {
                TelemetrySnapshot snap{};
                std::string presetName;
                {
                    std::lock_guard<std::mutex> lock(simMutex);
                    sim.setTimeScale(cmd.timeScale);
                    snap = sim.computeTelemetry();
                    presetName = sim.activeConfig().presetName;
                }
                broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
                break;
            }

            case ClientCommandType::SetSlicePlane: {
                TelemetrySnapshot snap{};
                std::string presetName;
                {
                    std::lock_guard<std::mutex> lock(simMutex);
                    sim.setSlicePlane(cmd.slicePlane);
                    snap = sim.computeTelemetry();
                    presetName = sim.activeConfig().presetName;
                }
                broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
                break;
            }

            case ClientCommandType::SetVisMode: {
                visMode = cmd.visMode;
                includeSpeedArray = (visMode == "Velocity");
                TelemetrySnapshot snap{};
                std::string presetName;
                std::vector<uint8_t> binFrame;
                extractCurrentFrameAndTelemetry(snap, presetName, binFrame);
                broadcastWsBinary(binFrame);
                broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
                break;
            }

            case ClientCommandType::RequestExport: {
                std::string content;
                std::string presetName;
                {
                    std::lock_guard<std::mutex> lock(simMutex);
                    presetName = sim.activeConfig().presetName;
                    if (cmd.exportFormat == "csv") {
                        content = sim.exportTelemetryCsv();
                    } else {
                        content = sim.exportTelemetryJson();
                    }
                }
                const std::string filename =
                    (presetName.empty() ? std::string("astraea") : presetName) +
                    "_telemetry." + cmd.exportFormat;
                queueBytes(c, encodeWsTextFrame(serializeExportData(cmd.exportFormat, filename, content)));
                break;
            }

            case ClientCommandType::Ping: {
                queueBytes(c, encodeWsTextFrame(serializePong(cmd.pingTs)));
                break;
            }
        }
    }

    void processWsFrames(ClientConn& c) {
        while (c.state == ConnState::WebSocket && c.recvBuf.size() >= 2u) {
            const uint8_t b0 = c.recvBuf[0];
            const uint8_t b1 = c.recvBuf[1];
            const bool fin = (b0 & 0x80u) != 0u;
            const uint8_t rsv = static_cast<uint8_t>(b0 & 0x70u);
            const uint8_t opcode = static_cast<uint8_t>(b0 & 0x0Fu);
            const bool masked = (b1 & 0x80u) != 0u;
            const uint8_t len7 = static_cast<uint8_t>(b1 & 0x7Fu);

            if (rsv != 0u || !masked) {
                beginClosingWsClient(c, 1002u);
                return;
            }

            std::size_t hdrBytes = 2u;
            uint64_t payloadLen = len7;
            if (len7 == 126u) {
                if (c.recvBuf.size() < 4u) {
                    return;
                }
                payloadLen = (static_cast<uint64_t>(c.recvBuf[2]) << 8) |
                             static_cast<uint64_t>(c.recvBuf[3]);
                hdrBytes = 4u;
            } else if (len7 == 127u) {
                if (c.recvBuf.size() < 10u) {
                    return;
                }
                if ((c.recvBuf[2] & 0x80u) != 0u) {
                    beginClosingWsClient(c, 1002u);
                    return;
                }
                payloadLen = 0;
                for (int i = 0; i < 8; ++i) {
                    payloadLen = (payloadLen << 8) | static_cast<uint64_t>(c.recvBuf[2u + static_cast<std::size_t>(i)]);
                }
                hdrBytes = 10u;
            }

            if (payloadLen > kMaxWsFrameBytes) {
                beginClosingWsClient(c, 1009u);
                return;
            }

            const std::size_t totalFrameBytes = hdrBytes + 4u + static_cast<std::size_t>(payloadLen);
            if (c.recvBuf.size() < totalFrameBytes) {
                return;
            }

            uint8_t maskKey[4] = {
                c.recvBuf[hdrBytes + 0u],
                c.recvBuf[hdrBytes + 1u],
                c.recvBuf[hdrBytes + 2u],
                c.recvBuf[hdrBytes + 3u]
            };

            std::vector<uint8_t> payload(static_cast<std::size_t>(payloadLen));
            const std::size_t payloadOffset = hdrBytes + 4u;
            for (std::size_t i = 0; i < payload.size(); ++i) {
                payload[i] = static_cast<uint8_t>(c.recvBuf[payloadOffset + i] ^ maskKey[i & 3u]);
            }

            c.recvBuf.erase(
                c.recvBuf.begin(),
                c.recvBuf.begin() + static_cast<std::ptrdiff_t>(totalFrameBytes));

            // Control frames (0x8..0xF)
            if ((opcode & 0x08u) != 0u) {
                if (!fin || payload.size() > 125u) {
                    beginClosingWsClient(c, 1002u);
                    return;
                }
                if (opcode == 0x08u) {
                    uint16_t replyCode = 1000u;
                    if (payload.size() >= 2u) {
                        replyCode = static_cast<uint16_t>(
                            (static_cast<uint16_t>(payload[0]) << 8) |
                            static_cast<uint16_t>(payload[1]));
                    }
                    beginClosingWsClient(c, replyCode);
                    return;
                }
                if (opcode == 0x09u) {
                    queueBytes(c, encodeWsFrame(0x0Au, payload.data(), payload.size()));
                    continue;
                }
                if (opcode == 0x0Au) {
                    continue;
                }
                beginClosingWsClient(c, 1002u);
                return;
            }

            // Data frames (0x0, 0x1, 0x2)
            if (opcode == 0x01u || opcode == 0x02u) {
                if (c.fragActive) {
                    beginClosingWsClient(c, 1002u);
                    return;
                }
                if (fin) {
                    if (opcode == 0x01u) {
                        const std::string text(
                            reinterpret_cast<const char*>(payload.data()),
                            payload.size());
                        handleWsCommand(c, text);
                    }
                } else {
                    c.fragActive = true;
                    c.fragOpcode = opcode;
                    c.fragBuf = std::move(payload);
                }
            } else if (opcode == 0x00u) {
                if (!c.fragActive) {
                    beginClosingWsClient(c, 1002u);
                    return;
                }
                if (c.fragBuf.size() + payload.size() > kMaxWsFrameBytes) {
                    beginClosingWsClient(c, 1009u);
                    return;
                }
                c.fragBuf.insert(c.fragBuf.end(), payload.begin(), payload.end());
                if (fin) {
                    const uint8_t msgOp = c.fragOpcode;
                    std::vector<uint8_t> fullMsg = std::move(c.fragBuf);
                    c.fragActive = false;
                    c.fragBuf.clear();
                    if (msgOp == 0x01u) {
                        const std::string text(
                            reinterpret_cast<const char*>(fullMsg.data()),
                            fullMsg.size());
                        handleWsCommand(c, text);
                    }
                }
            } else {
                beginClosingWsClient(c, 1003u);
                return;
            }
        }
    }

    void acceptPendingClients() {
        while (listenFd >= 0) {
            struct sockaddr_storage clientAddr{};
            socklen_t clientLen = sizeof(clientAddr);
            const int clientFd = ::accept(
                listenFd,
                reinterpret_cast<struct sockaddr*>(&clientAddr),
                &clientLen);
            if (clientFd < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }

            if (!setNonBlockingCloexec(clientFd)) {
                ::close(clientFd);
                continue;
            }
            const int nodelay = 1;
            (void)::setsockopt(clientFd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
            const int sndBufBytes = 131072;
            (void)::setsockopt(clientFd, SOL_SOCKET, SO_SNDBUF, &sndBufBytes, sizeof(sndBufBytes));

            ClientConn conn;
            conn.fd = clientFd;
            conn.state = ConnState::HttpRequest;
            clients.push_back(std::move(conn));
        }
    }

    void stepSimulationAndBroadcast() {
        TelemetrySnapshot snap{};
        std::string presetName;
        std::vector<uint8_t> binFrame;
        try {
            std::lock_guard<std::mutex> lock(simMutex);
            if (sim.computeTelemetry().particleCount == 0u) {
                return;
            }
            snap = sim.step(5.0);
            presetName = sim.activeConfig().presetName;

            std::vector<uint8_t> particleBytes;
            std::vector<uint8_t> tracerBytes;
            std::vector<uint8_t> speedBytes;
            sim.extractRenderFrame(
                particleBytes,
                tracerBytes,
                includeSpeedArray ? &speedBytes : nullptr);
            binFrame = buildBinaryFrame(
                snap,
                particleBytes,
                tracerBytes,
                includeSpeedArray ? &speedBytes : nullptr,
                cfg.compressBinaryZlib);
        } catch (...) {
            playing = false;
            return;
        }

        broadcastWsBinary(binFrame, kStreamBackpressureBytes);
        broadcastWsText(serializeTelemetryDto(snap, playing, presetName));
    }
};

HttpWsServer::HttpWsServer(Simulation& sim, const ServerConfig& cfg)
    : impl_(std::make_unique<Impl>(sim, cfg)) {
    std::signal(SIGPIPE, SIG_IGN);
}

HttpWsServer::~HttpWsServer() {
    stop();
    impl_->cleanupNetworkSockets();
    impl_->closeWakePipe();
}

bool HttpWsServer::start() {
    if (impl_->listenFd >= 0 || impl_->cfg.bindHost.empty()) {
        return false;
    }

    if (impl_->wakeReadFd < 0 || impl_->wakeWriteFd.load(std::memory_order_acquire) < 0) {
        impl_->closeWakePipe();
        int wakeFds[2] = {-1, -1};
        if (::pipe2(wakeFds, O_NONBLOCK | O_CLOEXEC) < 0) {
            return false;
        }
        impl_->wakeReadFd = wakeFds[0];
        impl_->wakeWriteFd.store(wakeFds[1], std::memory_order_release);
    } else {
        uint8_t drain[64];
        while (::read(impl_->wakeReadFd, drain, sizeof(drain)) > 0) {
        }
    }

    impl_->stopRequested.store(false, std::memory_order_release);

    const std::string portStr = std::to_string(impl_->cfg.port);
    auto tryResolveAndBind = [&](int family, int extraFlags) -> int {
        struct addrinfo hints{};
        hints.ai_family = family;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE | extraFlags;

        struct addrinfo* res = nullptr;
        if (::getaddrinfo(impl_->cfg.bindHost.c_str(), portStr.c_str(), &hints, &res) != 0) {
            return -1;
        }

        int boundFd = -1;
        uint16_t actualPort = 0;
        for (const struct addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
            boundFd = bindAndListenAddress(ai, actualPort);
            if (boundFd >= 0) {
                impl_->boundPort.store(actualPort, std::memory_order_release);
                break;
            }
        }
        ::freeaddrinfo(res);
        return boundFd;
    };

    int lfd = tryResolveAndBind(AF_INET, AI_NUMERICHOST);
    if (lfd < 0) {
        lfd = tryResolveAndBind(AF_INET6, AI_NUMERICHOST);
    }
    if (lfd < 0) {
        lfd = tryResolveAndBind(AF_INET, 0);
    }
    if (lfd < 0) {
        lfd = tryResolveAndBind(AF_UNSPEC, 0);
    }

    if (lfd < 0) {
        impl_->closeWakePipe();
        impl_->boundPort.store(0u, std::memory_order_release);
        return false;
    }

    impl_->listenFd = lfd;
    return true;
}

void HttpWsServer::stop() {
    impl_->stopRequested.store(true, std::memory_order_release);
    const int fd = impl_->wakeWriteFd.load(std::memory_order_acquire);
    if (fd >= 0) {
        const uint8_t b = 1;
        const ssize_t ignored = ::write(fd, &b, 1);
        (void)ignored;
    }
}

void HttpWsServer::runEventLoop() {
    if (impl_->listenFd < 0) {
        return;
    }

    impl_->loopRunning.store(true, std::memory_order_release);
    const uint32_t fps = std::max(1u, impl_->cfg.targetFps);
    const auto framePeriod = std::chrono::microseconds(1000000ULL / fps);
    impl_->nextStepTime = std::chrono::steady_clock::now() + framePeriod;

    while (!impl_->stopRequested.load(std::memory_order_acquire)) {
        std::vector<struct pollfd> pfds;
        pfds.reserve(2u + impl_->clients.size());

        pfds.push_back(pollfd{impl_->wakeReadFd, POLLIN, 0});
        pfds.push_back(pollfd{impl_->listenFd, POLLIN, 0});

        for (const ClientConn& c : impl_->clients) {
            short events = 0;
            if (c.state != ConnState::Closing) {
                events |= POLLIN;
            }
            if (!c.sendQueue.empty()) {
                events |= POLLOUT;
            }
            pfds.push_back(pollfd{c.fd, events, 0});
        }

        int timeoutMs = 200;
        if (impl_->playing) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= impl_->nextStepTime) {
                timeoutMs = 0;
            } else {
                const auto remMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    impl_->nextStepTime - now).count();
                timeoutMs = static_cast<int>(std::clamp<long long>(remMs, 0, 50));
            }
        }

        const int ready = ::poll(pfds.data(), static_cast<nfds_t>(pfds.size()), timeoutMs);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        if ((pfds[0].revents & POLLIN) != 0) {
            uint8_t drain[64];
            while (::read(impl_->wakeReadFd, drain, sizeof(drain)) > 0) {
            }
        }

        if (impl_->stopRequested.load(std::memory_order_acquire)) {
            break;
        }

        if ((pfds[1].revents & POLLIN) != 0) {
            impl_->acceptPendingClients();
        }

        const std::size_t snapshotCount = pfds.size() - 2u;
        std::vector<bool> dropClient(impl_->clients.size(), false);

        for (std::size_t idx = 0; idx < snapshotCount; ++idx) {
            ClientConn& c = impl_->clients[idx];
            const short rev = pfds[2u + idx].revents;

            if ((rev & (POLLERR | POLLNVAL)) != 0) {
                dropClient[idx] = true;
                continue;
            }

            if ((rev & POLLIN) != 0 && c.state != ConnState::Closing) {
                uint8_t buf[65536];
                bool peerClosed = false;
                bool fatalRead = false;
                while (true) {
                    const ssize_t n = ::recv(c.fd, buf, sizeof(buf), 0);
                    if (n > 0) {
                        c.recvBuf.insert(c.recvBuf.end(), buf, buf + n);
                        if (c.recvBuf.size() > kMaxWsFrameBytes + 65536u) {
                            fatalRead = true;
                            break;
                        }
                    } else if (n == 0) {
                        peerClosed = true;
                        break;
                    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        break;
                    } else if (errno == EINTR) {
                        continue;
                    } else {
                        fatalRead = true;
                        break;
                    }
                }

                if (fatalRead) {
                    dropClient[idx] = true;
                    continue;
                }

                if (c.state == ConnState::HttpRequest && !c.recvBuf.empty()) {
                    impl_->handleHttpRequest(c);
                } else if (c.state == ConnState::WebSocket && !c.recvBuf.empty()) {
                    impl_->processWsFrames(c);
                }

                if (peerClosed && c.sendQueue.empty()) {
                    dropClient[idx] = true;
                    continue;
                }
            }

            if ((rev & POLLHUP) != 0 && c.sendQueue.empty()) {
                dropClient[idx] = true;
                continue;
            }

            if (!Impl::flushClient(c)) {
                dropClient[idx] = true;
                continue;
            }

            if (c.state == ConnState::Closing && c.sendQueue.empty()) {
                dropClient[idx] = true;
            }
        }

        // Compact dropped clients before potentially broadcasting new simulation frames
        for (std::size_t i = impl_->clients.size(); i-- > 0;) {
            if (dropClient[i]) {
                impl_->decrementWsCount(impl_->clients[i]);
                if (impl_->clients[i].fd >= 0) {
                    ::close(impl_->clients[i].fd);
                }
                impl_->clients.erase(impl_->clients.begin() + static_cast<std::ptrdiff_t>(i));
            }
        }

        if (impl_->playing && std::chrono::steady_clock::now() >= impl_->nextStepTime) {
            impl_->nextStepTime = std::chrono::steady_clock::now() + framePeriod;
            impl_->stepSimulationAndBroadcast();
        }

        // Flush any newly queued responses/broadcasts immediately
        for (std::size_t i = impl_->clients.size(); i-- > 0;) {
            ClientConn& c = impl_->clients[i];
            const bool ok = Impl::flushClient(c);
            if (!ok || (c.state == ConnState::Closing && c.sendQueue.empty())) {
                impl_->decrementWsCount(c);
                if (c.fd >= 0) {
                    ::close(c.fd);
                }
                impl_->clients.erase(impl_->clients.begin() + static_cast<std::ptrdiff_t>(i));
            }
        }
    }

    impl_->cleanupNetworkSockets();
    impl_->loopRunning.store(false, std::memory_order_release);
}

uint16_t HttpWsServer::boundPort() const {
    return impl_->boundPort.load(std::memory_order_acquire);
}

size_t HttpWsServer::connectedClientCount() const {
    return impl_->wsClientCount.load(std::memory_order_acquire);
}

} // namespace astraea::server
