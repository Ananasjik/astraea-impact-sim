#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "simulation.hpp"

namespace astraea::server {

struct ServerConfig {
    std::string bindHost           = "127.0.0.1";
    uint16_t    port               = 8080;
    std::string webRoot            = "./web/dist";
    uint32_t    targetFps          = 30;
    bool        compressBinaryZlib = false;
};

class HttpWsServer {
public:
    HttpWsServer(Simulation& sim, const ServerConfig& cfg);
    ~HttpWsServer();

    HttpWsServer(const HttpWsServer&) = delete;
    HttpWsServer& operator=(const HttpWsServer&) = delete;

    /// Bind and listen on configured host/port. Returns false if bind/listen fails.
    bool start();

    /// Signal the event loop to stop and wake poll() immediately. Async-signal-safe.
    void stop();

    /// Block processing HTTP, WebSocket, and simulation stepping until stop() is called.
    void runEventLoop();

    /// Actual bound TCP port (resolves ephemeral port when cfg.port == 0).
    uint16_t boundPort() const;

    /// Current number of active connected WebSocket clients.
    size_t connectedClientCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace astraea::server
