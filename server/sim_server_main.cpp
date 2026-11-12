#include <atomic>
#include <charconv>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "http_ws_server.hpp"
#include "presets.hpp"
#include "simulation.hpp"

namespace {

std::atomic<astraea::server::HttpWsServer*> gServerInstance{nullptr};

extern "C" void handleSignal(int /*signum*/) {
    astraea::server::HttpWsServer* srv = gServerInstance.load(std::memory_order_acquire);
    if (srv != nullptr) {
        srv->stop();
    }
}

void installSignalHandlers() {
    struct sigaction sa{};
    sa.sa_handler = handleSignal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    (void)::sigaction(SIGINT, &sa, nullptr);
    (void)::sigaction(SIGTERM, &sa, nullptr);
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --port <0..65535>     TCP port to listen on (0 for ephemeral, default: 8080)\n"
              << "  --bind <host>         Bind host/address (default: 127.0.0.1)\n"
              << "  --web-root <dir>      Static frontend asset directory (default: ./web/dist)\n"
              << "  --particles <uint>    Initial particle count (default: 100000)\n"
              << "  --preset <name>       Initial scenario preset (default: LivingGaiaAsteroidCataclysm)\n"
              << "  --target-fps <uint>   Simulation broadcast cadence in FPS (default: 30)\n"
              << "  --zlib                Enable zlib compression on PCLS binary payloads\n"
              << "  --help                Print this help message and exit\n";
}

uint64_t parseStrictUnsignedInRange(
    const std::string& raw,
    uint64_t minVal,
    uint64_t maxVal,
    const char* flagName) {
    if (raw.empty()) {
        throw std::invalid_argument(std::string("Empty value for ") + flagName);
    }
    uint64_t val = 0;
    const char* begin = raw.data();
    const char* end = raw.data() + raw.size();
    const auto res = std::from_chars(begin, end, val);
    if (res.ec != std::errc() || res.ptr != end || val < minVal || val > maxVal) {
        throw std::invalid_argument(
            std::string(flagName) + " must be an integer in [" +
            std::to_string(minVal) + ", " + std::to_string(maxVal) + "]");
    }
    return val;
}

} // namespace

int main(int argc, char** argv) {
    astraea::server::ServerConfig serverCfg;
    std::string presetName = "LivingGaiaAsteroidCataclysm";
    uint32_t particles = 100000;

    auto requireValue = [&](int& idx, const char* flag) -> std::string {
        if (idx + 1 >= argc) {
            throw std::invalid_argument(std::string("Missing value for ") + flag);
        }
        return std::string(argv[++idx]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            } else if (arg == "--port") {
                serverCfg.port = static_cast<uint16_t>(
                    parseStrictUnsignedInRange(requireValue(i, "--port"), 0, 65535, "--port"));
            } else if (arg == "--bind") {
                serverCfg.bindHost = requireValue(i, "--bind");
                if (serverCfg.bindHost.empty()) {
                    throw std::invalid_argument("--bind must be non-empty");
                }
            } else if (arg == "--web-root") {
                serverCfg.webRoot = requireValue(i, "--web-root");
            } else if (arg == "--particles") {
                particles = static_cast<uint32_t>(
                    parseStrictUnsignedInRange(requireValue(i, "--particles"), 2, 2000000, "--particles"));
            } else if (arg == "--preset") {
                presetName = requireValue(i, "--preset");
            } else if (arg == "--target-fps") {
                serverCfg.targetFps = static_cast<uint32_t>(
                    parseStrictUnsignedInRange(requireValue(i, "--target-fps"), 1, 1000, "--target-fps"));
            } else if (arg == "--zlib") {
                serverCfg.compressBinaryZlib = true;
            } else {
                throw std::invalid_argument("Unknown option: " + arg);
            }
        }

        const astraea::ScenarioPresetId presetId = astraea::parsePresetName(presetName);
        const astraea::CollisionScenarioConfig scenarioCfg = astraea::makePresetConfig(presetId, particles);

        astraea::Simulation sim(particles);
        sim.loadScenario(scenarioCfg);

        astraea::server::HttpWsServer server(sim, serverCfg);
        if (!server.start()) {
            std::cerr << "Failed to bind server on " << serverCfg.bindHost << ':' << serverCfg.port << '\n';
            return 1;
        }

        gServerInstance.store(&server, std::memory_order_release);
        installSignalHandlers();

        std::cout << "Astraea Collision Server listening on http://"
                  << serverCfg.bindHost << ':' << server.boundPort() << '\n'
                  << "Preset: " << scenarioCfg.presetName
                  << " | Particles: " << scenarioCfg.totalParticles
                  << " | Target FPS: " << serverCfg.targetFps
                  << " | Zlib: " << (serverCfg.compressBinaryZlib ? "on" : "off")
                  << " | Web root: " << serverCfg.webRoot << std::endl;

        server.runEventLoop();
        gServerInstance.store(nullptr, std::memory_order_release);
        return 0;
    } catch (const std::exception& ex) {
        gServerInstance.store(nullptr, std::memory_order_release);
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
