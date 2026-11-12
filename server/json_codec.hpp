#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "types.hpp"

namespace astraea::server {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    Type type = Type::Null;
    bool boolVal = false;
    double numVal = 0.0;
    std::string strVal;
    Array arrVal;
    Object objVal;

    bool isNull() const { return type == Type::Null; }
    bool isBool() const { return type == Type::Bool; }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }
    bool isArray() const { return type == Type::Array; }
    bool isObject() const { return type == Type::Object; }

    const JsonValue* find(const std::string& key) const;
};

bool parseJson(const std::string& input, JsonValue& out, std::string& outError);
std::string jsonEscape(const std::string& s);
std::string formatFiniteJsonNumber(double value);
std::string serializeJson(const JsonValue& value);

const char* outcomeToString(CollisionOutcome outcome);
const char* archetypeToString(PlanetArchetype archetype);
bool parseArchetypeName(const std::string& name, PlanetArchetype& outArchetype);
bool isValidVisMode(const std::string& mode);

struct CustomScenarioDto {
    std::string targetArchetype;
    std::string impactorArchetype;
    double      targetMassKg = units::M_EARTH_SI;
    double      impactorMassKg = 0.025 * units::M_EARTH_SI;
    double      impactVelocityRatioVesc = 1.65;
    double      impactParamB = 0.30;
    double      initialSepRadii = 3.2;
    double      targetSpinRadS = 7.292115e-5;
    uint32_t    particles = 100000;
};

bool buildCustomScenarioConfig(
    const CustomScenarioDto& dto,
    CollisionScenarioConfig& outConfig,
    std::string& outError);

enum class ClientCommandType {
    SetPreset,
    LaunchCustom,
    Play,
    Pause,
    Step,
    SetTimescale,
    SetSlicePlane,
    SetVisMode,
    RequestExport,
    Ping
};

struct ClientCommand {
    ClientCommandType type = ClientCommandType::Ping;
    std::string preset;
    bool        hasParticles = false;
    uint32_t    particles = 100000;
    CustomScenarioDto customConfig;
    uint32_t    stepCount = 1;
    double      timeScale = 1.0;
    SlicePlaneConfig slicePlane{};
    std::string visMode = "NaturalBlackbody";
    std::string exportFormat = "json";
    double      pingTs = 0.0;
};

bool parseClientCommand(
    const std::string& jsonText,
    ClientCommand& outCmd,
    std::string& outErrCode,
    std::string& outErrMsg);

std::string serializeTelemetryDto(
    const TelemetrySnapshot& snap,
    bool isPlaying,
    const std::string& presetName);

std::string serializePresetLoaded(
    const std::string& presetName,
    uint32_t particleCount,
    uint32_t tracerCount,
    uint32_t tracerGridRes,
    bool livingBodyPresent);

std::string serializeExportData(
    const std::string& format,
    const std::string& filename,
    const std::string& content);

std::string serializePong(double ts);

std::string serializeError(const std::string& code, const std::string& message);

} // namespace astraea::server
