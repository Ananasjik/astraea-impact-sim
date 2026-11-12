#include "json_codec.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "presets.hpp"

namespace astraea::server {
namespace {

constexpr int kMaxJsonDepth = 64;

void appendUtf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | ((cp >> 6) & 0x1Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | ((cp >> 12) & 0x0Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back(static_cast<char>(0xF0u | ((cp >> 18) & 0x07u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : src_(src) {}

    bool parseRoot(JsonValue& out, std::string& err) {
        skipWs();
        if (!parseValue(out, 0, err)) {
            return false;
        }
        skipWs();
        if (pos_ != src_.size()) {
            err = "Unexpected trailing characters after JSON value";
            return false;
        }
        return true;
    }

private:
    const std::string& src_;
    std::size_t pos_ = 0;

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

    bool parseValue(JsonValue& out, int depth, std::string& err) {
        if (depth > kMaxJsonDepth) {
            err = "Maximum JSON nesting depth exceeded";
            return false;
        }
        skipWs();
        if (pos_ >= src_.size()) {
            err = "Unexpected end of JSON input";
            return false;
        }
        const char c = src_[pos_];
        if (c == '{') {
            return parseObject(out, depth + 1, err);
        }
        if (c == '[') {
            return parseArray(out, depth + 1, err);
        }
        if (c == '"') {
            out = JsonValue{};
            out.type = JsonValue::Type::String;
            return parseString(out.strVal, err);
        }
        if (c == 't') {
            return parseLiteral("true", true, out, err);
        }
        if (c == 'f') {
            return parseLiteral("false", false, out, err);
        }
        if (c == 'n') {
            if (src_.compare(pos_, 4, "null") == 0) {
                pos_ += 4;
                out = JsonValue{};
                out.type = JsonValue::Type::Null;
                return true;
            }
            err = "Invalid literal";
            return false;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parseNumber(out, err);
        }
        err = "Unexpected character in JSON";
        return false;
    }

    bool parseLiteral(const char* lit, bool val, JsonValue& out, std::string& err) {
        const std::size_t len = std::char_traits<char>::length(lit);
        if (src_.compare(pos_, len, lit) == 0) {
            pos_ += len;
            out = JsonValue{};
            out.type = JsonValue::Type::Bool;
            out.boolVal = val;
            return true;
        }
        err = "Invalid boolean literal";
        return false;
    }

    bool parseHex4(uint32_t& outHex) {
        if (pos_ + 4u > src_.size()) {
            return false;
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = src_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') {
                v |= static_cast<uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                v |= static_cast<uint32_t>(10 + (c - 'a'));
            } else if (c >= 'A' && c <= 'F') {
                v |= static_cast<uint32_t>(10 + (c - 'A'));
            } else {
                return false;
            }
        }
        outHex = v;
        return true;
    }

    bool parseString(std::string& out, std::string& err) {
        if (pos_ >= src_.size() || src_[pos_] != '"') {
            err = "Expected '\"'";
            return false;
        }
        ++pos_;
        out.clear();

        while (pos_ < src_.size()) {
            const unsigned char c = static_cast<unsigned char>(src_[pos_++]);
            if (c == '"') {
                return true;
            }
            if (c < 0x20u) {
                err = "Unescaped control character in JSON string";
                return false;
            }
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= src_.size()) {
                err = "Unterminated escape sequence in JSON string";
                return false;
            }
            const char esc = src_[pos_++];
            switch (esc) {
                case '"':  out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/'); break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!parseHex4(cp)) {
                        err = "Invalid \\uXXXX escape in JSON string";
                        return false;
                    }
                    if (cp >= 0xD800u && cp <= 0xDBFFu) {
                        if (pos_ + 2u <= src_.size() && src_[pos_] == '\\' && src_[pos_ + 1u] == 'u') {
                            pos_ += 2u;
                            uint32_t low = 0;
                            if (!parseHex4(low) || low < 0xDC00u || low > 0xDFFFu) {
                                err = "Invalid UTF-16 surrogate pair in JSON string";
                                return false;
                            }
                            cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                        } else {
                            err = "Unpaired high surrogate in JSON string";
                            return false;
                        }
                    } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
                        err = "Unpaired low surrogate in JSON string";
                        return false;
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default:
                    err = "Invalid escape sequence in JSON string";
                    return false;
            }
        }
        err = "Unterminated JSON string";
        return false;
    }

    bool parseNumber(JsonValue& out, std::string& err) {
        const std::size_t start = pos_;
        if (src_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= src_.size()) {
            err = "Invalid JSON number";
            return false;
        }
        if (src_[pos_] == '0') {
            ++pos_;
            if (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                err = "Leading zeros not allowed in JSON numbers";
                return false;
            }
        } else if (src_[pos_] >= '1' && src_[pos_] <= '9') {
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            err = "Invalid JSON number";
            return false;
        }

        if (pos_ < src_.size() && src_[pos_] == '.') {
            ++pos_;
            if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9') {
                err = "Invalid fraction in JSON number";
                return false;
            }
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        }

        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= src_.size() || src_[pos_] < '0' || src_[pos_] > '9') {
                err = "Invalid exponent in JSON number";
                return false;
            }
            while (pos_ < src_.size() && src_[pos_] >= '0' && src_[pos_] <= '9') {
                ++pos_;
            }
        }

        double parsed = 0.0;
        const auto res = std::from_chars(src_.data() + start, src_.data() + pos_, parsed);
        if (res.ec != std::errc() || res.ptr != src_.data() + pos_ || !std::isfinite(parsed)) {
            err = "Numeric overflow or unparseable JSON number";
            return false;
        }

        out = JsonValue{};
        out.type = JsonValue::Type::Number;
        out.numVal = parsed;
        return true;
    }

    bool parseArray(JsonValue& out, int depth, std::string& err) {
        ++pos_; // '['
        out = JsonValue{};
        out.type = JsonValue::Type::Array;
        skipWs();
        if (pos_ < src_.size() && src_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            JsonValue elem;
            if (!parseValue(elem, depth, err)) {
                return false;
            }
            out.arrVal.push_back(std::move(elem));
            skipWs();
            if (pos_ >= src_.size()) {
                err = "Unterminated JSON array";
                return false;
            }
            if (src_[pos_] == ']') {
                ++pos_;
                return true;
            }
            if (src_[pos_] != ',') {
                err = "Expected ',' or ']' in JSON array";
                return false;
            }
            ++pos_;
        }
    }

    bool parseObject(JsonValue& out, int depth, std::string& err) {
        ++pos_; // '{'
        out = JsonValue{};
        out.type = JsonValue::Type::Object;
        skipWs();
        if (pos_ < src_.size() && src_[pos_] == '}') {
            ++pos_;
            return true;
        }
        while (true) {
            skipWs();
            if (pos_ >= src_.size() || src_[pos_] != '"') {
                err = "Expected string key in JSON object";
                return false;
            }
            std::string key;
            if (!parseString(key, err)) {
                return false;
            }
            skipWs();
            if (pos_ >= src_.size() || src_[pos_] != ':') {
                err = "Expected ':' after key in JSON object";
                return false;
            }
            ++pos_;
            JsonValue val;
            if (!parseValue(val, depth, err)) {
                return false;
            }
            out.objVal[std::move(key)] = std::move(val);
            skipWs();
            if (pos_ >= src_.size()) {
                err = "Unterminated JSON object";
                return false;
            }
            if (src_[pos_] == '}') {
                ++pos_;
                return true;
            }
            if (src_[pos_] != ',') {
                err = "Expected ',' or '}' in JSON object";
                return false;
            }
            ++pos_;
        }
    }
};

bool getRequiredString(
    const JsonValue& obj,
    const char* key,
    std::string& out,
    std::string& errCode,
    std::string& errMsg) {
    const JsonValue* v = obj.find(key);
    if (v == nullptr || !v->isString()) {
        errCode = "INVALID_PARAMS";
        errMsg = std::string("Missing or non-string field: ") + key;
        return false;
    }
    out = v->strVal;
    return true;
}

bool getOptionalPositiveParticles(
    const JsonValue& obj,
    const char* key,
    bool& outHas,
    uint32_t& outParticles,
    std::string& errCode,
    std::string& errMsg) {
    const JsonValue* v = obj.find(key);
    if (v == nullptr || v->isNull()) {
        outHas = false;
        return true;
    }
    if (!v->isNumber() || !std::isfinite(v->numVal) || v->numVal < 2.0 || v->numVal > 2000000.0) {
        errCode = "INVALID_PARAMS";
        errMsg = "particles must be a number in [2, 2000000]";
        return false;
    }
    outHas = true;
    outParticles = static_cast<uint32_t>(std::llround(v->numVal));
    return true;
}

bool getNumberField(
    const JsonValue& obj,
    const char* key,
    double& inOutVal,
    bool required,
    std::string& errCode,
    std::string& errMsg) {
    const JsonValue* v = obj.find(key);
    if (v == nullptr) {
        if (required) {
            errCode = "INVALID_PARAMS";
            errMsg = std::string("Missing required numeric field: ") + key;
            return false;
        }
        return true;
    }
    if (!v->isNumber() || !std::isfinite(v->numVal)) {
        errCode = "INVALID_PARAMS";
        errMsg = std::string("Field must be a finite number: ") + key;
        return false;
    }
    inOutVal = v->numVal;
    return true;
}

} // namespace

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type != Type::Object) {
        return nullptr;
    }
    const auto it = objVal.find(key);
    if (it == objVal.end()) {
        return nullptr;
    }
    return &it->second;
}

bool parseJson(const std::string& input, JsonValue& out, std::string& outError) {
    JsonParser parser(input);
    return parser.parseRoot(out, outError);
}

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4u);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

std::string formatFiniteJsonNumber(double value) {
    if (!std::isfinite(value) || value == 0.0) {
        return "0";
    }
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), value);
    if (res.ec != std::errc()) {
        return "0";
    }
    return std::string(buf, res.ptr);
}

std::string serializeJson(const JsonValue& value) {
    switch (value.type) {
        case JsonValue::Type::Null:
            return "null";
        case JsonValue::Type::Bool:
            return value.boolVal ? "true" : "false";
        case JsonValue::Type::Number:
            return formatFiniteJsonNumber(value.numVal);
        case JsonValue::Type::String:
            return "\"" + jsonEscape(value.strVal) + "\"";
        case JsonValue::Type::Array: {
            std::string out = "[";
            for (std::size_t i = 0; i < value.arrVal.size(); ++i) {
                if (i > 0u) {
                    out += ',';
                }
                out += serializeJson(value.arrVal[i]);
            }
            out += ']';
            return out;
        }
        case JsonValue::Type::Object: {
            std::string out = "{";
            bool first = true;
            for (const auto& kv : value.objVal) {
                if (!first) {
                    out += ',';
                }
                first = false;
                out += "\"" + jsonEscape(kv.first) + "\":" + serializeJson(kv.second);
            }
            out += '}';
            return out;
        }
    }
    return "null";
}

const char* outcomeToString(CollisionOutcome outcome) {
    switch (outcome) {
        case CollisionOutcome::Approaching:            return "Approaching";
        case CollisionOutcome::Merging:                return "Merging";
        case CollisionOutcome::GrazeAndMerge:          return "GrazeAndMerge";
        case CollisionOutcome::HitAndRun:              return "HitAndRun";
        case CollisionOutcome::CatastrophicDisruption: return "CatastrophicDisruption";
    }
    return "Approaching";
}

const char* archetypeToString(PlanetArchetype archetype) {
    switch (archetype) {
        case PlanetArchetype::LivingTerra: return "LivingTerra";
        case PlanetArchetype::IronCore:    return "IronCore";
        case PlanetArchetype::OceanIce:    return "OceanIce";
        case PlanetArchetype::LavaIo:      return "LavaIo";
        case PlanetArchetype::GasGiant:    return "GasGiant";
        case PlanetArchetype::ProtoTheia:  return "ProtoTheia";
    }
    return "LivingTerra";
}

bool parseArchetypeName(const std::string& name, PlanetArchetype& outArchetype) {
    if (name == "LivingTerra") {
        outArchetype = PlanetArchetype::LivingTerra;
        return true;
    }
    if (name == "IronCore") {
        outArchetype = PlanetArchetype::IronCore;
        return true;
    }
    if (name == "OceanIce") {
        outArchetype = PlanetArchetype::OceanIce;
        return true;
    }
    if (name == "LavaIo") {
        outArchetype = PlanetArchetype::LavaIo;
        return true;
    }
    if (name == "GasGiant") {
        outArchetype = PlanetArchetype::GasGiant;
        return true;
    }
    if (name == "ProtoTheia") {
        outArchetype = PlanetArchetype::ProtoTheia;
        return true;
    }
    return false;
}

bool isValidVisMode(const std::string& mode) {
    return mode == "NaturalBlackbody" ||
           mode == "TemperatureK" ||
           mode == "PressureGPa" ||
           mode == "MaterialLayer" ||
           mode == "DamageFracture" ||
           mode == "Velocity";
}

bool buildCustomScenarioConfig(
    const CustomScenarioDto& dto,
    CollisionScenarioConfig& outConfig,
    std::string& outError) {
    PlanetArchetype targetArch{};
    if (!parseArchetypeName(dto.targetArchetype, targetArch)) {
        outError = "Unknown targetArchetype: " + dto.targetArchetype;
        return false;
    }
    PlanetArchetype impactorArch{};
    if (!parseArchetypeName(dto.impactorArchetype, impactorArch)) {
        outError = "Unknown impactorArchetype: " + dto.impactorArchetype;
        return false;
    }
    if (!std::isfinite(dto.targetMassKg) || dto.targetMassKg <= 0.0) {
        outError = "targetMassKg must be > 0";
        return false;
    }
    if (!std::isfinite(dto.impactorMassKg) || dto.impactorMassKg < 0.0) {
        outError = "impactorMassKg must be >= 0";
        return false;
    }
    if (!std::isfinite(dto.impactVelocityRatioVesc) || dto.impactVelocityRatioVesc <= 0.0) {
        outError = "impactVelocityRatioVesc must be > 0";
        return false;
    }
    if (!std::isfinite(dto.impactParamB) || dto.impactParamB < 0.0) {
        outError = "impactParamB must be >= 0";
        return false;
    }
    if (!std::isfinite(dto.initialSepRadii) || dto.initialSepRadii <= 0.0) {
        outError = "initialSepRadii must be > 0";
        return false;
    }
    if (!std::isfinite(dto.targetSpinRadS)) {
        outError = "targetSpinRadS must be finite";
        return false;
    }
    if (dto.particles < 2u || dto.particles > 2000000u) {
        outError = "particles must be in [2, 2000000]";
        return false;
    }

    CollisionScenarioConfig cfg;
    cfg.presetName = "Custom";
    cfg.target = makeDefaultBodySpec(targetArch);
    cfg.target.massKg = dto.targetMassKg;
    cfg.target.spinOmegaRadS = dto.targetSpinRadS;
    cfg.impactor = makeDefaultBodySpec(impactorArch);
    cfg.impactor.massKg = dto.impactorMassKg;
    cfg.impactVelocityRatioVesc = std::clamp(dto.impactVelocityRatioVesc, 0.1, 20.0);
    cfg.impactParamB = std::clamp(dto.impactParamB, 0.0, 1.2);
    cfg.initialSepRadii = std::clamp(dto.initialSepRadii, 1.5, 20.0);
    cfg.totalParticles = dto.particles;
    cfg.tracerGridRes = 16;
    cfg.randomSeed = 42;
    outConfig = cfg;
    return true;
}

bool parseClientCommand(
    const std::string& jsonText,
    ClientCommand& outCmd,
    std::string& outErrCode,
    std::string& outErrMsg) {
    JsonValue root;
    std::string parseErr;
    if (!parseJson(jsonText, root, parseErr)) {
        outErrCode = "INVALID_JSON";
        outErrMsg = parseErr;
        return false;
    }
    if (!root.isObject()) {
        outErrCode = "INVALID_COMMAND";
        outErrMsg = "Command payload must be a JSON object";
        return false;
    }

    std::string cmdType;
    if (!getRequiredString(root, "type", cmdType, outErrCode, outErrMsg)) {
        return false;
    }

    outCmd = ClientCommand{};

    if (cmdType == "set_preset") {
        outCmd.type = ClientCommandType::SetPreset;
        if (!getRequiredString(root, "preset", outCmd.preset, outErrCode, outErrMsg)) {
            return false;
        }
        try {
            (void)parsePresetName(outCmd.preset);
        } catch (const std::exception& ex) {
            outErrCode = "INVALID_PRESET";
            outErrMsg = ex.what();
            return false;
        }
        return getOptionalPositiveParticles(
            root, "particles", outCmd.hasParticles, outCmd.particles, outErrCode, outErrMsg);
    }

    if (cmdType == "launch_custom") {
        outCmd.type = ClientCommandType::LaunchCustom;
        const JsonValue* cfgVal = root.find("config");
        if (cfgVal == nullptr || !cfgVal->isObject()) {
            outErrCode = "INVALID_PARAMS";
            outErrMsg = "launch_custom requires a 'config' object";
            return false;
        }
        CustomScenarioDto dto;
        if (!getRequiredString(*cfgVal, "targetArchetype", dto.targetArchetype, outErrCode, outErrMsg) ||
            !getRequiredString(*cfgVal, "impactorArchetype", dto.impactorArchetype, outErrCode, outErrMsg)) {
            return false;
        }
        if (!getNumberField(*cfgVal, "targetMassKg", dto.targetMassKg, true, outErrCode, outErrMsg) ||
            !getNumberField(*cfgVal, "impactorMassKg", dto.impactorMassKg, true, outErrCode, outErrMsg) ||
            !getNumberField(*cfgVal, "impactVelocityRatioVesc", dto.impactVelocityRatioVesc, true, outErrCode, outErrMsg) ||
            !getNumberField(*cfgVal, "impactParamB", dto.impactParamB, true, outErrCode, outErrMsg) ||
            !getNumberField(*cfgVal, "initialSepRadii", dto.initialSepRadii, false, outErrCode, outErrMsg) ||
            !getNumberField(*cfgVal, "targetSpinRadS", dto.targetSpinRadS, false, outErrCode, outErrMsg)) {
            return false;
        }
        bool hasParticles = false;
        uint32_t particles = 100000;
        if (!getOptionalPositiveParticles(*cfgVal, "particles", hasParticles, particles, outErrCode, outErrMsg)) {
            return false;
        }
        if (hasParticles) {
            dto.particles = particles;
        }
        CollisionScenarioConfig validated;
        std::string customErr;
        if (!buildCustomScenarioConfig(dto, validated, customErr)) {
            outErrCode = "INVALID_PARAMS";
            outErrMsg = customErr;
            return false;
        }
        outCmd.customConfig = dto;
        return true;
    }

    if (cmdType == "play") {
        outCmd.type = ClientCommandType::Play;
        return true;
    }

    if (cmdType == "pause") {
        outCmd.type = ClientCommandType::Pause;
        return true;
    }

    if (cmdType == "step") {
        outCmd.type = ClientCommandType::Step;
        double count = 1.0;
        if (!getNumberField(root, "count", count, false, outErrCode, outErrMsg)) {
            return false;
        }
        if (count < 1.0 || count > 1000.0) {
            outErrCode = "INVALID_PARAMS";
            outErrMsg = "step count must be in [1, 1000]";
            return false;
        }
        outCmd.stepCount = static_cast<uint32_t>(std::llround(count));
        return true;
    }

    if (cmdType == "set_timescale") {
        outCmd.type = ClientCommandType::SetTimescale;
        double scale = 1.0;
        if (!getNumberField(root, "scale", scale, true, outErrCode, outErrMsg)) {
            return false;
        }
        outCmd.timeScale = std::clamp(scale, 0.1, 100.0);
        return true;
    }

    if (cmdType == "set_slice_plane") {
        outCmd.type = ClientCommandType::SetSlicePlane;
        const JsonValue* enabledVal = root.find("enabled");
        if (enabledVal == nullptr || !enabledVal->isBool()) {
            outErrCode = "INVALID_PARAMS";
            outErrMsg = "set_slice_plane requires boolean 'enabled'";
            return false;
        }
        outCmd.slicePlane.enabled = enabledVal->boolVal;

        const JsonValue* normalVal = root.find("normal");
        if (normalVal != nullptr) {
            if (!normalVal->isArray() || normalVal->arrVal.size() != 3u ||
                !normalVal->arrVal[0].isNumber() || !normalVal->arrVal[1].isNumber() || !normalVal->arrVal[2].isNumber()) {
                outErrCode = "INVALID_PARAMS";
                outErrMsg = "slice normal must be a 3-element numeric array";
                return false;
            }
            outCmd.slicePlane.normalX = static_cast<float>(normalVal->arrVal[0].numVal);
            outCmd.slicePlane.normalY = static_cast<float>(normalVal->arrVal[1].numVal);
            outCmd.slicePlane.normalZ = static_cast<float>(normalVal->arrVal[2].numVal);
        }

        double offsetKm = 0.0;
        if (!getNumberField(root, "offsetKm", offsetKm, false, outErrCode, outErrMsg)) {
            return false;
        }
        outCmd.slicePlane.offsetKm = static_cast<float>(offsetKm);
        return true;
    }

    if (cmdType == "set_vis_mode") {
        outCmd.type = ClientCommandType::SetVisMode;
        if (!getRequiredString(root, "mode", outCmd.visMode, outErrCode, outErrMsg)) {
            return false;
        }
        if (!isValidVisMode(outCmd.visMode)) {
            outErrCode = "INVALID_VIS_MODE";
            outErrMsg = "Unknown visualization mode: " + outCmd.visMode;
            return false;
        }
        return true;
    }

    if (cmdType == "request_export") {
        outCmd.type = ClientCommandType::RequestExport;
        if (!getRequiredString(root, "format", outCmd.exportFormat, outErrCode, outErrMsg)) {
            return false;
        }
        if (outCmd.exportFormat != "csv" && outCmd.exportFormat != "json") {
            outErrCode = "INVALID_PARAMS";
            outErrMsg = "Export format must be 'csv' or 'json'";
            return false;
        }
        return true;
    }

    if (cmdType == "ping") {
        outCmd.type = ClientCommandType::Ping;
        double ts = 0.0;
        if (!getNumberField(root, "ts", ts, false, outErrCode, outErrMsg)) {
            return false;
        }
        outCmd.pingTs = ts;
        return true;
    }

    outErrCode = "UNKNOWN_COMMAND";
    outErrMsg = "Unknown command type: " + cmdType;
    return false;
}

std::string serializeTelemetryDto(
    const TelemetrySnapshot& snap,
    bool isPlaying,
    const std::string& presetName) {
    std::ostringstream os;
    os << "{\"type\":\"telemetry\""
       << ",\"frameSeq\":" << snap.frameSeq
       << ",\"simTimeSec\":" << formatFiniteJsonNumber(snap.simTimeSec)
       << ",\"dtSec\":" << formatFiniteJsonNumber(static_cast<double>(snap.dtSec))
       << ",\"state\":\"" << (isPlaying ? "playing" : "paused") << "\""
       << ",\"preset\":\"" << jsonEscape(presetName) << "\""
       << ",\"outcome\":\"" << outcomeToString(snap.outcome) << "\""
       << ",\"kineticEnergyJ\":" << formatFiniteJsonNumber(snap.kineticEnergyJ)
       << ",\"potentialEnergyJ\":" << formatFiniteJsonNumber(snap.potentialEnergyJ)
       << ",\"internalEnergyJ\":" << formatFiniteJsonNumber(snap.internalEnergyJ)
       << ",\"radiatedEnergyJ\":" << formatFiniteJsonNumber(snap.radiatedEnergyJ)
       << ",\"totalEnergyJ\":" << formatFiniteJsonNumber(snap.totalEnergyJ)
       << ",\"energyDriftRel\":" << formatFiniteJsonNumber(snap.energyDriftRel)
       << ",\"momentumKgMS\":["
       << formatFiniteJsonNumber(snap.momentumKgMS[0]) << ","
       << formatFiniteJsonNumber(snap.momentumKgMS[1]) << ","
       << formatFiniteJsonNumber(snap.momentumKgMS[2]) << "]"
       << ",\"angularMomentumKgM2S\":["
       << formatFiniteJsonNumber(snap.angularMomentumKgM2S[0]) << ","
       << formatFiniteJsonNumber(snap.angularMomentumKgM2S[1]) << ","
       << formatFiniteJsonNumber(snap.angularMomentumKgM2S[2]) << "]"
       << ",\"virialRatio\":" << formatFiniteJsonNumber(snap.virialRatio)
       << ",\"maxTempK\":" << formatFiniteJsonNumber(snap.maxTempK)
       << ",\"maxPressureGPa\":" << formatFiniteJsonNumber(snap.maxPressureGPa)
       << ",\"boundMassFrac\":" << formatFiniteJsonNumber(snap.boundMassFrac)
       << ",\"diskMassFrac\":" << formatFiniteJsonNumber(snap.diskMassFrac)
       << ",\"ejectaMassFrac\":" << formatFiniteJsonNumber(snap.ejectaMassFrac)
       << ",\"qRatioQrd\":" << formatFiniteJsonNumber(snap.qRatioQrd)
       << ",\"gpuStepMs\":" << formatFiniteJsonNumber(snap.gpuStepMs)
       << ",\"livingBodyComKm\":["
       << formatFiniteJsonNumber(snap.livingBodyComKm[0]) << ","
       << formatFiniteJsonNumber(snap.livingBodyComKm[1]) << ","
       << formatFiniteJsonNumber(snap.livingBodyComKm[2]) << "]"
       << ",\"livingBodyRadiusKm\":" << formatFiniteJsonNumber(snap.livingBodyRadiusKm)
       << ",\"livingBodySpinRadS\":" << formatFiniteJsonNumber(snap.livingBodySpinRadS)
       << ",\"impactorComKm\":["
       << formatFiniteJsonNumber(snap.impactorComKm[0]) << ","
       << formatFiniteJsonNumber(snap.impactorComKm[1]) << ","
       << formatFiniteJsonNumber(snap.impactorComKm[2]) << "]"
       << ",\"tracerGridRes\":" << snap.tracerGridRes
       << ",\"livingBodyPresent\":" << (snap.livingBodyPresent ? "true" : "false")
       << "}";
    return os.str();
}

std::string serializePresetLoaded(
    const std::string& presetName,
    uint32_t particleCount,
    uint32_t tracerCount,
    uint32_t tracerGridRes,
    bool livingBodyPresent) {
    std::ostringstream os;
    os << "{\"type\":\"preset_loaded\""
       << ",\"preset\":\"" << jsonEscape(presetName) << "\""
       << ",\"particleCount\":" << particleCount
       << ",\"tracerCount\":" << tracerCount
       << ",\"tracerGridRes\":" << tracerGridRes
       << ",\"livingBodyPresent\":" << (livingBodyPresent ? "true" : "false")
       << "}";
    return os.str();
}

std::string serializeExportData(
    const std::string& format,
    const std::string& filename,
    const std::string& content) {
    std::ostringstream os;
    os << "{\"type\":\"export_data\""
       << ",\"format\":\"" << jsonEscape(format) << "\""
       << ",\"filename\":\"" << jsonEscape(filename) << "\""
       << ",\"content\":\"" << jsonEscape(content) << "\""
       << "}";
    return os.str();
}

std::string serializePong(double ts) {
    return "{\"type\":\"pong\",\"ts\":" + formatFiniteJsonNumber(ts) + "}";
}

std::string serializeError(const std::string& code, const std::string& message) {
    return "{\"type\":\"error\",\"code\":\"" + jsonEscape(code) +
           "\",\"message\":\"" + jsonEscape(message) + "\"}";
}

} // namespace astraea::server
