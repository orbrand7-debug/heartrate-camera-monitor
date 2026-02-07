#include "Config.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <algorithm>

std::expected<AppConfig, std::string> AppConfig::load(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        return std::unexpected("Config missing: " + path);
    }
    try {
        YAML::Node node = YAML::LoadFile(path);
        AppConfig c;
        c.camera.fps = node["camera"]["fps"].as<double>(30.0);
        c.camera.acquisition_fps = node["camera"]["acquisition_fps"].as<double>(10.0);
        c.camera.acquisition_fps = std::clamp(c.camera.acquisition_fps, 10.0, 60.0);
        auto roi = node["camera"]["frame_roi"].as<std::vector<int>>();
        c.camera.frame_roi = cv::Rect(roi[0], roi[1], roi[2], roi[3]);

        if (node["analysis"] && node["analysis"]["window_duration_seconds"]) {
            c.analysis.window_duration_seconds = node["analysis"]["window_duration_seconds"].as<double>(8.5);
        } else if (node["analysis"] && node["analysis"]["window_size"]) {
            const int legacy_ws = node["analysis"]["window_size"].as<int>(256);
            c.analysis.window_duration_seconds = legacy_ws / std::max(1.0, c.camera.acquisition_fps);
        } else {
            c.analysis.window_duration_seconds = 8.5;
        }
        c.analysis.min_bpm = node["analysis"]["min_bpm"].as<double>(45.0);
        c.analysis.max_bpm = node["analysis"]["max_bpm"].as<double>(180.0);

        c.hud.x = node["hud"]["x"].as<int>();
        c.hud.y = node["hud"]["y"].as<int>();
        c.hud.width = node["hud"]["width"].as<int>();
        c.hud.height = node["hud"]["height"].as<int>();
        c.hud.alpha = (uint8_t)node["hud"]["alpha"].as<int>(255);
        c.hud.font_name = node["hud"]["font_name"].as<std::string>("Arial");
        c.hud.font_size = node["hud"]["font_size"].as<int>(40);
        c.hud.hotkey_toggle_debug = node["hud"]["hotkey_toggle_debug"].as<std::string>("Ctrl+Alt+D");
        std::transform(c.hud.hotkey_toggle_debug.begin(), c.hud.hotkey_toggle_debug.end(), c.hud.hotkey_toggle_debug.begin(),
                                    [](unsigned char c){ return std::toupper(c); }
                                );

        auto col = node["hud"]["color"].as<std::vector<int>>();
        c.hud.r = col[0]; c.hud.g = col[1]; c.hud.b = col[2];
        return c;
    } catch (const std::exception& e) {
        return std::unexpected(e.what());
    }
}
