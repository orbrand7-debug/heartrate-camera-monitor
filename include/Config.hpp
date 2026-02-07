#pragma once
#include <string>
#include <vector>
#include <expected>
#include <opencv2/core.hpp>

/**
 * @struct AppConfig
 * @brief Thread-safe configuration container loaded from YAML.
 */
struct AppConfig {
    struct {
        double fps;
        double acquisition_fps;
        cv::Rect frame_roi;
    } camera;

    struct {
        double window_duration_seconds;
        double min_bpm;
        double max_bpm;
    } analysis;

    struct {
        int x, y, width, height;
        uint8_t alpha;
        std::string font_name;
        int font_size;
        int r, g, b;
        std::string hotkey_toggle_debug;
    } hud;

    /**
     * @brief Parses config.yaml into the struct.
     * @return std::expected containing config or error string.
     */
    static std::expected<AppConfig, std::string> load(const std::string& path);
};
