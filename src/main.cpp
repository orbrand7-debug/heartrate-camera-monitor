#include <print>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace {
struct RunningStats {
    size_t count{0};
    double mean{0.0};
    double m2{0.0};
    double min{0.0};
    double max{0.0};

    void add(double x) {
        if (count == 0) {
            min = max = x;
        } else {
            min = std::min(min, x);
            max = std::max(max, x);
        }
        ++count;
        const double delta = x - mean;
        mean += delta / static_cast<double>(count);
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double variance() const {
        return (count > 1) ? (m2 / static_cast<double>(count - 1)) : 0.0;
    }
};
} // namespace
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "FaceProcessor.hpp"
#include "HeartbeatAnalyzer.hpp"
#include "Overlay.hpp"


namespace {
cv::Mat resize_plot_to_fit(const cv::Mat& plot, int max_w, int max_h) {
    if (plot.empty() || max_w <= 0 || max_h <= 0) {
        return cv::Mat();
    }
    double scale = std::min(static_cast<double>(max_w) / plot.cols,
                            static_cast<double>(max_h) / plot.rows);
    scale = std::clamp(scale, 0.1, 1.0);
    const int w = std::max(2, static_cast<int>(std::lround(plot.cols * scale)));
    const int h = std::max(2, static_cast<int>(std::lround(plot.rows * scale)));
    cv::Mat resized;
    cv::resize(plot, resized, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    return resized;
}

void blit_plot(cv::Mat& frame, const cv::Mat& plot, const cv::Point& origin, const char* label) {
    if (frame.empty() || plot.empty()) {
        return;
    }
    const int x = std::clamp(origin.x, 0, frame.cols - 1);
    const int y = std::clamp(origin.y, 0, frame.rows - 1);
    const int w = std::min(plot.cols, frame.cols - x);
    const int h = std::min(plot.rows, frame.rows - y);
    if (w < 2 || h < 2) {
        return;
    }
    cv::Rect roi(x, y, w, h);
    plot(cv::Rect(0, 0, w, h)).copyTo(frame(roi));
    cv::rectangle(frame, roi, cv::Scalar(0, 255, 255), 1);
    if (label) {
        cv::putText(frame, label, cv::Point(x + 4, y + 16), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
}
} // namespace

int main() {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    spdlog::info("Starting HeartbeatMonitor...");

    auto app_start = std::chrono::steady_clock::now();
    auto config_res = AppConfig::load("config.yaml");
    if (!config_res) {
        spdlog::error("Config Error: {}", config_res.error());
        std::println(stderr, "Config Error: {}", config_res.error());
        return -1;
    }
    const auto config = *config_res;
    spdlog::info("Config loaded in {:.1f} ms", std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - app_start).count());
    spdlog::info("Camera fps={}, acquisition_fps={}, window_duration_seconds={}",
        config.camera.fps, config.camera.acquisition_fps, config.analysis.window_duration_seconds);

    try {
        auto cam_start = std::chrono::steady_clock::now();
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::println(stderr, "Error: Could not open camera.");
            return -1;
        }
        cap.set(cv::CAP_PROP_FPS, config.camera.fps);
        spdlog::info("Camera opened in {:.1f} ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cam_start).count());
        spdlog::info("Camera props: {}x{} @ {:.1f} fps",
            cap.get(cv::CAP_PROP_FRAME_WIDTH),
            cap.get(cv::CAP_PROP_FRAME_HEIGHT),
            cap.get(cv::CAP_PROP_FPS));

        auto model_start = std::chrono::steady_clock::now();
        FaceProcessor processor(MODEL_PATH);
        spdlog::info("Dlib model loaded in {:.1f} ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - model_start).count());
        const double window_seconds = std::max(1.0, config.analysis.window_duration_seconds);
        const int window_size = std::max(
            2, static_cast<int>(std::lround(window_seconds * config.camera.acquisition_fps)));
        HeartbeatAnalyzer analyzer(window_size, config.camera.acquisition_fps);
        spdlog::info("Analysis window: {} samples (~{:.2f}s)", window_size,
            window_size / config.camera.acquisition_fps);

        auto hud_start = std::chrono::steady_clock::now();
        Overlay hud(config); // Pass config to HUD
        spdlog::info("HUD created in {:.1f} ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - hud_start).count());

        std::jthread hud_thread([&hud]() { hud.run(); });
        spdlog::info("HUD thread started");

        cv::Mat frame;
        const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / config.camera.acquisition_fps));
        auto last_buffer_log = std::chrono::steady_clock::now();
        auto last_stats_log = std::chrono::steady_clock::now();
        RunningStats sample_dt_stats;
        bool has_last_sample = false;
        std::chrono::steady_clock::time_point last_sample_time;
        size_t frame_count = 0;
        size_t face_found_count = 0;
        bool buffer_ready_logged = false;
        bool last_debug_mode = false;
        while (true) {
            auto frame_start = std::chrono::steady_clock::now();
            if (!cap.read(frame)) {
                break;
            }
            auto read_end = std::chrono::steady_clock::now();
            ++frame_count;

            bool debug_mode = hud.is_debug_mode();
            if (debug_mode != last_debug_mode) {
                spdlog::info("Debug mode {}", debug_mode ? "ON" : "OFF");
                spdlog::set_level(debug_mode ? spdlog::level::debug : spdlog::level::info);
                last_debug_mode = debug_mode;
            }
            
            // Apply frame ROI if defined
            cv::Mat processing_frame = frame;
            if (config.camera.frame_roi.area() > 0) {
                processing_frame = frame(config.camera.frame_roi & cv::Rect(0,0,frame.cols,frame.rows));
            }

            FaceTimings face_timings;
            auto face_start = std::chrono::steady_clock::now();
            auto face_res = processor.get_central_face(processing_frame, debug_mode ? &face_timings : nullptr);
            auto face_end = std::chrono::steady_clock::now();
            auto forehead_end = face_end;
            auto sample_end = face_end;
            auto bpm_end = face_end;
            auto plots_end = face_end;
            auto overlay_end = face_end;
            if (face_res) {
                ++face_found_count;
                cv::Mat forehead;
                if (debug_mode) {
                    cv::Mat forehead_rect;
                    forehead = processor.get_stabilized_forehead(processing_frame, *face_res, &forehead_rect);
                    processor.draw_debug(processing_frame, *face_res, forehead_rect);
                }
                else {
                    forehead = processor.get_stabilized_forehead(processing_frame, *face_res);
                }
                forehead_end = std::chrono::steady_clock::now();
                analyzer.add_sample(processor.get_avg_bgr(forehead));
                if (debug_mode) {
                    auto now = std::chrono::steady_clock::now();
                    if (has_last_sample) {
                        const double dt_ms = std::chrono::duration<double, std::milli>(now - last_sample_time).count();
                        sample_dt_stats.add(dt_ms);
                    }
                    last_sample_time = now;
                    has_last_sample = true;
                }
                sample_end = std::chrono::steady_clock::now();
                auto bpm = analyzer.calculate_bpm(config.analysis.min_bpm, config.analysis.max_bpm, debug_mode);
                bpm_end = std::chrono::steady_clock::now();
                if (bpm) {
                    hud.update_bpm(*bpm);
                }
            }

            if (debug_mode && analyzer.has_debug_plots()) {
                const int margin = 10;
                const int max_w = std::min(360, std::max(160, processing_frame.cols / 2));
                const int max_h = std::min(180, std::max(120, (processing_frame.rows - 3 * margin) / 2));
                cv::Mat plot_input = resize_plot_to_fit(analyzer.debug_fft_input(), max_w, max_h);
                cv::Mat plot_fft = resize_plot_to_fit(analyzer.debug_fft_magnitude(), max_w, max_h);

                int x = processing_frame.cols - plot_input.cols - margin;
                int y = margin;
                blit_plot(processing_frame, plot_input, cv::Point(x, y), "FFT Input");

                y += plot_input.rows + margin;
                if (!plot_fft.empty()) {
                    x = processing_frame.cols - plot_fft.cols - margin;
                    blit_plot(processing_frame, plot_fft, cv::Point(x, y), "FFT Mag");
                }
                plots_end = std::chrono::steady_clock::now();
            }

            hud.update_frame(processing_frame);
            overlay_end = std::chrono::steady_clock::now();
            if (cv::waitKey(1) == 27) {
                break;
            }

            auto elapsed = std::chrono::steady_clock::now() - frame_start;
            if (debug_mode) {
                const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
                spdlog::debug("Timing ms: read {:.2f}, face {:.2f} (detect {:.2f}, select {:.2f}, predict {:.2f}), forehead {:.2f}, sample {:.2f}, bpm {:.2f}, plots {:.2f}, overlay {:.2f}, total {:.2f}",
                    ms(read_end - frame_start),
                    ms(face_end - face_start),
                    face_timings.detect_ms,
                    face_timings.select_ms,
                    face_timings.predict_ms,
                    ms(forehead_end - face_end),
                    ms(sample_end - forehead_end),
                    ms(bpm_end - sample_end),
                    ms(plots_end - bpm_end),
                    ms(overlay_end - plots_end),
                    ms(elapsed));
                auto now = std::chrono::steady_clock::now();
                if (now - last_stats_log > std::chrono::seconds(2) && sample_dt_stats.count > 1) {
                    const double target_dt_ms = 1000.0 / config.camera.acquisition_fps;
                    const double std_ms = std::sqrt(sample_dt_stats.variance());
                    const double est_fps = 1000.0 / sample_dt_stats.mean;
                    const double max_jitter = sample_dt_stats.max - target_dt_ms;
                    const double min_jitter = sample_dt_stats.min - target_dt_ms;
                    const double face_ratio = frame_count > 0
                        ? (100.0 * static_cast<double>(face_found_count) / static_cast<double>(frame_count))
                        : 0.0;
                    spdlog::debug("Sample dt: mean {:.2f} ms (std {:.2f}), min {:.2f}, max {:.2f}, est {:.2f} fps, jitter [min {:.2f}, max {:.2f}] ms, faces {:.0f}% ({}/{})",
                        sample_dt_stats.mean, std_ms, sample_dt_stats.min, sample_dt_stats.max,
                        est_fps, min_jitter, max_jitter, face_ratio, face_found_count, frame_count);
                    last_stats_log = now;
                    sample_dt_stats = RunningStats{};
                    frame_count = 0;
                    face_found_count = 0;
                }
            }
            if (elapsed > interval * 2) {
                spdlog::warn("Frame processing overrun: {:.1f} ms (interval {:.1f} ms)",
                    std::chrono::duration<double, std::milli>(elapsed).count(),
                    std::chrono::duration<double, std::milli>(interval).count());
            } else if (spdlog::should_log(spdlog::level::debug)) {
                spdlog::debug("Frame processing time: {:.1f} ms",
                    std::chrono::duration<double, std::milli>(elapsed).count());
            }
            if (!buffer_ready_logged && analyzer.buffer_size() >= analyzer.window_size()) {
                spdlog::info("Buffer filled: {} samples", analyzer.window_size());
                buffer_ready_logged = true;
            } else if (!buffer_ready_logged) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_buffer_log > std::chrono::seconds(2)) {
                    const double pct = 100.0 * analyzer.buffer_size() /
                        std::max<size_t>(1, analyzer.window_size());
                    spdlog::info("Buffering: {}/{} ({:.0f}%)",
                        analyzer.buffer_size(), analyzer.window_size(), pct);
                    last_buffer_log = now;
                }
            }
            if (elapsed < interval) {
                std::this_thread::sleep_for(interval - elapsed);
            }
        }
        hud.stop();
    } catch (const std::exception& e) {
        std::println(stderr, "Fatal: {}", e.what());
    }
    return 0;
}
