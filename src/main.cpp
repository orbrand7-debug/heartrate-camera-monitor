#include <print>
#include <thread>
#include <opencv2/highgui.hpp>
#include "FaceProcessor.hpp"
#include "HeartbeatAnalyzer.hpp"
#include "Overlay.hpp"


int main() {
    auto config_res = AppConfig::load("config.yaml");
    if (!config_res) {
        std::println(stderr, "Config Error: {}", config_res.error());
        return -1;
    }
    const auto config = *config_res;

    try {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::println(stderr, "Error: Could not open camera.");
            return -1;
        }

        FaceProcessor processor(MODEL_PATH);
        HeartbeatAnalyzer analyzer(config.analysis.window_size, config.camera.fps);
        Overlay hud(config); // Pass config to HUD

        std::jthread hud_thread([&hud]() { hud.run(); });

        cv::Mat frame;
        while (cap.read(frame)) {
            bool debug_mode = hud.is_debug_mode();
            
            // Apply frame ROI if defined
            cv::Mat processing_frame = frame;
            if (config.camera.frame_roi.area() > 0) {
                processing_frame = frame(config.camera.frame_roi & cv::Rect(0,0,frame.cols,frame.rows));
            }

            auto face_res = processor.get_central_face(processing_frame);
            if (face_res) {
                cv::Mat forehead;
                if (debug_mode) {
                    cv::Mat forehead_rect;
                    forehead = processor.get_stabilized_forehead(processing_frame, *face_res, &forehead_rect);
                    processor.draw_debug(processing_frame, *face_res, forehead_rect);
                }
                else {
                    forehead = processor.get_stabilized_forehead(processing_frame, *face_res);
                }
                analyzer.add_sample(processor.get_avg_bgr(forehead));
                auto bpm = analyzer.calculate_bpm(config.analysis.min_bpm, config.analysis.max_bpm);
                if (bpm) {
                    hud.update_bpm(*bpm);
                }
            }
            hud.update_frame(processing_frame);
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
        hud.stop();
    } catch (const std::exception& e) {
        std::println(stderr, "Fatal: {}", e.what());
    }
    return 0;
}