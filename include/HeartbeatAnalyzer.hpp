#pragma once
#include <deque>
#include <vector>
#include <expected>
#include <string>
#include <opencv2/core.hpp>

/**
 * @class HeartbeatAnalyzer
 * @brief Implements the POS (Plane-Orthogonal-to-Skin) algorithm for rPPG.
 */
class HeartbeatAnalyzer {
public:
    /**
     * @param window_size Number of frames to analyze (e.g., 256).
     * @param fps Camera frame rate.
     */
    HeartbeatAnalyzer(int window_size, double fps);

    /**
     * @brief Adds BGR averages from the ROI to the temporal buffer.
     */
    void add_sample(const cv::Scalar& bgr);

    /**
     * @brief Processes the BGR buffer using the POS algorithm and FFT.
     * @return std::expected containing the BPM or an error message.
     */
    std::expected<double, std::string> calculate_bpm(double min_b, double max_b);

private:
    std::deque<cv::Scalar> m_buffer;
    size_t m_ws;
    double m_fps;
};