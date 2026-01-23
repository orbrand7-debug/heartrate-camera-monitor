#ifndef HEARTBEAT_ANALYZER_HPP
#define HEARTBEAT_ANALYZER_HPP

#include <vector>
#include <deque>
#include <string>
#include <expected>

/**
 * @class HeartbeatAnalyzer
 * @brief Handles the temporal signal processing of hue values to extract heart rate.
 */
class HeartbeatAnalyzer {
public:
    /**
     * @brief Constructor for the analyzer.
     * @param window_size Number of samples to collect before performing FFT.
     * @param fps The frames per second of the source video.
     */
    explicit HeartbeatAnalyzer(size_t window_size = 256, double fps = 30.0);

    /**
     * @brief Adds a new hue measurement to the sliding window.
     * @param hue The average hue value calculated from the forehead ROI.
     */
    void add_sample(double hue);

    /**
     * @brief Analyzes the stored signal using a Discrete Fourier Transform.
     * @param min_b Minimum BPM to consider in the analysis.
     * @param max_b Maximum BPM to consider in the analysis.
     * @return std::expected containing the BPM on success, or an error string.
     */
    [[nodiscard]] std::expected<double, std::string> calculate_bpm(double min_b, double max_b) const;

private:
    std::deque<double> m_hue_buffer;
    size_t m_window_size;
    double m_fps;
};

#endif