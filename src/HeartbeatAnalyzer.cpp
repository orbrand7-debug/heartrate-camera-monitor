#include "HeartbeatAnalyzer.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

HeartbeatAnalyzer::HeartbeatAnalyzer(size_t window_size, double fps)
    : m_window_size(window_size), m_fps(fps) {}

void HeartbeatAnalyzer::add_sample(double hue) {
    m_hue_buffer.push_back(hue);
    if (m_hue_buffer.size() > m_window_size) {
        m_hue_buffer.pop_front();
    }
}

[[nodiscard]] std::expected<double, std::string> HeartbeatAnalyzer::calculate_bpm(double min_b, double max_b) const {
    if (m_hue_buffer.size() < m_window_size) {
        return std::unexpected("Buffering...");
    }

    std::vector<float> sig(m_hue_buffer.begin(), m_hue_buffer.end());
    float m = std::accumulate(sig.begin(), sig.end(), 0.0f) / sig.size();
    for (size_t idx = 0; idx < sig.size(); ++idx) {
        sig[idx] = (sig[idx] - m) * (0.54f - 0.46f * cosf(2.0f * (float)CV_PI * idx / (m_window_size - 1)));
    }

    std::vector<cv::Mat> planes = { cv::Mat_<float>(sig), cv::Mat::zeros((int)m_window_size, 1, CV_32F) };
    cv::Mat complex;
    cv::merge(planes, complex);
    cv::dft(complex, complex);
    cv::split(complex, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);

    int low = (int)(min_b * m_window_size / (60.0 * m_fps));
    int high = (int)(max_b * m_window_size / (60.0 * m_fps));
    int peak = -1; float max_v = -1.0f;
    for (int i = low; i <= high && i < (int)m_window_size / 2; ++i) {
        if (planes[0].at<float>(i) > max_v) { max_v = planes[0].at<float>(i); peak = i; }
    }
    if (peak <= 0) {
        return std::unexpected("No peak"); 
    }
    
    double bpm = (peak * m_fps / m_window_size) * 60.0;
    return bpm;
}