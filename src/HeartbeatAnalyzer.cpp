#include "HeartbeatAnalyzer.hpp"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <cmath>

HeartbeatAnalyzer::HeartbeatAnalyzer(int window_size, double fps) 
    : m_ws(window_size), m_fps(fps) {}

void HeartbeatAnalyzer::add_sample(const cv::Scalar& bgr) {
    m_buffer.push_back(bgr);
    if (m_buffer.size() > m_ws) m_buffer.pop_front();
}

std::expected<double, std::string> HeartbeatAnalyzer::calculate_bpm(double min_b, double max_b) {
    if (m_buffer.size() < m_ws) return std::unexpected("Buffering...");

    // 1. Extract R, G, B channels
    std::vector<double> R, G, B;
    for (const auto& s : m_buffer) {
        B.push_back(s[0]); G.push_back(s[1]); R.push_back(s[2]);
    }

    // 2. Temporal Normalization (Mean centering)
    auto normalize = [](std::vector<double>& vec) {
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        double mean = sum / vec.size();
        for (auto& v : vec) v /= (mean + 1e-6); // Avoid division by zero
    };
    normalize(R); normalize(G); normalize(B);

    // 3. POS Projections
    // S1 = G - B
    // S2 = G + B - 2R
    std::vector<double> S1(m_ws), S2(m_ws);
    for (size_t i = 0; i < m_ws; ++i) {
        S1[i] = G[i] - B[i];
        S2[i] = G[i] + B[i] - 2.0 * R[i];
    }

    // 4. Calculate Alpha (Ratio of standard deviations)
    auto get_std = [](const std::vector<double>& v) {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
        return std::sqrt(sq_sum / v.size() - mean * mean);
    };
    double alpha = get_std(S1) / (get_std(S2) + 1e-6);

    // 5. Final POS Signal: H = S1 + alpha * S2
    std::vector<float> H(m_ws);
    for (size_t i = 0; i < m_ws; ++i) {
        H[i] = static_cast<float>(S1[i] + alpha * S2[i]);
    }

    // 6. Apply Hamming Window to POS signal
    for (size_t i = 0; i < m_ws; ++i) {
        H[i] *= 0.54f - 0.46f * cosf(2.0f * (float)CV_PI * i / (m_ws - 1));
    }

    // 7. FFT Analysis (Same as before)
    cv::Mat planes[] = { cv::Mat_<float>(H), cv::Mat::zeros((int)m_ws, 1, CV_32F) }, complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);
    cv::split(complex, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);

    // 8. Peak detection in human heart range
    int low = (int)(min_b * m_ws / (60.0 * m_fps));
    int high = (int)(max_b * m_ws / (60.0 * m_fps));
    int peak = -1; float max_v = -1.0f;

    for (int i = low; i <= high && i < (int)m_ws / 2; ++i) {
        if (planes[0].at<float>(i) > max_v) {
            max_v = planes[0].at<float>(i);
            peak = i;
        }
    }

    if (peak <= 0) return std::unexpected("Noise floor too high");
    return (peak * m_fps / m_ws) * 60.0;
}