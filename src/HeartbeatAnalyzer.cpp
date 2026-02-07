#include "HeartbeatAnalyzer.hpp"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <array>
#include <spdlog/spdlog.h>

namespace {
cv::Mat plot_signal(const std::vector<float>& data, int width, int height) {
    if (data.size() < 2) {
        return cv::Mat();
    }
    float min_v = *std::min_element(data.begin(), data.end());
    float max_v = *std::max_element(data.begin(), data.end());
    if (std::fabs(max_v - min_v) < 1e-6f) {
        max_v = min_v + 1.0f;
    }

    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    auto to_y = [&](float v) {
        float t = (v - min_v) / (max_v - min_v);
        return static_cast<int>((1.0f - t) * (height - 1));
    };

    for (size_t i = 1; i < data.size(); ++i) {
        int x0 = static_cast<int>((i - 1) * (width - 1) / (data.size() - 1));
        int x1 = static_cast<int>(i * (width - 1) / (data.size() - 1));
        int y0 = to_y(data[i - 1]);
        int y1 = to_y(data[i]);
        cv::line(plot, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    return plot;
}
} // namespace

HeartbeatAnalyzer::HeartbeatAnalyzer(int window_size, double fps) 
    : m_ws(window_size), m_fps(fps) {}

void HeartbeatAnalyzer::add_sample(const cv::Scalar& bgr) {
    m_buffer.push_back(bgr);
    if (m_buffer.size() > m_ws) m_buffer.pop_front();
}

std::expected<double, std::string> HeartbeatAnalyzer::calculate_bpm(double min_b, double max_b, bool debug_plot) {
    if (m_buffer.size() < m_ws) return std::unexpected("Buffering...");

    // 1. Extract R, G, B channels
    std::vector<double> R, G, B;
    for (const auto& s : m_buffer) {
        B.push_back(s[0]); G.push_back(s[1]); R.push_back(s[2]);
    }

    // 2. Temporal Normalization (Mean centering)
    auto normalize = [](std::vector<double>& vec) {
        const double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        const double mean = sum / static_cast<double>(vec.size());
        for (auto& v : vec) {
            v = (v / (mean + 1e-6)) - 1.0; // Normalize and remove DC
        }
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
    const float h_mean = std::accumulate(H.begin(), H.end(), 0.0f) / static_cast<float>(H.size());
    for (auto& v : H) {
        v -= h_mean;
    }

    // 6. Apply Hamming Window to POS signal
    for (size_t i = 0; i < m_ws; ++i) {
        H[i] *= 0.54f - 0.46f * cosf(2.0f * (float)CV_PI * i / (m_ws - 1));
    }

    if (debug_plot) {
        m_debug_fft_input = plot_signal(H, 320, 160);
    } else {
        m_debug_fft_input.release();
        m_debug_fft_magnitude.release();
    }

    // 7. FFT Analysis
    cv::Mat planes[] = { cv::Mat_<float>(H), cv::Mat::zeros((int)m_ws, 1, CV_32F) }, complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);
    cv::split(complex, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);

    if (debug_plot) {
        std::vector<float> mag;
        mag.reserve(m_ws / 2);
        for (int i = 0; i < (int)m_ws / 2; ++i) {
            mag.push_back(planes[0].at<float>(i));
        }
        m_debug_fft_magnitude = plot_signal(mag, 320, 160);
    }

    // 8. Peak detection in human heart range
    double min_hz = min_b / 60.0;
    double max_hz = max_b / 60.0;
    double nyquist = m_fps / 2.0;
    min_hz = std::clamp(min_hz, 0.0, nyquist);
    max_hz = std::clamp(max_hz, min_hz, nyquist);

    int low = static_cast<int>(std::floor(min_hz * m_ws / m_fps));
    int high = static_cast<int>(std::ceil(max_hz * m_ws / m_fps));
    int max_bin = static_cast<int>(m_ws / 2) - 1;
    low = std::clamp(low, 1, max_bin);
    high = std::clamp(high, low, max_bin);
    int peak = -1; float max_v = -1.0f;

    for (int i = low; i <= high && i < (int)m_ws / 2; ++i) {
        if (planes[0].at<float>(i) > max_v) {
            max_v = planes[0].at<float>(i);
            peak = i;
        }
    }

    if (debug_plot) {
        struct Peak { int idx; float mag; };
        std::array<Peak, 3> top{{{-1, -1.0f}, {-1, -1.0f}, {-1, -1.0f}}};
        for (int i = low; i <= high && i < (int)m_ws / 2; ++i) {
            float v = planes[0].at<float>(i);
            for (size_t k = 0; k < top.size(); ++k) {
                if (v > top[k].mag) {
                    for (size_t s = top.size() - 1; s > k; --s) {
                        top[s] = top[s - 1];
                    }
                    top[k] = {i, v};
                    break;
                }
            }
        }
        if (top[0].idx > 0) {
            const double hz0 = top[0].idx * m_fps / m_ws;
            const double bpm0 = hz0 * 60.0;
            const double hz1 = top[1].idx > 0 ? top[1].idx * m_fps / m_ws : 0.0;
            const double bpm1 = hz1 * 60.0;
            const double hz2 = top[2].idx > 0 ? top[2].idx * m_fps / m_ws : 0.0;
            const double bpm2 = hz2 * 60.0;
            const double ratio = (top[1].mag > 0.0f) ? (top[0].mag / top[1].mag) : 0.0;
            const double ratio_db = (ratio > 0.0) ? (20.0 * std::log10(ratio)) : 0.0;
            spdlog::debug("FFT peaks: #1 {:.2f} bpm (mag {:.3f}), #2 {:.2f} bpm (mag {:.3f}), #3 {:.2f} bpm (mag {:.3f})",
                bpm0, top[0].mag, bpm1, top[1].mag, bpm2, top[2].mag);
            spdlog::debug("FFT peak ratio: {:.2f}x ({:.2f} dB) between #1 and #2", ratio, ratio_db);
        }
    }

    if (peak <= 0) return std::unexpected("Noise floor too high");
    return (peak * m_fps / m_ws) * 60.0;
}
