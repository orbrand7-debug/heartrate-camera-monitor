#include "FaceProcessor.hpp"
#include <dlib/opencv.h>
#include <filesystem>

FaceProcessor::FaceProcessor(const std::string& model_path) {
    m_detector = dlib::get_frontal_face_detector();
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Dlib model file not found at: " + model_path);
    }
    dlib::deserialize(model_path) >> m_shape_predictor;
}


void FaceProcessor::draw_debug(cv::Mat& frame, const dlib::full_object_detection& landmarks, cv::Rect forehead) const {
    // 1. Draw Landmarks
    for (unsigned long i = 0; i < landmarks.num_parts(); ++i) {
        cv::circle(frame, cv::Point(landmarks.part(i).x(), landmarks.part(i).y()), 2, cv::Scalar(0, 255, 255), -1);
    }
    // 2. Draw Face Rect (approx from landmarks)
    cv::Rect face_rect(landmarks.get_rect().left(), landmarks.get_rect().top(), 
                       landmarks.get_rect().width(), landmarks.get_rect().height());
    cv::rectangle(frame, face_rect, cv::Scalar(255, 0, 0), 2);
    // 3. Draw Forehead
    cv::rectangle(frame, forehead, cv::Scalar(0, 255, 0), 2);
}

std::expected<dlib::full_object_detection, std::string> FaceProcessor::get_central_face(const cv::Mat& frame) {
    dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
    auto faces = m_detector(dlib_img);

    if (faces.empty()) {
        return std::unexpected("No faces in view");
    }

    dlib::point frame_center(frame.cols / 2, frame.rows / 2);
    
    auto closest_face = std::min_element(faces.begin(), faces.end(), [&](const auto& a, const auto& b) {
        return dlib::length(center(a) - frame_center) < dlib::length(center(b) - frame_center);
    });

    return m_shape_predictor(dlib_img, *closest_face);
}

cv::Rect FaceProcessor::get_forehead_roi(const dlib::full_object_detection& landmarks) const {
    // Indices 17-21: Left eyebrow, 22-26: Right eyebrow
    long eyebrow_y = 0;
    for (int i = 17; i <= 26; ++i) eyebrow_y += landmarks.part(i).y();
    eyebrow_y /= 10;

    // Horizontal scale: distance between outer eye corners (36 and 45)
    long eye_dist = landmarks.part(45).x() - landmarks.part(36).x();
    
    int roi_width = static_cast<int>(eye_dist * 0.5);
    int roi_height = static_cast<int>(eye_dist * 0.2);
    
    // Position: Center on point 27 (top of nose), shift up
    int x = landmarks.part(27).x() - (roi_width / 2);
    int y = static_cast<int>(eyebrow_y - roi_height - 10);

    return {x, y, roi_width, roi_height};
}

double FaceProcessor::get_avg_hue(const cv::Mat& frame, cv::Rect roi) const {
    // Keep ROI inside frame boundaries
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (roi.area() <= 0) {
        return 0.0;
    }

    cv::Mat hsv_roi;
    cv::cvtColor(frame(roi), hsv_roi, cv::COLOR_BGR2HSV);
    double hue = cv::mean(hsv_roi)[0];
    return hue;
}