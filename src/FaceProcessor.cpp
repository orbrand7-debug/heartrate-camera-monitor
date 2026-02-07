#include "FaceProcessor.hpp"
#include <dlib/opencv.h>
#include <filesystem>
#include <chrono>

FaceProcessor::FaceProcessor(const std::string& model_path) {
    m_detector = dlib::get_frontal_face_detector();
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Dlib model file not found at: " + model_path);
    }
    dlib::deserialize(model_path) >> m_shape_predictor;
}


void FaceProcessor::draw_debug(cv::Mat& frame, const dlib::full_object_detection& landmarks, cv::Mat forehead_rect) const {
    // 1. Draw Landmarks
    for (unsigned long i = 0; i < landmarks.num_parts(); ++i) {
        cv::circle(frame, cv::Point(landmarks.part(i).x(), landmarks.part(i).y()), 2, cv::Scalar(0, 255, 255), -1);
    }
    // 2. Draw Face Rect (approx from landmarks)
    cv::Rect face_rect(landmarks.get_rect().left(), landmarks.get_rect().top(), 
                       landmarks.get_rect().width(), landmarks.get_rect().height());
    cv::rectangle(frame, face_rect, cv::Scalar(255, 0, 0), 2);
    // 3. Draw Forehead
    cv::polylines(frame, forehead_rect, true, cv::Scalar(0, 255, 0), 2);
}

std::expected<dlib::full_object_detection, std::string> FaceProcessor::get_central_face(
    const cv::Mat& frame, FaceTimings* timings) {
    auto to_ms = [](auto d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    auto t0 = std::chrono::steady_clock::now();
    dlib::cv_image<dlib::bgr_pixel> dlib_img(frame);
    auto faces = m_detector(dlib_img);
    auto t1 = std::chrono::steady_clock::now();
    if (timings) {
        timings->detect_ms = to_ms(t1 - t0);
    }

    if (faces.empty()) {
        return std::unexpected("No faces in view");
    }

    auto t2 = std::chrono::steady_clock::now();
    dlib::point frame_center(frame.cols / 2, frame.rows / 2);
    
    auto closest_face = std::min_element(faces.begin(), faces.end(), [&](const auto& a, const auto& b) {
        return dlib::length(center(a) - frame_center) < dlib::length(center(b) - frame_center);
    });
    auto t3 = std::chrono::steady_clock::now();
    if (timings) {
        timings->select_ms = to_ms(t3 - t2);
    }

    auto t4 = std::chrono::steady_clock::now();
    auto landmarks = m_shape_predictor(dlib_img, *closest_face);
    auto t5 = std::chrono::steady_clock::now();
    if (timings) {
        timings->predict_ms = to_ms(t5 - t4);
    }
    return landmarks;
}

cv::Mat FaceProcessor::get_stabilized_forehead(const cv::Mat& frame, const dlib::full_object_detection& landmarks, cv::Mat* out_corners) const
{
    // 1. Define Standard Space Landmarks (3x1 CV_32FC2)
    cv::Mat dstTri = (cv::Mat_<cv::Vec2f>(3, 1) << 
        cv::Vec2f(60.0f, 100.0f),  // Left Eyebrow Peak
        cv::Vec2f(140.0f, 100.0f), // Right Eyebrow Peak
        cv::Vec2f(100.0f, 130.0f)  // Nose Bridge
    );

    // Using Rect2f (float) ensures tl() and br() return Point2f
    const cv::Rect2f std_forehead_rect(70.0f, 40.0f, 60.0f, 45.0f);

    // 2. Extract Source Landmarks (Dlib points directly to cv::Vec2f)
    cv::Mat srcTri = (cv::Mat_<cv::Vec2f>(3, 1) << 
        cv::Vec2f(cv::Point2f(landmarks.part(19).x(), landmarks.part(19).y())),
        cv::Vec2f(cv::Point2f(landmarks.part(24).x(), landmarks.part(24).y())),
        cv::Vec2f(cv::Point2f(landmarks.part(27).x(), landmarks.part(27).y()))
    );

    // 3. Coordinate Transformation
    cv::Mat M = cv::getAffineTransform(srcTri, dstTri);
    cv::Mat M_inv;
    cv::invertAffineTransform(M, M_inv);

    // 4. Vectorized Corner Calculation
    // No casting needed: tl() and br() on Rect2f return Point2f
    cv::Mat std_corners = (cv::Mat_<cv::Vec2f>(4, 1) << 
        cv::Vec2f(std_forehead_rect.tl()),
        cv::Vec2f(std_forehead_rect.x + std_forehead_rect.width, std_forehead_rect.y),
        cv::Vec2f(std_forehead_rect.br()),
        cv::Vec2f(std_forehead_rect.x, std_forehead_rect.y + std_forehead_rect.height)
    );

    cv::Mat frame_corners;
    cv::transform(std_corners, frame_corners, M_inv);

    // out_corners is used for drawing, so we eventually need integers
    if (out_corners) {
        frame_corners.convertTo(*out_corners, CV_32S);
    }

    // 5. Create Micro-Warp Source Crop
    // boundingRect returns a standard Rect (int), which we need for Mat indexing
    cv::Rect frame_roi = cv::boundingRect(frame_corners);
    frame_roi &= cv::Rect(0, 0, frame.cols, frame.rows);

    if (frame_roi.width < 2 || frame_roi.height < 2) return cv::Mat();

    // 6. Vectorized Point Adjustment
    // frame_roi.tl() returns Point2i, which converts implicitly to Point2f for the Scalar
    cv::Point2f src_offset = frame_roi.tl();
    cv::Point2f dst_offset = std_forehead_rect.tl();

    cv::Mat adjSrcTri = srcTri - cv::Scalar(src_offset.x, src_offset.y);
    cv::Mat adjDstTri = dstTri - cv::Scalar(dst_offset.x, dst_offset.y);

    // 7. Execution
    cv::Mat final_M = cv::getAffineTransform(adjSrcTri, adjDstTri);
    cv::Mat result;
    
    // cv::Size takes integers, so we use the size of the standard rect
    cv::warpAffine(frame(frame_roi), result, final_M, std_forehead_rect.size());

    return result;
}

cv::Scalar FaceProcessor::get_avg_bgr(const cv::Mat& frame) const {
    return cv::mean(frame);
}
