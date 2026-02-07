#ifndef FACE_PROCESSOR_HPP
#define FACE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <expected>
#include <string>

struct FaceTimings {
    double detect_ms{0.0};
    double select_ms{0.0};
    double predict_ms{0.0};
};

/**
 * @class FaceProcessor
 * @brief Logic for face detection and landmark-based ROI extraction.
 */
class FaceProcessor {
public:
    /**
     * @brief Constructor. Loads the dlib shape predictor model.
     * @param model_path Path to the .dat landmark model file.
     * @throws std::runtime_error if model cannot be loaded.
     */
    explicit FaceProcessor(const std::string& model_path);

    /**
    * @brief Draws face bounding box, landmarks, and forehead ROI onto the frame.
    */
    void draw_debug(cv::Mat& frame, const dlib::full_object_detection& landmarks, cv::Mat forehead_rect) const;

    /**
     * @brief Finds the face closest to the center of the image.
     * @param frame The input BGR image.
     * @param timings Optional timing breakdown (detect/select/predict).
     * @return std::expected containing landmarks on success.
     */
    std::expected<dlib::full_object_detection, std::string> get_central_face(
        const cv::Mat& frame, FaceTimings* timings = nullptr);

    /**
     * @brief Calculates a rectangular ROI on the forehead based on eyebrow landmarks.
     * @param frame The input BGR image.
     * @param landmarks The facial landmarks detected.
     * @param out_corners Optional output for the transformed corners.
     * @return cv::Mat The stabilized forehead region.
     */
    cv::Mat get_stabilized_forehead(const cv::Mat& frame, const dlib::full_object_detection& landmarks, cv::Mat* out_corners = nullptr) const;

    /**
     * @brief Computes the mean BGR values within an ROI.
     * @return cv::Scalar containing B, G, R averages.
     */
    cv::Scalar get_avg_bgr(const cv::Mat& frame) const;

private:
    dlib::frontal_face_detector m_detector;
    dlib::shape_predictor m_shape_predictor;
};

#endif
