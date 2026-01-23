#ifndef FACE_PROCESSOR_HPP
#define FACE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <expected>
#include <string>

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
    void draw_debug(cv::Mat& frame, const dlib::full_object_detection& landmarks, cv::Rect forehead) const;

    /**
     * @brief Finds the face closest to the center of the image.
     * @param frame The input BGR image.
     * @return std::expected containing landmarks on success.
     */
    std::expected<dlib::full_object_detection, std::string> get_central_face(const cv::Mat& frame);

    /**
     * @brief Calculates a rectangular ROI on the forehead based on eyebrow landmarks.
     * @param landmarks The 68-point landmark detection result.
     * @return cv::Rect The bounding box for the forehead.
     */
    cv::Rect get_forehead_roi(const dlib::full_object_detection& landmarks) const;

    /**
     * @brief Calculates the average hue value within an ROI.
     * @param frame Input BGR frame.
     * @param roi The region to analyze.
     * @return Average hue value (0-180 in OpenCV).
     */
    double get_avg_hue(const cv::Mat& frame, cv::Rect roi) const;

private:
    dlib::frontal_face_detector m_detector;
    dlib::shape_predictor m_shape_predictor;
};

#endif