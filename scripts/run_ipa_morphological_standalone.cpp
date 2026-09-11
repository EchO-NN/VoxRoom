#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ipa_room_segmentation/morphological_segmentation.h>

int main(int argc, char** argv)
{
    if (argc != 6) {
        std::cerr << "Usage: morph INPUT.png OUTPUT.yml.gz RESOLUTION LOWER_M2 UPPER_M2\n";
        return 2;
    }
    try {
        // The original action server reads map resolution into a float.
        const float resolution = std::stof(argv[3]);
        const double lower = std::stod(argv[4]), upper = std::stod(argv[5]);
        if (!std::isfinite(resolution) || resolution <= 0 || !std::isfinite(lower)
            || !std::isfinite(upper) || lower < 0 || upper <= lower)
            return 3;
        cv::setNumThreads(1);
        std::srand(1);  // Repeatable label IDs; labels are categorical, not geometric priorities.
        cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE), output;
        if (input.empty()) return 4;
        cv::Mat invalid = (input != 0) & (input != 255);
        if (cv::countNonZero(invalid)) return 5;
        MorphologicalSegmentation algorithm;
        algorithm.segmentMap(input, output, resolution, lower, upper);
        cv::FileStorage storage(argv[2], cv::FileStorage::WRITE);
        if (!storage.isOpened()) return 6;
        storage << "segmented_map" << output;
        storage << "resolution" << resolution << "lower_m2" << lower << "upper_m2" << upper;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 7;
    }
}
