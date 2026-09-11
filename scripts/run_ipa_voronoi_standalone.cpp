#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include <ipa_room_segmentation/voronoi_segmentation.h>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: run_ipa_voronoi_standalone INPUT.png OUTPUT.yml.gz\n";
        return 2;
    }

    const cv::Mat input = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (input.empty())
    {
        std::cerr << "failed to read input image: " << argv[1] << "\n";
        return 3;
    }

    cv::Mat segmented;
    VoronoiSegmentation algorithm;
    algorithm.segmentMap(
        input,
        segmented,
        0.05,       // map resolution [m/cell]
        0.1,        // minimum room area [m^2]
        1000000.0,  // maximum room area [m^2]
        280,        // Voronoi neighborhood index
        150,        // maximum neighborhood iterations
        0.5,        // minimum critical-point distance factor
        12.5,       // maximum area for room merging [m^2]
        false);

    cv::FileStorage output(argv[2], cv::FileStorage::WRITE);
    if (!output.isOpened())
    {
        std::cerr << "failed to open output: " << argv[2] << "\n";
        return 4;
    }
    output << "segmented_map" << segmented;
    output.release();
    return EXIT_SUCCESS;
}
