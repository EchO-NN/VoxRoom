// Deterministic saved-snapshot adapter. All geometry/segmentation methods are
// compiled verbatim from the pinned upstream SysNav room_segmentation.cpp.
// No executor is spun: callbacks and the segmentation timer are ordered here.
#include "room_segmentation/room_segmentation_node.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

sensor_msgs::msg::PointCloud2::SharedPtr cloud(const fs::path &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() % 16 != 0) throw std::runtime_error("Invalid XYZI cloud: " + path.string());
    const auto count = static_cast<size_t>(in.tellg()) / 16;
    in.seekg(0);
    pcl::PointCloud<pcl::PointXYZI> points;
    points.resize(count);
    for (auto &point : points) {
        float values[4];
        in.read(reinterpret_cast<char *>(values), 16);
        point.x = values[0]; point.y = values[1]; point.z = values[2]; point.intensity = values[3];
    }
    auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(points, *msg);
    msg->header.frame_id = "map";
    return msg;
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const fs::path input(argv[1]), output(argv[2]);
    try {
        rclcpp::init(argc, argv);
        auto node = std::make_shared<room_segmentation::RoomSegmentationNode>();
        node->timer_->cancel();
        auto odom = std::make_shared<nav_msgs::msg::Odometry>();
        std::ifstream pose(input / "pose.txt");
        if (!(pose >> odom->pose.pose.position.x >> odom->pose.pose.position.y >> odom->pose.pose.position.z))
            throw std::runtime_error("Missing saved robot pose");
        odom->pose.pose.orientation.w = 1;
        node->stateEstimationCallback(odom);

        // Five disjoint chunks: insert each observed occupied point exactly once.
        // Do not insert invented floor points or repeatedly insert the snapshot.
        for (int i = 0; i < 5; ++i)
            node->laserCloudCallback(cloud(input / ("scan_" + std::to_string(i) + ".bin")));
        if (!node->segment_flag_) throw std::runtime_error("Upstream scan accumulator did not complete");
        const double before = cv::sum(node->navigable_map_all_)[0];
        node->freespaceCloudCallback(cloud(input / "free.bin"));
        const double after_free = cv::sum(node->navigable_map_all_)[0];
        node->occupiedCloudCallback(cloud(input / "occupied.bin"));
        const double after_state = cv::sum(node->navigable_map_all_)[0];
        const int free_state_cells = cv::countNonZero(node->state_map_all_);
        // This is the unmodified upstream timer, including roomSegmentation().
        node->timerCallback();
        fs::create_directories(output);
        cv::Mat labels = node->room_mask_.clone();
        std::ofstream mask(output / "labels.i32", std::ios::binary);
        mask.write(reinterpret_cast<const char *>(labels.data), labels.total() * sizeof(int32_t));
        std::ofstream doors(output / "doors.f32", std::ios::binary);
        for (const auto &p : node->door_cloud_->points) {
            float values[3] = {p.x, p.y, p.z};
            doors.write(reinterpret_cast<const char *>(values), sizeof(values));
        }
        std::ofstream audit(output / "native_audit.json");
        audit << "{\"rows\":" << labels.rows << ",\"cols\":" << labels.cols
              << ",\"accumulated_before_free\":" << before
              << ",\"accumulated_after_free\":" << after_free
              << ",\"accumulated_after_state\":" << after_state
              << ",\"state_free_cells\":" << free_state_cells
              << ",\"sequence\":\"scan5_then_free_then_occupied_then_segmentation\"}\n";
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        if (rclcpp::ok()) rclcpp::shutdown();
        return 1;
    }
}
