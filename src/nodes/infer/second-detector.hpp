#ifndef __SECOND_DETECTOR_HPP__
#define __SECOND_DETECTOR_HPP__

#include <memory>
#include <string>
#include <vector>
#include "lidar-postprocess.hpp"

namespace pointpillar {
namespace lidar {

struct SecondDetectorConfig {
    // Voxelization (SECOND KITTI)
    float min_range[3] = {0.0f, -39.68f, -3.0f};
    float max_range[3] = {69.12f, 39.68f, 1.0f};
    float voxel_size[3] = {0.05f, 0.05f, 0.1f};
    int max_voxels = 60000;
    int max_points_per_voxel = 5;
    int max_points = 300000;
    int num_feature = 4;

    // Model paths
    std::string backbone_3d_weights = "../model/second_3d_backbone.weights";
    std::string backbone_2d_engine = "../model/second_2d_backbone.plan";

    // Postprocess
    float score_thresh = 0.1f;
    float nms_thresh = 0.01f;
};

class SecondDetector {
public:
    static std::shared_ptr<SecondDetector> create(const SecondDetectorConfig& config);

    virtual ~SecondDetector() = default;

    virtual std::vector<BoundingBox> detect(
        const float* points, int num_points, void* stream = nullptr
    ) = 0;

    virtual void set_timer(bool enable) = 0;
    virtual void print() = 0;
};

}  // namespace lidar
}  // namespace pointpillar

#endif  // __SECOND_DETECTOR_HPP__
