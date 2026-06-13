#ifndef __DETECTOR_HPP__
#define __DETECTOR_HPP__

#include <memory>
#include <string>
#include <vector>
#include "lidar-postprocess.hpp"

namespace pointpillar {
namespace lidar {

struct DetectorConfig {
    // Voxelization
    float min_range[3] = {0.0f, -39.68f, -3.0f};
    float max_range[3] = {69.12f, 39.68f, 1.0f};
    float voxel_size[3] = {0.16f, 0.16f, 4.0f};
    int max_voxels = 40000;
    int max_points_per_voxel = 32;
    int max_points = 300000;
    int num_feature = 4;

    // Model
    std::string model_path = "../model/pointpillar.plan";

    // Postprocess
    float score_thresh = 0.1f;
    float nms_thresh = 0.01f;
};

class Detector {
public:
    static std::shared_ptr<Detector> create(const DetectorConfig& config);

    virtual ~Detector() = default;

    virtual std::vector<BoundingBox> detect(
        const float* points, int num_points, void* stream = nullptr
    ) = 0;

    virtual void set_timer(bool enable) = 0;
    virtual void print() = 0;
};

}  // namespace lidar
}  // namespace pointpillar

#endif  // __DETECTOR_HPP__
