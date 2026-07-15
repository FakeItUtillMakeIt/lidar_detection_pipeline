#ifndef __SECOND_BACKBONE_HPP__
#define __SECOND_BACKBONE_HPP__

#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "common/dtype.hpp"

namespace pointpillar {
namespace lidar {

class SecondBackbone {
public:
    virtual ~SecondBackbone() = default;

    virtual bool load_weights(const std::string& path) = 0;

    // Forward: voxel_features (N, 4) after MeanVFE, coords (N, 4), num_voxels
    // Output: bev_features (1, 256, 200, 176) — dense BEV for 2D backbone
    virtual void forward(
        const float* voxel_features,
        const unsigned int* coords,
        unsigned int num_voxels,
        float* bev_features,
        void* stream = nullptr
    ) = 0;

    virtual void print() = 0;
};

std::shared_ptr<SecondBackbone> create_second_backbone(const std::string& weights_path);

}  // namespace lidar
}  // namespace pointpillar

#endif  // __SECOND_BACKBONE_HPP__
