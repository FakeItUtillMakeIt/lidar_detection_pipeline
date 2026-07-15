#ifndef __SECOND_VOXELIZATION_HPP__
#define __SECOND_VOXELIZATION_HPP__

#include <memory>
#include "common/dtype.hpp"

namespace pointpillar {
namespace lidar {

struct SecondVoxelizationParameter {
    nvtype::Float3 min_range;
    nvtype::Float3 max_range;
    nvtype::Float3 voxel_size;
    nvtype::Int3 grid_size;
    int max_voxels;
    int max_points_per_voxel;
    int max_points;
    int num_feature;

    static nvtype::Int3 compute_grid_size(const nvtype::Float3& max_range, const nvtype::Float3& min_range,
                                          const nvtype::Float3& voxel_size);
};

class SecondVoxelization {
public:
    virtual ~SecondVoxelization() = default;
    virtual void forward(const float* points, int num_points, void* stream = nullptr) = 0;

    virtual const float* features() = 0;       // (num_voxels, num_feature) after MeanVFE
    virtual const unsigned int* coords() = 0;  // (num_voxels, 4) (batch, z, y, x)
    virtual const unsigned int* params() = 0;  // (1) num_voxels
};

std::shared_ptr<SecondVoxelization> create_second_voxelization(SecondVoxelizationParameter param);

}  // namespace lidar
}  // namespace pointpillar

#endif  // __SECOND_VOXELIZATION_HPP__
