#include <cuda_fp16.h>
#include "second-voxelization.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"

namespace pointpillar {
namespace lidar {

// Kernel 1: Count points per voxel (just counting, capped at max_points_per_voxel)
static __global__ void count_voxels_kernel(
    const float* points, int num_points,
    float min_x, float max_x, float min_y, float max_y, float min_z, float max_z,
    float voxel_x, float voxel_y, float voxel_z,
    int grid_z, int grid_y, int grid_x,
    unsigned int* mask, int max_points_per_voxel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float4 p = ((float4*)points)[idx];

    if (p.x < min_x || p.x >= max_x ||
        p.y < min_y || p.y >= max_y ||
        p.z < min_z || p.z >= max_z) return;

    int vx = (p.x - min_x) / voxel_x;
    int vy = (p.y - min_y) / voxel_y;
    int vz = (p.z - min_z) / voxel_z;

    vx = min(vx, grid_x - 1);
    vy = min(vy, grid_y - 1);
    vz = min(vz, grid_z - 1);

    unsigned int voxel_index = (vz * grid_y + vy) * grid_x + vx;
    unsigned int count = atomicAdd(&mask[voxel_index], 1);

    if (count >= (unsigned int)max_points_per_voxel) {
        atomicSub(&mask[voxel_index], 1);
    }
}

// Kernel 2: Assign voxel IDs for non-empty voxels, store coords and counts
static __global__ void assign_voxel_ids_kernel(
    unsigned int* mask,
    int grid_z, int grid_y, int grid_x,
    unsigned int* num_voxels_out,
    unsigned int* voxel_num,
    unsigned int* voxel_idxs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid_z * grid_y * grid_x;
    if (idx >= total) return;

    // Deterministic (z,y,x) order matching sorted(voxel_id) in Python
    int vz = idx / (grid_y * grid_x);
    int vy = (idx / grid_x) % grid_y;
    int vx = idx % grid_x;

    unsigned int voxel_index = (vz * grid_y + vy) * grid_x + vx;
    unsigned int count = mask[voxel_index];
    if (count == 0) return;

    unsigned int voxel_id = atomicAdd(num_voxels_out, 1);
    voxel_num[voxel_id] = count;
    ((uint4*)voxel_idxs)[voxel_id] = make_uint4(0, (unsigned int)vz, (unsigned int)vy, (unsigned int)vx);
    mask[voxel_index] = voxel_id + 1;
}

// Kernel 3: Accumulate features per voxel via atomicAdd (only first max_points per voxel)
static __global__ void accumulate_vfe_kernel(
    const float* points, int num_points,
    float min_x, float max_x, float min_y, float max_y, float min_z, float max_z,
    float voxel_x, float voxel_y, float voxel_z,
    int grid_z, int grid_y, int grid_x,
    unsigned int* mask,
    float* voxel_sum,
    unsigned int* voxel_count,
    int num_feature,
    int max_points_per_voxel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float4 p = ((float4*)points)[idx];

    if (p.x < min_x || p.x >= max_x ||
        p.y < min_y || p.y >= max_y ||
        p.z < min_z || p.z >= max_z) return;

    int vx = (p.x - min_x) / voxel_x;
    int vy = (p.y - min_y) / voxel_y;
    int vz = (p.z - min_z) / voxel_z;

    vx = min(vx, grid_x - 1);
    vy = min(vy, grid_y - 1);
    vz = min(vz, grid_z - 1);

    unsigned int voxel_index = (vz * grid_y + vy) * grid_x + vx;
    unsigned int voxel_id_plus_1 = mask[voxel_index];
    if (voxel_id_plus_1 == 0) return;

    unsigned int voxel_id = voxel_id_plus_1 - 1;

    // Only accumulate if within the first max_points_per_voxel points for this voxel
    unsigned int count_before = atomicAdd(&voxel_count[voxel_id], 1);
    if (count_before >= (unsigned int)max_points_per_voxel) return;

    atomicAdd(&voxel_sum[voxel_id * num_feature + 0], p.x);
    atomicAdd(&voxel_sum[voxel_id * num_feature + 1], p.y);
    atomicAdd(&voxel_sum[voxel_id * num_feature + 2], p.z);
    atomicAdd(&voxel_sum[voxel_id * num_feature + 3], p.w);
}

// Kernel 4: MeanVFE normalization (divide sum by count)
// num_voxels_ptr is a device pointer (params_input_)
static __global__ void normalize_vfe_kernel(
    float* voxel_sum,
    unsigned int* voxel_num,
    unsigned int* num_voxels_ptr,
    int num_feature)
{
    int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_voxels = *num_voxels_ptr;
    if (voxel_id >= num_voxels) return;

    unsigned int count = voxel_num[voxel_id];
    if (count == 0) return;

    float inv_count = 1.0f / count;
    for (int c = 0; c < num_feature; c++) {
        voxel_sum[voxel_id * num_feature + c] *= inv_count;
    }
}

nvtype::Int3 SecondVoxelizationParameter::compute_grid_size(
    const nvtype::Float3& max_range, const nvtype::Float3& min_range,
    const nvtype::Float3& voxel_size)
{
    nvtype::Int3 size;
    size.x = (int)((max_range.x - min_range.x) / voxel_size.x + 0.5f);
    size.y = (int)((max_range.y - min_range.y) / voxel_size.y + 0.5f);
    size.z = (int)((max_range.z - min_range.z) / voxel_size.z + 0.5f);
    return size;
}

class SecondVoxelizationImpl : public SecondVoxelization {
public:
    ~SecondVoxelizationImpl() override {
        cudaFree(mask_);
        cudaFree(voxel_sum_);
        cudaFree(voxel_num_);
        cudaFree(voxel_count_);
        cudaFree(voxel_idxs_);
        cudaFree(params_input_);
    }

    bool init(SecondVoxelizationParameter param) {
        param_ = param;

        size_t grid_volume = (size_t)param_.grid_size.z * param_.grid_size.y * param_.grid_size.x;
        mask_size_ = grid_volume * sizeof(unsigned int);
        voxel_sum_size_ = (size_t)param_.max_voxels * param_.num_feature * sizeof(float);
        voxel_num_size_ = (size_t)param_.max_voxels * sizeof(unsigned int);
        voxel_idxs_size_ = (size_t)param_.max_voxels * 4 * sizeof(unsigned int);
        voxel_count_size_ = (size_t)param_.max_voxels * sizeof(unsigned int);

        checkRuntime(cudaMalloc(&mask_, mask_size_));
        checkRuntime(cudaMalloc(&voxel_sum_, voxel_sum_size_));
        checkRuntime(cudaMalloc(&voxel_num_, voxel_num_size_));
        checkRuntime(cudaMalloc(&voxel_count_, voxel_count_size_));
        checkRuntime(cudaMalloc(&voxel_idxs_, voxel_idxs_size_));
        checkRuntime(cudaMalloc(&params_input_, sizeof(unsigned int)));

        return true;
    }

    void forward(const float* points, int num_points, void* stream) override {
        cudaStream_t _stream = (cudaStream_t)stream;

        checkRuntime(cudaMemsetAsync(mask_, 0, mask_size_, _stream));
        checkRuntime(cudaMemsetAsync(voxel_sum_, 0, voxel_sum_size_, _stream));
        checkRuntime(cudaMemsetAsync(voxel_num_, 0, voxel_num_size_, _stream));
        checkRuntime(cudaMemsetAsync(voxel_count_, 0, voxel_count_size_, _stream));
        checkRuntime(cudaMemsetAsync(voxel_idxs_, 0, voxel_idxs_size_, _stream));
        checkRuntime(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), _stream));
        checkRuntime(cudaStreamSynchronize(_stream));
        checkRuntime(cudaGetLastError());

        dim3 threads_1d(256);
        dim3 blocks_1d((num_points + 255) / 256);

        count_voxels_kernel<<<blocks_1d, threads_1d, 0, _stream>>>(
            points, num_points,
            param_.min_range.x, param_.max_range.x,
            param_.min_range.y, param_.max_range.y,
            param_.min_range.z, param_.max_range.z,
            param_.voxel_size.x, param_.voxel_size.y, param_.voxel_size.z,
            param_.grid_size.z, param_.grid_size.y, param_.grid_size.x,
            mask_, param_.max_points_per_voxel);
        checkRuntime(cudaStreamSynchronize(_stream));
        checkRuntime(cudaGetLastError());

        size_t grid_volume = (size_t)param_.grid_size.z * param_.grid_size.y * param_.grid_size.x;
        dim3 threads_assign(512);
        dim3 blocks_assign((unsigned int)((grid_volume + 511) / 512));
        assign_voxel_ids_kernel<<<blocks_assign, threads_assign, 0, _stream>>>(
            mask_,
            param_.grid_size.z, param_.grid_size.y, param_.grid_size.x,
            params_input_,
            voxel_num_,
            voxel_idxs_);
        checkRuntime(cudaStreamSynchronize(_stream));
        checkRuntime(cudaGetLastError());

        accumulate_vfe_kernel<<<blocks_1d, threads_1d, 0, _stream>>>(
            points, num_points,
            param_.min_range.x, param_.max_range.x,
            param_.min_range.y, param_.max_range.y,
            param_.min_range.z, param_.max_range.z,
            param_.voxel_size.x, param_.voxel_size.y, param_.voxel_size.z,
            param_.grid_size.z, param_.grid_size.y, param_.grid_size.x,
            mask_,
            voxel_sum_,
            voxel_count_,
            param_.num_feature,
            param_.max_points_per_voxel);
        checkRuntime(cudaStreamSynchronize(_stream));
        checkRuntime(cudaGetLastError());

        dim3 threads_norm(256);
        dim3 blocks_norm((param_.max_voxels + 255) / 256);
        normalize_vfe_kernel<<<blocks_norm, threads_norm, 0, _stream>>>(
            voxel_sum_,
            voxel_num_,
            params_input_,
            param_.num_feature);
        checkRuntime(cudaStreamSynchronize(_stream));
        checkRuntime(cudaGetLastError());
    }

    const float* features() override   { return voxel_sum_; }
    const unsigned int* coords() override { return voxel_idxs_; }
    const unsigned int* params() override { return params_input_; }

private:
    SecondVoxelizationParameter param_;

    unsigned int* mask_ = nullptr;
    float* voxel_sum_ = nullptr;
    unsigned int* voxel_num_ = nullptr;
    unsigned int* voxel_count_ = nullptr;
    unsigned int* voxel_idxs_ = nullptr;
    unsigned int* params_input_ = nullptr;

    size_t mask_size_ = 0;
    size_t voxel_sum_size_ = 0;
    size_t voxel_num_size_ = 0;
    size_t voxel_count_size_ = 0;
    size_t voxel_idxs_size_ = 0;
};

std::shared_ptr<SecondVoxelization> create_second_voxelization(SecondVoxelizationParameter param) {
    auto impl = std::make_shared<SecondVoxelizationImpl>();
    if (!impl->init(param)) {
        impl.reset();
    }
    return impl;
}

}  // namespace lidar
}  // namespace pointpillar
