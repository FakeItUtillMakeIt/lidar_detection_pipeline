#include "detector.hpp"
#include "lidar-voxelization.hpp"
#include "lidar-backbone.hpp"
#include "lidar-postprocess.hpp"
#include "check.hpp"
#include "timer.hpp"

namespace pointpillar {
namespace lidar {

class DetectorImpl : public Detector {
public:
    ~DetectorImpl() override {
        if (points_device_) cudaFree(points_device_);
        if (points_host_) cudaFreeHost(points_host_);
    }

    bool init(const DetectorConfig& config) {
        config_ = config;

        // Create voxelization
        VoxelizationParameter vp;
        vp.min_range = {config.min_range[0], config.min_range[1], config.min_range[2]};
        vp.max_range = {config.max_range[0], config.max_range[1], config.max_range[2]};
        vp.voxel_size = {config.voxel_size[0], config.voxel_size[1], config.voxel_size[2]};
        vp.grid_size = VoxelizationParameter::compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
        vp.max_voxels = config.max_voxels;
        vp.max_points_per_voxel = config.max_points_per_voxel;
        vp.max_points = config.max_points;
        vp.num_feature = config.num_feature;
        voxelization_ = create_voxelization(vp);
        if (!voxelization_) return false;

        // Create backbone (TRT engine)
        backbone_ = create_backbone(config.model_path);
        if (!backbone_) return false;

        // Create postprocess
        PostProcessParameter pp;
        pp.min_range = vp.min_range;
        pp.max_range = vp.max_range;
        pp.feature_size = {vp.grid_size.x / 2, vp.grid_size.y / 2};
        pp.score_thresh = config.score_thresh;
        pp.nms_thresh = config.nms_thresh;
        postprocess_ = create_postprocess(pp);
        if (!postprocess_) return false;

        // Allocate point buffers
        capacity_points_ = config.max_points;
        size_t bytes = capacity_points_ * config.num_feature * sizeof(float);
        cudaMalloc(&points_device_, bytes);
        cudaMallocHost(&points_host_, bytes);

        return true;
    }

    std::vector<BoundingBox> detect(const float* points, int num_points, void* stream) override {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        int cap = static_cast<int>(capacity_points_);
        num_points = std::min(cap, num_points);

        size_t bytes = num_points * config_.num_feature * sizeof(float);
        cudaMemcpyAsync(points_host_, points, bytes, cudaMemcpyHostToHost, _stream);
        cudaMemcpyAsync(points_device_, points_host_, bytes, cudaMemcpyHostToDevice, _stream);

        voxelization_->forward(points_device_, num_points, _stream);
        backbone_->forward(
            voxelization_->features(),
            voxelization_->coords(),
            voxelization_->params(),
            _stream
        );
        postprocess_->forward(backbone_->cls(), backbone_->box(), backbone_->dir(), _stream);

        return postprocess_->bndBoxVec();
    }

    void set_timer(bool enable) override { timer_enabled_ = enable; }

    void print() override {
        backbone_->print();
    }

private:
    DetectorConfig config_;
    std::shared_ptr<Voxelization> voxelization_;
    std::shared_ptr<Backbone> backbone_;
    std::shared_ptr<PostProcess> postprocess_;

    float* points_device_ = nullptr;
    float* points_host_ = nullptr;
    size_t capacity_points_ = 0;
    bool timer_enabled_ = false;
};

std::shared_ptr<Detector> Detector::create(const DetectorConfig& config) {
    auto impl = std::make_shared<DetectorImpl>();
    if (!impl->init(config)) return nullptr;
    return impl;
}

}  // namespace lidar
}  // namespace pointpillar
