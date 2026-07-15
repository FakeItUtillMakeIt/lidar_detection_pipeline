#include "second-detector.hpp"
#include "second-voxelization.hpp"
#include "second-backbone.hpp"
#include "lidar-postprocess.hpp"
#include "common/tensorrt.hpp"
#include "common/check.hpp"
#include <numeric>

namespace pointpillar {
namespace lidar {

class SecondDetectorImpl : public SecondDetector {
public:
    ~SecondDetectorImpl() override {
        if (points_device_) cudaFree(points_device_);
        if (bev_device_) cudaFree(bev_device_);
        if (cls_device_) cudaFree(cls_device_);
        if (box_device_) cudaFree(box_device_);
        if (dir_device_) cudaFree(dir_device_);
    }

    bool init(const SecondDetectorConfig& config) {
        config_ = config;

        // 1. Create voxelization
        SecondVoxelizationParameter vp;
        vp.min_range = {config.min_range[0], config.min_range[1], config.min_range[2]};
        vp.max_range = {config.max_range[0], config.max_range[1], config.max_range[2]};
        vp.voxel_size = {config.voxel_size[0], config.voxel_size[1], config.voxel_size[2]};
        vp.grid_size = SecondVoxelizationParameter::compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
        vp.max_voxels = config.max_voxels;
        vp.max_points_per_voxel = config.max_points_per_voxel;
        vp.max_points = config.max_points;
        vp.num_feature = config.num_feature;
        voxelization_ = create_second_voxelization(vp);
        if (!voxelization_) return false;

        // 2. Load 3D sparse backbone
        backbone_3d_ = create_second_backbone(config.backbone_3d_weights);
        if (!backbone_3d_) return false;

        // 3. Load 2D backbone + head TRT engine
        engine_2d_ = TensorRT::load(config.backbone_2d_engine);
        if (!engine_2d_) return false;

        // 4. Query output dims and allocate buffers for cls/box/dir
        auto cls_dims = engine_2d_->static_dims(1);
        auto box_dims = engine_2d_->static_dims(2);
        auto dir_dims = engine_2d_->static_dims(3);

        int cls_vol = std::accumulate(cls_dims.begin(), cls_dims.end(), 1, std::multiplies<int>());
        int box_vol = std::accumulate(box_dims.begin(), box_dims.end(), 1, std::multiplies<int>());
        int dir_vol = std::accumulate(dir_dims.begin(), dir_dims.end(), 1, std::multiplies<int>());

        checkRuntime(cudaMalloc(&cls_device_, cls_vol * sizeof(float)));
        checkRuntime(cudaMalloc(&box_device_, box_vol * sizeof(float)));
        checkRuntime(cudaMalloc(&dir_device_, dir_vol * sizeof(float)));

        // 5. Create postprocess
        // SECOND feature map stride = 8, so feature_size = grid / 8
        // feature_size = (W, H) = (grid_x/8, grid_y/8)
        PostProcessParameter pp;
        pp.min_range = vp.min_range;
        pp.max_range = vp.max_range;
        pp.feature_size = {vp.grid_size.x / 8, vp.grid_size.y / 8};
        pp.score_thresh = config.score_thresh;
        pp.nms_thresh = config.nms_thresh;
        pp.dir_offset = 0.78539f;
        postprocess_ = create_postprocess(pp);
        if (!postprocess_) return false;

        // 6. Allocate buffers
        capacity_points_ = config.max_points;
        size_t bytes = (size_t)capacity_points_ * config.num_feature * sizeof(float);
        checkRuntime(cudaMalloc(&points_device_, bytes));

        // BEV buffer: (1, 256, 200, 176)
        bev_size_ = 256 * 200 * 176;
        checkRuntime(cudaMalloc(&bev_device_, bev_size_ * sizeof(float)));

        return true;
    }

    std::vector<BoundingBox> detect(const float* points, int num_points, void* stream) override {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        int cap = static_cast<int>(capacity_points_);
        num_points = std::min(cap, num_points);

        size_t bytes = (size_t)num_points * config_.num_feature * sizeof(float);
        checkRuntime(cudaMemcpyAsync(points_device_, points, bytes, cudaMemcpyHostToDevice, _stream));

        // 1. Voxelization + MeanVFE
        voxelization_->forward(points_device_, num_points, _stream);

        unsigned int num_voxels = 0;
        checkRuntime(cudaMemcpyAsync(&num_voxels, voxelization_->params(), sizeof(unsigned int),
                                      cudaMemcpyDeviceToHost, _stream));
        checkRuntime(cudaStreamSynchronize(_stream));

        if (num_voxels == 0) {
            return {};
        }

        // 2. 3D sparse backbone → BEV features (256, 200, 176)
        backbone_3d_->forward(
            voxelization_->features(),
            voxelization_->coords(),
            num_voxels,
            bev_device_,
            _stream);

        // 3. 2D backbone + head TRT engine
        engine_2d_->forward({bev_device_, cls_device_, box_device_, dir_device_}, _stream);

        // 4. Postprocess (decode + NMS)
        postprocess_->forward(cls_device_, box_device_, dir_device_, _stream);

        return postprocess_->bndBoxVec();
    }

    void set_timer(bool enable) override { timer_enabled_ = enable; }

    void print() override {
        backbone_3d_->print();
        engine_2d_->print("SECOND 2D Backbone");
    }

private:
    SecondDetectorConfig config_;
    std::shared_ptr<SecondVoxelization> voxelization_;
    std::shared_ptr<SecondBackbone> backbone_3d_;
    std::shared_ptr<TensorRT::Engine> engine_2d_;
    std::shared_ptr<PostProcess> postprocess_;

    float* points_device_ = nullptr;
    float* bev_device_ = nullptr;
    float* cls_device_ = nullptr;
    float* box_device_ = nullptr;
    float* dir_device_ = nullptr;
    size_t capacity_points_ = 0;
    size_t bev_size_ = 0;
    bool timer_enabled_ = false;
};

std::shared_ptr<SecondDetector> SecondDetector::create(const SecondDetectorConfig& config) {
    auto impl = std::make_shared<SecondDetectorImpl>();
    if (!impl->init(config)) return nullptr;
    return impl;
}

}  // namespace lidar
}  // namespace pointpillar
