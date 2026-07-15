#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "second-detector.hpp"
#include "lidar-postprocess.hpp"

int main() {
    // Load a KITTI point cloud
    const char* bin_path = "../data/000000.bin";
    FILE* f = fopen(bin_path, "rb");
    if (!f) { printf("Cannot open %s\n", bin_path); return 1; }
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    rewind(f);
    int num_floats = file_size / sizeof(float);
    float* points = new float[num_floats];
    fread(points, sizeof(float), num_floats, f);
    fclose(f);
    int num_points = num_floats / 4;
    printf("Loaded %d points from %s\n", num_points, bin_path);

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create SECOND detector
    pointpillar::lidar::SecondDetectorConfig config;
    config.score_thresh = 0.3f;
    config.nms_thresh = 0.01f;

    auto detector = pointpillar::lidar::SecondDetector::create(config);
    if (!detector) {
        printf("Failed to create SecondDetector\n");
        delete[] points;
        return 1;
    }
    printf("SecondDetector created successfully\n");
    detector->print();

    // Run inference
    printf("Running inference...\n");
    auto detections = detector->detect(points, num_points, stream);
    cudaStreamSynchronize(stream);

    printf("Detected %zu objects:\n", detections.size());
    for (size_t i = 0; i < detections.size() && i < 20; i++) {
        auto& d = detections[i];
        printf("  [%zu] x=%.2f y=%.2f z=%.2f w=%.2f l=%.2f h=%.2f rt=%.2f id=%d score=%.4f\n",
               i, d.x, d.y, d.z, d.w, d.l, d.h, d.rt, d.id, d.score);
    }

    cudaStreamDestroy(stream);
    delete[] points;
    printf("Done.\n");
    return 0;
}
