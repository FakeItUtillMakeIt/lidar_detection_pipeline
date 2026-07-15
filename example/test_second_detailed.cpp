#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "second-voxelization.hpp"
#include "second-backbone.hpp"
#include "common/tensorrt.hpp"

using namespace pointpillar::lidar;

void check_cuda(const char* msg) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("CUDA error after %s: %s\n", msg, cudaGetErrorString(e));
}

int main() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Load KITTI point cloud
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
    printf("Loaded %d points\n", num_points);

    for (int i = 0; i < 10 && i < num_points; i++) {
        printf("  pt[%d]: x=%.2f y=%.2f z=%.2f r=%.2f\n", i,
               points[i*4], points[i*4+1], points[i*4+2], points[i*4+3]);
    }

    // Voxelization
    printf("\n--- Voxelization ---\n");
    SecondVoxelizationParameter vp;
    vp.min_range = {0.0f, -40.0f, -3.0f};
    vp.max_range = {70.4f, 40.0f, 1.0f};
    vp.voxel_size = {0.05f, 0.05f, 0.1f};
    vp.grid_size = SecondVoxelizationParameter::compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    vp.max_voxels = 60000;
    vp.max_points_per_voxel = 5;
    vp.max_points = 300000;
    vp.num_feature = 4;
    printf("grid: (%d, %d, %d)\n", vp.grid_size.x, vp.grid_size.y, vp.grid_size.z);

    // 3D Backbone — load BEFORE voxelization forward to isolate any CUDA errors
    printf("\n--- 3D Backbone (loading weights) ---\n");
    auto backbone = create_second_backbone("../model/second_3d_backbone.weights");
    if (!backbone) { printf("Failed to create backbone\n"); return 1; }
    backbone->print();
    cudaDeviceSynchronize();
    printf("Backbone loaded successfully\n");

    auto voxelizer = create_second_voxelization(vp);
    if (!voxelizer) { printf("Failed to create voxelizer\n"); return 1; }
    cudaDeviceSynchronize();
    printf("Voxelizer created\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Upload points to GPU first (the CUDA kernel reads from device memory)
    float* d_points;
    cudaMalloc(&d_points, num_points * 4 * sizeof(float));
    cudaMemcpy(d_points, points, num_points * 4 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start, stream);
    voxelizer->forward(d_points, num_points, stream);
    cudaStreamSynchronize(stream);
    check_cuda("voxelization forward");

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaDeviceSynchronize();
    check_cuda("voxelization device sync");

    // Verify voxelization results before using them
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("ERROR before backbone forward: %s\n", cudaGetErrorString(e)); return 1; }

    unsigned int num_voxels = 0;
    cudaMemcpy(&num_voxels, voxelizer->params(), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Voxelization: %d voxels, %.2f ms\n", num_voxels, ms);

    if (num_voxels > 0) {
        int* h_coords = new int[num_voxels * 4];
        float* h_feat = new float[num_voxels * 4];
        cudaMemcpy(h_coords, voxelizer->coords(), num_voxels * 4 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_feat, voxelizer->features(), num_voxels * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 20 && i < (int)num_voxels; i++) {
            printf("  vox[%d]: coords(%d,%d,%d,%d) feat(%.4f,%.4f,%.4f,%.4f)\n", i,
                   h_coords[i*4+0], h_coords[i*4+1], h_coords[i*4+2], h_coords[i*4+3],
                   h_feat[i*4+0], h_feat[i*4+1], h_feat[i*4+2], h_feat[i*4+3]);
        }
        delete[] h_coords;
        delete[] h_feat;
    }

    float* d_bev;
    cudaError_t ce = cudaMalloc(&d_bev, 256 * 200 * 176 * sizeof(float));
    printf("cudaMalloc d_bev: %s (%d), ptr=%p\n", cudaGetErrorString(ce), ce, d_bev);
    ce = cudaMemset(d_bev, 0, 256 * 200 * 176 * sizeof(float));
    printf("cudaMemset d_bev: %s (%d)\n", cudaGetErrorString(ce), ce);

    cudaDeviceSynchronize();
    check_cuda("pre-backbone-forward sync");

    printf("Calling backbone->forward with num_voxels=%d...\n", num_voxels);
    cudaEventRecord(start, stream);
    backbone->forward(voxelizer->features(), voxelizer->coords(), num_voxels, d_bev, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    check_cuda("backbone");

    printf("BEV output: %.2f ms\n", ms);

    float* h_bev = new float[256 * 200 * 176];
    cudaMemcpy(h_bev, d_bev, 256 * 200 * 176 * sizeof(float), cudaMemcpyDeviceToHost);
    float mn = h_bev[0], mx = h_bev[0];
    double sum = 0;
    for (int i = 0; i < 256 * 200 * 176; i++) {
        mn = std::min(mn, h_bev[i]);
        mx = std::max(mx, h_bev[i]);
        sum += h_bev[i];
    }
    printf("BEV: min=%.4f max=%.4f mean=%.6f\n", mn, mx, sum/(256*200*176));

    // Check if BEV is all zeros
    int nz = 0;
    for (int i = 0; i < 256 * 200 * 176; i++) {
        if (h_bev[i] != 0) nz++;
    }
    printf("BEV non-zero elements: %d / %d\n", nz, 256*200*176);

    // 2D Backbone (TRT)
    printf("\n--- 2D Backbone (TRT) ---\n");
    auto trt_engine = TensorRT::load("../model/second_2d_backbone.plan");
    if (!trt_engine) { printf("Failed to load 2D backbone engine\n"); return 1; }
    trt_engine->print("SECOND 2D Backbone");

    float *d_cls, *d_box, *d_dir;
    cudaMalloc(&d_cls, 1*200*176*18*sizeof(float));
    cudaMalloc(&d_box, 1*200*176*42*sizeof(float));
    cudaMalloc(&d_dir, 1*200*176*12*sizeof(float));

    std::vector<const void*> bindings = {d_bev, d_cls, d_box, d_dir};

    cudaEventRecord(start, stream);
    bool ok = trt_engine->forward(bindings, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    check_cuda("2d backbone");

    printf("2D backbone: %.2f ms, ok=%d\n", ms, ok);

    // Print cls stats
    float* h_cls = new float[1*200*176*18];
    cudaMemcpy(h_cls, d_cls, 1*200*176*18*sizeof(float), cudaMemcpyDeviceToHost);
    float cmn = h_cls[0], cmx = h_cls[0];
    double csum = 0;
    for (int i = 0; i < 1*200*176*18; i++) {
        cmn = std::min(cmn, h_cls[i]);
        cmx = std::max(cmx, h_cls[i]);
        csum += h_cls[i];
    }
    printf("  cls: min=%.4f max=%.4f mean=%.6f\n", cmn, cmx, csum/(1*200*176*18));

    float* h_box = new float[1*200*176*42];
    cudaMemcpy(h_box, d_box, 1*200*176*42*sizeof(float), cudaMemcpyDeviceToHost);
    float bmn = h_box[0], bmx = h_box[0];
    double bsum = 0;
    for (int i = 0; i < 1*200*176*42; i++) {
        bmn = std::min(bmn, h_box[i]);
        bmx = std::max(bmx, h_box[i]);
        bsum += h_box[i];
    }
    printf("  box: min=%.4f max=%.4f mean=%.6f\n", bmn, bmx, bsum/(1*200*176*42));

    delete[] h_cls;
    delete[] h_box;
    delete[] h_bev;
    cudaFree(d_points);
    cudaFree(d_bev);
    cudaFree(d_cls);
    cudaFree(d_box);
    cudaFree(d_dir);
    cudaStreamDestroy(stream);
    delete[] points;
    printf("\nDone.\n");
    return 0;
}
