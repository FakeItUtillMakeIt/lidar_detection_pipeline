#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include "second-voxelization.hpp"
#include "second-backbone.hpp"

using namespace pointpillar::lidar;

int main() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Load backbone
    auto backbone = create_second_backbone("../model/second_3d_backbone.weights");
    if (!backbone) { std::cerr << "Failed backbone\n"; return 1; }

    // Allocate buffers
    float* d_bev;
    cudaMalloc(&d_bev, 256 * 200 * 176 * sizeof(float));

    // Create synthetic voxels: 100 voxels, all at various (z,y,x) positions
    unsigned int num_voxels = 100;
    float* d_features;
    unsigned int* d_coords;
    cudaMalloc(&d_features, num_voxels * 4 * sizeof(float));
    cudaMalloc(&d_coords, num_voxels * 4 * sizeof(unsigned int));

    unsigned int h_coords[100*4];
    float h_features[100*4];
    for (int i = 0; i < 100; i++) {
        h_coords[i*4+0] = 0;            // batch
        h_coords[i*4+1] = rand() % 2;   // z: 0 or 1
        h_coords[i*4+2] = rand() % 200; // y: 0..199
        h_coords[i*4+3] = rand() % 176; // x: 0..175
        h_features[i*4+0] = (float)(rand() % 100) / 100.0f;
        h_features[i*4+1] = (float)(rand() % 100) / 100.0f;
        h_features[i*4+2] = (float)(rand() % 100) / 100.0f;
        h_features[i*4+3] = (float)(rand() % 100) / 100.0f;
    }

    cudaMemcpy(d_features, h_features, num_voxels * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords, h_coords, num_voxels * 4 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Forward
    std::cerr << "Running backbone forward with " << num_voxels << " synthetic voxels...\n" << std::flush;
    backbone->forward(d_features, d_coords, num_voxels, d_bev, stream);

    cudaStreamSynchronize(stream);
    cudaError_t e = cudaGetLastError();
    std::cerr << "Backbone forward: " << cudaGetErrorString(e) << " (" << e << ")\n" << std::flush;
    if (e != cudaSuccess) return 1;

    // Check BEV
    float* h_bev = new float[256*200*176];
    cudaMemcpy(h_bev, d_bev, 256*200*176*sizeof(float), cudaMemcpyDeviceToHost);
    int nz = 0;
    float mn = h_bev[0], mx = h_bev[0];
    double sum = 0;
    for (int i = 0; i < 256*200*176; i++) {
        if (h_bev[i] != 0) nz++;
        mn = std::min(mn, h_bev[i]); mx = std::max(mx, h_bev[i]); sum += h_bev[i];
    }
    printf("BEV: nz=%d min=%.4f max=%.4f mean=%.6f\n", nz, mn, mx, sum/(256*200*176));
    delete[] h_bev;

    cudaFree(d_bev);
    cudaFree(d_features);
    cudaFree(d_coords);
    cudaStreamDestroy(stream);
    return 0;
}
