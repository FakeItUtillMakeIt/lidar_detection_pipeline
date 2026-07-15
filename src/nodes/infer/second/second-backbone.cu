#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <fstream>
#include <vector>
#include <cstring>
#include <nlohmann/json.hpp>

#include "second-backbone.hpp"
#include "common/check.hpp"

namespace pointpillar {
namespace lidar {

using json = nlohmann::json;

// Kernel offsets for 3×3×3 conv with padding 1 (centered): -1, 0, 1 per dim
static const int OFFSETS_3x3x3[27][3] = {
    {-1,-1,-1},{-1,-1,0},{-1,-1,1},{-1,0,-1},{-1,0,0},{-1,0,1},{-1,1,-1},{-1,1,0},{-1,1,1},
    {0,-1,-1},{0,-1,0},{0,-1,1},{0,0,-1},{0,0,0},{0,0,1},{0,1,-1},{0,1,0},{0,1,1},
    {1,-1,-1},{1,-1,0},{1,-1,1},{1,0,-1},{1,0,0},{1,0,1},{1,1,-1},{1,1,0},{1,1,1}
};

struct ConvLayer {
    bool is_subm;
    int in_channels, out_channels;
    int kv;
    int ksize[3];
    int stride[3];
    int center_offset[3]; // offset of center kernel position (from padding)
    int min_offset[3];    // min kernel offset
    int max_offset[3];    // max kernel offset
    int padding[3];       // padding from weight file header
    int corr[3];          // pad - k_center * dilation (correction per dim)
    int* kernel_offsets;  // [kv, 3] loaded from weight file

    float* weight_stacked;  // [kv, C_out, C_in]
    float* bias;
    float* bn_weight, *bn_bias, *bn_mean, *bn_var;
    float bn_eps;
    bool has_bn, has_relu;
};

// Sparse conv fused kernel (gather → inner product → scatter)
__global__ void sparse_conv_fused_kernel(
    const float* input, int num_in, int C_in,
    const float* weight_stacked, int C_out,
    const int* gather_map, int kv, int num_out,
    float* output)
{
    int i = blockIdx.x;
    if (i >= num_out) return;

    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    for (int co = tid; co < C_out; co += total_threads) {
        float sum = 0.0f;
        for (int k = 0; k < kv; k++) {
            int src = gather_map[k * num_out + i];
            if (src >= 0 && src < num_in) {
                float accum = 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    accum += weight_stacked[(k * C_out + co) * C_in + ci] *
                             input[src * C_in + ci];
                }
                sum += accum;
            }
        }
        output[i * C_out + co] = sum;
    }
}

__global__ void bn_relu_kernel(
    float* data, int num, int C,
    const float* gamma, const float* beta,
    const float* mean, const float* var, float eps, bool has_relu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num * C) return;
    int c = idx % C;
    float x = data[idx];
    float y = (x - mean[c]) / sqrtf(var[c] + eps) * gamma[c] + beta[c];
    data[idx] = (has_relu && y < 0.0f) ? 0.0f : y;
}

__global__ void add_bias_kernel(float* data, int num, int C, const float* bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num * C) return;
    data[idx] += bias[idx % C];
}

// SparseToDense + HeightCompression
__global__ void sparse_to_dense_kernel(
    const float* features, int num_points, int C,
    const unsigned int* coords,
    int D, int H, int W,
    float* bev)
{
    int i = blockIdx.x;
    if (i >= num_points) return;

    int z = coords[i * 4 + 1];
    int y = coords[i * 4 + 2];
    int x = coords[i * 4 + 3];
    if (z < 0 || z >= D || y < 0 || y >= H || x < 0 || x >= W) return;

    for (int c = 0; c < C; c++) {
        bev[c * D * H * W + z * H * W + y * W + x] = features[i * C + c];
    }
}

__device__ void atomicMaxFloat(float* addr, float val) {
    int* addr_int = (int*)addr;
    int old = *addr_int;
    int expected;
    do {
        expected = old;
        if (__int_as_float(expected) >= val) break;
        old = atomicCAS(addr_int, expected, __float_as_int(val));
    } while (expected != old);
}

// Sparse-to-dense with height compression: write max-over-z per channel to [C, H, W]
__global__ void sparse_to_dense_maxz_kernel(
    const float* features, int num_points, int C,
    const unsigned int* coords,
    int D, int H, int W,
    float* bev) // [C, H, W] pre-zeroed
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    int z = coords[i * 4 + 1];
    int y = coords[i * 4 + 2];
    int x = coords[i * 4 + 3];
    if (z < 0 || z >= D || y < 0 || y >= H || x < 0 || x >= W) return;

    for (int c = 0; c < C; c++) {
        float val = features[i * C + c];
        atomicMaxFloat(bev + c * H * W + y * W + x, val);
    }
}

// Build coord-to-index lookup grid: grid[z][y][x] = idx + 1 (0 = empty)
__global__ void build_coord_grid_kernel(
    const unsigned int* coords, int num_points,
    int* grid, int GZ, int GY, int GX)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    int z = (int)coords[i * 4 + 1];
    int y = (int)coords[i * 4 + 2];
    int x = (int)coords[i * 4 + 3];

    if (z >= 0 && z < GZ && y >= 0 && y < GY && x >= 0 && x < GX) {
        grid[(z * GY + y) * GX + x] = i + 1;
    }
}

// Build gather_map for SubM conv (stride = 1)
__global__ void build_gather_map_subm_kernel(
    const int* grid, int GZ, int GY, int GX,
    const int* kernel_offsets, int kv,
    int* gather_map, int num_out)
{
    int i = blockIdx.x;
    if (i >= num_out) return;

    // The i-th output voxel corresponds to the i-th input voxel (SubM: same coords)
    // We need to find input indices for each kernel neighbor of the i-th output voxel.
    // But we don't know the coordinates directly here — we need to reconstruct them.
    // Actually, for SubM, the output coordinates are the same as input coordinates.
    // The grid array is indexed by coordinate. For output index i, its coordinate is at
    // the position where grid[coord] == i + 1.
    // We DON'T have the output coords directly. But we stored the input coords — the
    // kernel is called with num_out = num_in, and output index i corresponds to input i.
    // The input coords are stored in the original coords array passed to forward.
    // However, we don't have access to that array here.
    //
    // Alternative: use the gridd array to find the coordinate for index i.
    // But that requires scanning the grid...
    //
    // Better approach: pass the input coords array to this kernel.
    // For SubM, output i has the same coords as input i.
    // So we read input_coords[i] to get (z, y, x), then compute neighbor coords.
}

// We need a version that takes coords directly:
__global__ void build_gather_map_subm_kernel_v2(
    const unsigned int* coords, int num_out,
    const int* grid, int GZ, int GY, int GX,
    const int* kernel_offsets, int kv,
    int* gather_map)
{
    int i = blockIdx.x;
    if (i >= num_out) return;

    int z = (int)coords[i * 4 + 1];
    int y = (int)coords[i * 4 + 2];
    int x = (int)coords[i * 4 + 3];

    for (int k = 0; k < kv; k++) {
        int nz = z + kernel_offsets[k * 3 + 0];
        int ny = y + kernel_offsets[k * 3 + 1];
        int nx = x + kernel_offsets[k * 3 + 2];

        if (nz >= 0 && nz < GZ && ny >= 0 && ny < GY && nx >= 0 && nx < GX) {
            int val = grid[(nz * GY + ny) * GX + nx];
            gather_map[k * num_out + i] = (val > 0) ? (val - 1) : -1;
        } else {
            gather_map[k * num_out + i] = -1;
        }
    }
}

// For strided conv: compute output coords from input coords + stride + offsets
// Then assign unique output indices via atomic dedup
__global__ void compute_strided_output_kernel(
    const unsigned int* in_coords, int num_in,
    int* out_coords,
    int* out_grid,
    unsigned int* num_out_ptr,
    int GZ_out, int GY_out, int GX_out,
    int sz, int sy, int sx,
    const int* kernel_offsets, int kv,
    int corr_z, int corr_y, int corr_x,
    int max_num_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_in) return;

    int iz = (int)in_coords[i * 4 + 1];
    int iy = (int)in_coords[i * 4 + 2];
    int ix = (int)in_coords[i * 4 + 3];

    for (int k = 0; k < kv; k++) {
        int dz = kernel_offsets[k * 3 + 0];
        int dy = kernel_offsets[k * 3 + 1];
        int dx = kernel_offsets[k * 3 + 2];

        // spconv formula: out = (in + pad - k * dilation) / stride
        // With centered offset dz = k - k_center:
        //   out = (in + pad - (dz + k_center) * dilation) / stride
        //        = (in - dz * dilation + (pad - k_center * dilation)) / stride
        // correction_i = pad_i - k_center_i * dilation_i
        // For dilation=1: corr = pad - (ksize-1)/2
        int tz = iz - dz + corr_z;
        int ty = iy - dy + corr_y;
        int tx = ix - dx + corr_x;

        if (tz < 0 || ty < 0 || tx < 0) continue;
        if (tz % sz != 0 || ty % sy != 0 || tx % sx != 0) continue;

        int oz = tz / sz;
        int oy = ty / sy;
        int ox = tx / sx;

        if (oz >= GZ_out || oy >= GY_out || ox >= GX_out) continue;

        int out_flat = (oz * GY_out + oy) * GX_out + ox;
        int old = atomicCAS(&out_grid[out_flat], 0, -1);
        if (old == 0) {
            unsigned int idx = atomicAdd(num_out_ptr, 1);
            if (idx >= (unsigned int)max_num_out) return;
            out_grid[out_flat] = -((int)idx + 1);
            out_coords[idx * 4 + 0] = 0;
            out_coords[idx * 4 + 1] = oz;
            out_coords[idx * 4 + 2] = oy;
            out_coords[idx * 4 + 3] = ox;
        }
    }
}

// Convert output grid to positive indices
__global__ void finalize_output_grid_kernel(int* out_grid, int GZ, int GY, int GX, unsigned int num_out)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= GZ * GY * GX) return;
    int val = out_grid[flat];
    if (val < 0) {
        out_grid[flat] = -val - 1;  // convert back to positive index
    }
}

// Build gather_map for strided conv
__global__ void build_gather_map_strided_kernel(
    const int* in_grid, int GZ_in, int GY_in, int GX_in,
    const int* out_grid, int GZ_out, int GY_out, int GX_out,
    const int* kernel_offsets, int kv,
    int sz, int sy, int sx,
    int* gather_map, unsigned int num_out)
{
    int i = blockIdx.x;
    if (i >= (int)num_out) return;

    // We need the output coord for index i. It's stored in out_grid.
    // But out_grid maps coord → index, not index → coord.
    // We need another array: out_coords[i] = (0, oz, oy, ox)
    // This should be passed to the kernel.
}

// The above kernel needs out_coords. Let me write one that takes out_coords:
__global__ void build_gather_map_strided_kernel_v2(
    const unsigned int* out_coords, unsigned int num_out,
    const int* in_grid, int GZ_in, int GY_in, int GX_in,
    const int* kernel_offsets, int kv,
    int sz, int sy, int sx,
    int corr_z, int corr_y, int corr_x,
    int* gather_map)
{
    int i = blockIdx.x;
    if (i >= (int)num_out) return;

    int oz = (int)out_coords[i * 4 + 1];
    int oy = (int)out_coords[i * 4 + 2];
    int ox = (int)out_coords[i * 4 + 3];

    for (int k = 0; k < kv; k++) {
        int dz = kernel_offsets[k * 3 + 0];
        int dy = kernel_offsets[k * 3 + 1];
        int dx = kernel_offsets[k * 3 + 2];

        // Input coord derived from forward formula:
        // forward: out * stride = in + corr - dz  →  in = out * stride + dz - corr
        int iz = oz * sz + dz - corr_z;
        int iy = oy * sy + dy - corr_y;
        int ix = ox * sx + dx - corr_x;

        if (iz >= 0 && iz < GZ_in && iy >= 0 && iy < GY_in && ix >= 0 && ix < GX_in) {
            int val = in_grid[(iz * GY_in + iy) * GX_in + ix];
            gather_map[k * num_out + i] = (val > 0) ? (val - 1) : -1;
        } else {
            gather_map[k * num_out + i] = -1;
        }
    }
}


__global__ void apply_relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (data[idx] < 0) data[idx] = 0;
}

// ============================================================
// SecondBackbone implementation
// ============================================================
class SecondBackboneImpl : public SecondBackbone {
public:
    ~SecondBackboneImpl() override {
        for (auto& l : layers_) {
            cudaFree(l.weight_stacked);
            cudaFree(l.bias);
            cudaFree(l.bn_weight);
            cudaFree(l.bn_bias);
            cudaFree(l.bn_mean);
            cudaFree(l.bn_var);
            cudaFree(l.kernel_offsets);
        }
        if (temp_input_) cudaFree(temp_input_);
        if (temp_output_) cudaFree(temp_output_);
        if (coord_grid_) cudaFree(coord_grid_);
        if (gather_map_tmp_) cudaFree(gather_map_tmp_);
        if (out_coords_tmp_) cudaFree(out_coords_tmp_);
        if (out_grid_tmp_) cudaFree(out_grid_tmp_);
        if (num_out_tmp_) cudaFree(num_out_tmp_);
    }

    bool load_weights(const std::string& path) override {
        std::ifstream f(path, std::ios::binary);
        if (!f) { printf("[ERROR] Cannot open %s\n", path.c_str()); return false; }

        uint32_t header_len;
        f.read((char*)&header_len, sizeof(header_len));

        std::string header_json(header_len, '\0');
        f.read(&header_json[0], header_len);
        auto header = json::parse(header_json);
        int64_t data_base = (int64_t)sizeof(header_len) + header_len;

        layers_.resize(header["num_layers"]);
        max_num_out_ = 0;
        max_c_ = 0;

        for (int li = 0; li < (int)layers_.size(); li++) {
            auto& info = header["layer_info"][li];
            auto& layer = layers_[li];

            layer.is_subm = info["is_subm"];
            layer.in_channels = info["in_channels"];
            layer.out_channels = info["out_channels"];
            layer.kv = info["kv"];
            layer.has_bn = info["has_bn"];
            layer.has_relu = info["has_relu"];
            layer.bn_eps = info["bn_eps"];

            layer.stride[0] = info["stride"][0];
            layer.stride[1] = info["stride"][1];
            layer.stride[2] = info["stride"][2];

            // Kernel size from header
            layer.ksize[0] = info["kernel_size"][0];
            layer.ksize[1] = info["kernel_size"][1];
            layer.ksize[2] = info["kernel_size"][2];
            int kz = layer.ksize[0], ky = layer.ksize[1], kx = layer.ksize[2];

            // Load kernel offsets
            auto& offsets = info["kernel_offsets"];
            int* host_offsets = new int[layer.kv * 3];
            for (int k = 0; k < layer.kv; k++) {
                host_offsets[k * 3 + 0] = offsets[k][0];
                host_offsets[k * 3 + 1] = offsets[k][1];
                host_offsets[k * 3 + 2] = offsets[k][2];
            }
            // Normalize offsets to centered format: spconv stores 0-indexed (0..k-1)
            // for dims with pad=0, but centered (-k_center..k_center) for pad>0.
            // Centered offsets are required by our formula and SubM kernel.
            int kc_z = kz / 2, kc_y = ky / 2, kc_x = kx / 2;
            for (int dim = 0; dim < 3; dim++) {
                int kc = (dim == 0) ? kc_z : ((dim == 1) ? kc_y : kc_x);
                if (kc == 0) continue;
                bool is_zero_indexed = true;
                for (int k = 0; k < layer.kv && is_zero_indexed; k++)
                    if (host_offsets[k * 3 + dim] < 0) is_zero_indexed = false;
                if (is_zero_indexed) {
                    for (int k = 0; k < layer.kv; k++)
                        host_offsets[k * 3 + dim] -= kc;
                }
            }
            checkRuntime(cudaMalloc(&layer.kernel_offsets, layer.kv * 3 * sizeof(int)));
            checkRuntime(cudaMemcpy(layer.kernel_offsets, host_offsets,
                                     layer.kv * 3 * sizeof(int), cudaMemcpyHostToDevice));

            // Compute center offset from kernel size (z-major order)
            int center_k = (kz/2) * ky * kx + (ky/2) * kx + (kx/2);
            layer.center_offset[0] = host_offsets[center_k * 3 + 0];
            layer.center_offset[1] = host_offsets[center_k * 3 + 1];
            layer.center_offset[2] = host_offsets[center_k * 3 + 2];
            // Compute min/max offsets from kernel size: offset range = center ± (ksize-1)/2
            int delta_z = (kz - 1) / 2, delta_y = (ky - 1) / 2, delta_x = (kx - 1) / 2;
            layer.min_offset[0] = layer.center_offset[0] - delta_z;
            layer.min_offset[1] = layer.center_offset[1] - delta_y;
            layer.min_offset[2] = layer.center_offset[2] - delta_x;
            layer.max_offset[0] = layer.center_offset[0] + delta_z;
            layer.max_offset[1] = layer.center_offset[1] + delta_y;
            layer.max_offset[2] = layer.center_offset[2] + delta_x;
            // spconv formula with CENTERED offsets dz: out = (in + pad - dz) / stride
            // Using centered formula: tz = iz - dz + corr, where corr = pad
            // (NOT pad - k_center: the centered dz already accounts for the offset center)
            auto& pad = info["padding"];
            layer.padding[0] = (int)pad[0]; layer.padding[1] = (int)pad[1]; layer.padding[2] = (int)pad[2];
            layer.corr[0] = (int)pad[0];
            layer.corr[1] = (int)pad[1];
            layer.corr[2] = (int)pad[2];
            delete[] host_offsets;

            // Skip pre-computed gather_map — we compute at runtime
            // But we still need to advance the file pointer past it
            int skip_gather_size = info["gather_map_size"];
            int skip_oc_size = info["out_coords_size"];

            // Load weights
            layer.weight_stacked = load_array<float>(f, data_base, info["weight_stacked_offset"], info["weight_stacked_size"]);

            if (info["has_bias"]) {
                layer.bias = load_array<float>(f, data_base, info["bias_offset"], info["bias_size"]);
            } else {
                layer.bias = nullptr;
            }

            if (layer.has_bn) {
                layer.bn_weight = load_array<float>(f, data_base, info["bn_weight_offset"], info["bn_weight_size"]);
                layer.bn_bias = load_array<float>(f, data_base, info["bn_bias_offset"], info["bn_bias_size"]);
                layer.bn_mean = load_array<float>(f, data_base, info["bn_mean_offset"], info["bn_mean_size"]);
                layer.bn_var = load_array<float>(f, data_base, info["bn_var_offset"], info["bn_var_size"]);
            } else {
                layer.bn_weight = layer.bn_bias = layer.bn_mean = layer.bn_var = nullptr;
            }

            max_num_out_ = std::max(max_num_out_, (int)info["num_out"]);

            // Also track max possible output for runtime: use input max_voxels
            // For SubM: max_out = max_in = the number of input voxels (up to max_voxels)
            // For strided: max_out ≤ max_in
            // We'll use max_voxels as an upper bound
            max_c_ = std::max(max_c_, std::max(layer.in_channels, layer.out_channels));
        }

        max_possible_out_ = 300000;  // max outputs across all layers (strided conv can expand)

        // Allocate temp buffers
        checkRuntime(cudaMalloc(&temp_input_,  (size_t)max_possible_out_ * max_c_ * sizeof(float)));
        checkRuntime(cudaMalloc(&temp_output_, (size_t)max_possible_out_ * max_c_ * sizeof(float)));

        // Allocate coord_grid for the largest spatial shape: [41, 1600, 1408]
        max_grid_z_ = 41;
        max_grid_y_ = 1600;
        max_grid_x_ = 1408;
        size_t grid_vol = (size_t)max_grid_z_ * max_grid_y_ * max_grid_x_;
        checkRuntime(cudaMalloc(&coord_grid_, grid_vol * sizeof(int)));
        checkRuntime(cudaMemset(coord_grid_, 0, grid_vol * sizeof(int)));

        // Allocate output grid (same max size)
        checkRuntime(cudaMalloc(&out_grid_tmp_, grid_vol * sizeof(int)));

        // Allocate gather_map and out_coords temp buffers
        int max_kv = 27;
        max_possible_out_ = 300000;
        checkRuntime(cudaMalloc(&gather_map_tmp_, (size_t)max_kv * max_possible_out_ * sizeof(int)));
        checkRuntime(cudaMalloc(&out_coords_tmp_, (size_t)max_possible_out_ * 4 * sizeof(int)));
        checkRuntime(cudaMalloc(&num_out_tmp_, sizeof(unsigned int)));

        // Buffers for L10 (conv4_2) output save (used for BEV concatenation with L11)
        int l10_channels = 128;
        checkRuntime(cudaMalloc(&l10_saved_feats_, (size_t)max_possible_out_ * l10_channels * sizeof(float)));
        checkRuntime(cudaMalloc(&l10_saved_coords_, (size_t)max_possible_out_ * 4 * sizeof(unsigned int)));

        return true;
    }

    void forward(
        const float* voxel_features,
        const unsigned int* coords,
        unsigned int num_voxels,
        float* bev_features,
        void* stream) override
    {
        cudaStream_t _stream = (cudaStream_t)stream;

                const float* cur_input = voxel_features;
        unsigned int num_cur = num_voxels;
        const unsigned int* cur_coords = coords;
        int layer_idx = 0;
        int cur_GZ = max_grid_z_, cur_GY = max_grid_y_, cur_GX = max_grid_x_;

        for (auto& layer : layers_) {
            int C_in = layer.in_channels;
            int C_out = layer.out_channels;
            int kv = layer.kv;

            // Get spatial shape for this layer from input coords
            int GZ, GY, GX;
            // For the first layer, use max_grid dimensions.
            // For subsequent layers, the spatial shape changes.
            // We need to know the spatial shape for each layer.
            // We can derive it from the layer's kernel_offsets: the max absolute offset
            // gives us the padding, and we can compute the spatial shape.
            // Actually, let me use the approach: compute from max coords.
            // The coord range is determined by the current coords' max values.

            // Determine spatial shape from input coords
            // For efficiency, we track max coord values
            // Actually, we need the spatial shape for the coord grid allocation.
            // Since the grid is allocated for the max size (41,1600,1408), we can
            // reuse it for all layers. The grid dimensions for each layer should be
            // <= the max grid.

            // For SubM: input and output shapes are the same
            // For strided: output shape is (input_shape + 2*pad - kernel) / stride + 1

            // Let me hardcode the expected shapes from the weight header:
            if (layer.is_subm) {
                // Same spatial shape
                if (num_cur > max_possible_out_) {
                    printf("[ERROR] num_cur %d > max_possible_out_ %d\n", num_cur, max_possible_out_);
                    return;
                }

                // Clear coord_grid
                size_t grid_bytes = (size_t)max_grid_z_ * max_grid_y_ * max_grid_x_ * sizeof(int);
                checkRuntime(cudaMemsetAsync(coord_grid_, 0, grid_bytes, _stream));

                // Build coord_grid: (z, y, x) → index
                int block_size = 256;
                int grid_size = (num_cur + 255) / 256;
                build_coord_grid_kernel<<<grid_size, block_size, 0, _stream>>>(
                    cur_coords, num_cur, coord_grid_, max_grid_z_, max_grid_y_, max_grid_x_);

                // Build gather_map for SubM
                grid_size = num_cur;
                build_gather_map_subm_kernel_v2<<<grid_size, block_size, 0, _stream>>>(
                    cur_coords, num_cur,
                    coord_grid_, max_grid_z_, max_grid_y_, max_grid_x_,
                    layer.kernel_offsets, kv,
                    gather_map_tmp_);

                // Run conv
                grid_size = num_cur;
                sparse_conv_fused_kernel<<<grid_size, block_size, 0, _stream>>>(
                    cur_input, num_cur, C_in,
                    layer.weight_stacked, C_out,
                    gather_map_tmp_, kv, num_cur,
                    temp_output_);

                // Debug: check pre-BN output for layers 0, 1, 2
                if (layer_idx <= 2) {
                    checkRuntime(cudaStreamSynchronize(_stream));
                    float* dbg = new float[num_cur * C_out];
                    checkRuntime(cudaMemcpy(dbg, temp_output_, num_cur * C_out * sizeof(float), cudaMemcpyDeviceToHost));
                    printf("[DBG] L%d pre-BN: min=%.4f max=%.4f mean=%.6f nz=%d/%zu\n", layer_idx,
                           *std::min_element(dbg, dbg+num_cur*C_out),
                           *std::max_element(dbg, dbg+num_cur*C_out),
                           std::accumulate(dbg, dbg+num_cur*C_out, 0.0)/(num_cur*C_out),
                           (int)std::count_if(dbg, dbg+num_cur*C_out, [](float v){return v!=0;}),
                           (size_t)num_cur*C_out);
                    if (num_cur > 0) {
                        printf("[DBG] L%d pre-BN vox[0]:", layer_idx);
                        for (int c = 0; c < std::min(16, C_out); c++) printf(" %.4f", dbg[c]);
                        printf("\n");
                    }
                    delete[] dbg;
                }
                // Bias, BN, ReLU
                if (layer.bias) {
                    int vol = num_cur * C_out;
                    add_bias_kernel<<<(vol + 255) / 256, 256, 0, _stream>>>(
                        temp_output_, num_cur, C_out, layer.bias);
                }

                if (layer.has_bn) {
                    int vol = num_cur * C_out;
                    bn_relu_kernel<<<(vol + 255) / 256, 256, 0, _stream>>>(
                        temp_output_, num_cur, C_out,
                        layer.bn_weight, layer.bn_bias,
                        layer.bn_mean, layer.bn_var, layer.bn_eps, layer.has_relu);
                } else if (layer.has_relu) {
                    int vol = num_cur * C_out;
                    apply_relu_kernel<<<(vol + 255) / 256, 256, 0, _stream>>>(
                        temp_output_, num_cur * C_out);
                }

                // Debug: check post-BN output for layer 0
                if (layer_idx == 0) {
                    checkRuntime(cudaStreamSynchronize(_stream));
                    float* dbg = new float[num_cur * C_out];
                    checkRuntime(cudaMemcpy(dbg, temp_output_, num_cur * C_out * sizeof(float), cudaMemcpyDeviceToHost));
                    printf("[DBG] L%d post-BN: min=%.4f max=%.4f mean=%.6f nz=%d/%zu\n", layer_idx,
                           *std::min_element(dbg, dbg+num_cur*C_out),
                           *std::max_element(dbg, dbg+num_cur*C_out),
                           std::accumulate(dbg, dbg+num_cur*C_out, 0.0)/(num_cur*C_out),
                           (int)std::count_if(dbg, dbg+num_cur*C_out, [](float v){return v!=0;}),
                           (size_t)num_cur*C_out);
                    if (num_cur > 0) {
                        printf("[DBG] L%d post-BN vox[0]:", layer_idx);
                        for (int c = 0; c < std::min(16, C_out); c++) printf(" %.4f", dbg[c]);
                        printf("\n");
                    }
                    delete[] dbg;
                }

                cur_input = temp_output_;
                // cur_coords stays the same for SubM

            } else {
                // Strided conv
                int sz = layer.stride[0];
                int sy = layer.stride[1];
                int sx = layer.stride[2];

                // Compute output spatial shape from INPUT GRID SIZE (not max coord!)
                // out = (in + 2*pad - dilation*(ksize-1) - 1) / stride + 1
                int GZ_out = (cur_GZ + 2*layer.padding[0] - (layer.ksize[0]-1) - 1) / sz + 1;
                int GY_out = (cur_GY + 2*layer.padding[1] - (layer.ksize[1]-1) - 1) / sy + 1;
                int GX_out = (cur_GX + 2*layer.padding[2] - (layer.ksize[2]-1) - 1) / sx + 1;

                // Clear grids
                size_t grid_in_bytes = (size_t)max_grid_z_ * max_grid_y_ * max_grid_x_ * sizeof(int);
                checkRuntime(cudaMemsetAsync(coord_grid_, 0, grid_in_bytes, _stream));
                size_t grid_out_bytes = (size_t)max_grid_z_ * max_grid_y_ * max_grid_x_ * sizeof(int);
                checkRuntime(cudaMemsetAsync(out_grid_tmp_, 0, grid_out_bytes, _stream));
                checkRuntime(cudaMemsetAsync(num_out_tmp_, 0, sizeof(unsigned int), _stream));

                // Build input coord_grid
                int block_size = 256;
                int grid_size = (num_cur + 255) / 256;
                build_coord_grid_kernel<<<grid_size, block_size, 0, _stream>>>(
                    cur_coords, num_cur, coord_grid_, max_grid_z_, max_grid_y_, max_grid_x_);

                // Compute output coords and assign IDs
                grid_size = (num_cur + 255) / 256;
                compute_strided_output_kernel<<<grid_size, block_size, 0, _stream>>>(
                    cur_coords, num_cur,
                    out_coords_tmp_, out_grid_tmp_, num_out_tmp_,
                    GZ_out, GY_out, GX_out,
                    sz, sy, sx,
                    layer.kernel_offsets, kv,
                    layer.corr[0], layer.corr[1], layer.corr[2],
                    max_possible_out_);

                unsigned int num_out_host = 0;
                checkRuntime(cudaMemcpyAsync(&num_out_host, num_out_tmp_, sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost, _stream));
                checkRuntime(cudaStreamSynchronize(_stream));

                if (num_out_host == 0) {
                    printf("[WARN] Strided conv produced 0 output voxels\n");
                    num_cur = 0;
                    continue;
                }

                // Finalize output grid (convert negative indices to positive)
                size_t out_grid_flat = (size_t)max_grid_z_ * max_grid_y_ * max_grid_x_;
                grid_size = (out_grid_flat + 255) / 256;
                finalize_output_grid_kernel<<<grid_size, block_size, 0, _stream>>>(
                    out_grid_tmp_, max_grid_z_, max_grid_y_, max_grid_x_, num_out_host);

                // Build gather_map for strided conv
                grid_size = num_out_host;
                build_gather_map_strided_kernel_v2<<<grid_size, block_size, 0, _stream>>>(
                    (const unsigned int*)out_coords_tmp_, num_out_host,
                    coord_grid_, max_grid_z_, max_grid_y_, max_grid_x_,
                    layer.kernel_offsets, kv,
                    sz, sy, sx,
                    layer.corr[0], layer.corr[1], layer.corr[2],
                    gather_map_tmp_);

                // Run conv
                grid_size = num_out_host;
                if (grid_size > max_possible_out_) grid_size = max_possible_out_;

                sparse_conv_fused_kernel<<<grid_size, block_size, 0, _stream>>>(
                    cur_input, num_cur, C_in,
                    layer.weight_stacked, C_out,
                    gather_map_tmp_, kv, num_out_host,
                    temp_output_);

                // Debug: check pre-BN output for strided layers
                {
                    checkRuntime(cudaStreamSynchronize(_stream));
                    // Print z-distribution
                    unsigned int* host_oc = new unsigned int[num_out_host * 4];
                    checkRuntime(cudaMemcpy(host_oc, out_coords_tmp_, num_out_host * 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
                    int zhist[10] = {0};
                    int ymin = 99999, ymax = 0, xmin = 99999, xmax = 0;
                    for (unsigned int oi = 0; oi < num_out_host; oi++) {
                        int z = (int)host_oc[oi*4+1];
                        int y = (int)host_oc[oi*4+2];
                        int x = (int)host_oc[oi*4+3];
                        if (z < 10) zhist[z]++;
                        ymin = std::min(ymin, y); ymax = std::max(ymax, y);
                        xmin = std::min(xmin, x); xmax = std::max(xmax, x);
                    }
                    printf("[DBG] L%d strided: %d voxels, GZ=%d GY=%d GX=%d\n", layer_idx,
                           num_out_host, GZ_out, GY_out, GX_out);
                    printf("[DBG] L%d z-dist: z0=%d z1=%d z2=%d z3=%d z4=%d z5=%d z6=%d z7=%d z8=%d z9=%d\n", layer_idx,
                           zhist[0], zhist[1], zhist[2], zhist[3], zhist[4],
                           zhist[5], zhist[6], zhist[7], zhist[8], zhist[9]);
                    printf("[DBG] L%d yrange=[%d,%d] xrange=[%d,%d]\n", layer_idx, ymin, ymax, xmin, xmax);
                    delete[] host_oc;
                    
                    float* dbg = new float[num_out_host * C_out];
                    checkRuntime(cudaMemcpy(dbg, temp_output_, num_out_host * C_out * sizeof(float), cudaMemcpyDeviceToHost));
                    printf("[DBG] L%d strided pre-BN: min=%.4f max=%.4f mean=%.6f nz=%d/%zu\n", layer_idx,
                            *std::min_element(dbg, dbg+num_out_host*C_out),
                            *std::max_element(dbg, dbg+num_out_host*C_out),
                            std::accumulate(dbg, dbg+num_out_host*C_out, 0.0)/(num_out_host*C_out),
                            (int)std::count_if(dbg, dbg+num_out_host*C_out, [](float v){return v!=0;}),
                            (size_t)num_out_host*C_out);
                    if (num_out_host > 0) {
                        printf("[DBG] L%d strided pre-BN vox[0]:", layer_idx);
                        for (int c = 0; c < std::min(8, C_out); c++) printf(" %.4f", dbg[c]);
                        printf("\n");
                    }
                    delete[] dbg;
                }

                // Bias
                if (layer.bias) {
                    int vol = num_out_host * C_out;
                    add_bias_kernel<<<(vol + 255) / 256, 256, 0, _stream>>>(
                        temp_output_, num_out_host, C_out, layer.bias);
                }

                // BN + ReLU
                if (layer.has_bn) {
                    int vol = num_out_host * C_out;
                    bn_relu_kernel<<<(vol + 255) / 256, 256, 0, _stream>>>(
                        temp_output_, num_out_host, C_out,
                        layer.bn_weight, layer.bn_bias,
                        layer.bn_mean, layer.bn_var, layer.bn_eps, layer.has_relu);
                } else if (layer.has_relu) {
                    int vol = num_out_host * C_out;
                    apply_relu_kernel<<<(vol + 255) / 256, 256, 0, _stream>>>(
                        temp_output_, num_out_host * C_out);
                }

                cur_input = temp_output_;
                cur_coords = (const unsigned int*)out_coords_tmp_;
                num_cur = num_out_host;
                // Update spatial grid for next layer
                cur_GZ = GZ_out; cur_GY = GY_out; cur_GX = GX_out;
            }

            // Save L10 (conv4_2) output for BEV concatenation
            if (layer_idx == 9) {
                checkRuntime(cudaMemcpyAsync(l10_saved_feats_, cur_input,
                    (size_t)num_cur * 128 * sizeof(float), cudaMemcpyDeviceToDevice, _stream));
                checkRuntime(cudaMemcpyAsync(l10_saved_coords_, cur_coords,
                    (size_t)num_cur * 4 * sizeof(unsigned int), cudaMemcpyDeviceToDevice, _stream));
                l10_saved_num_ = num_cur;
            }

            layer_idx++;
            std::swap(temp_input_, temp_output_);
        }

        cudaStreamSynchronize(_stream);
        // SparseToDense + HeightCompression
        int H = 200, W = 176;

        // BEV: concat max-over-z of L10 (128ch) and L11 (128ch) → 256ch
        checkRuntime(cudaMemsetAsync(bev_features, 0, (size_t)256 * H * W * sizeof(float), _stream));

        // Height-compress L10 (conv4_2 SubM output, grid [5,200,176])
        if (l10_saved_num_ > 0) {
            int block_size = 256;
            int grid = (l10_saved_num_ + block_size - 1) / block_size;
            sparse_to_dense_maxz_kernel<<<grid, block_size, 0, _stream>>>(
                l10_saved_feats_, l10_saved_num_, 128,
                l10_saved_coords_, 5, H, W,
                bev_features);
        }

        // Height-compress L11 (conv_out output, grid [2,200,176])
        {
            int block_size = 256;
            int grid = (num_cur + block_size - 1) / block_size;
            sparse_to_dense_maxz_kernel<<<grid, block_size, 0, _stream>>>(
                cur_input, num_cur, 128,
                cur_coords, 2, H, W,
                bev_features + 128 * H * W);
        }
    }

    void print() override {
        printf("[SecondBackbone] %zu layers\n", layers_.size());
        for (size_t i = 0; i < layers_.size(); i++) {
            auto& l = layers_[i];
            printf("  Layer %zu: %s %d->%d kv=%d stride=[%d,%d,%d] BN=%d ReLU=%d\n",
                   i, l.is_subm ? "SubM" : "Sparse",
                   l.in_channels, l.out_channels, l.kv,
                   l.stride[0], l.stride[1], l.stride[2],
                   (int)l.has_bn, (int)l.has_relu);
        }
    }

private:
    template<typename T>
    T* load_array(std::ifstream& f, int64_t data_base, int64_t offset, int size) {
        T* ptr;
        checkRuntime(cudaMalloc(&ptr, size));
        std::vector<char> buf(size);
        f.seekg(data_base + offset, std::ios::beg);
        f.read(buf.data(), size);
        checkRuntime(cudaMemcpy(ptr, buf.data(), size, cudaMemcpyHostToDevice));
        return ptr;
    }

    std::vector<ConvLayer> layers_;
    float *temp_input_ = nullptr, *temp_output_ = nullptr;
    int *coord_grid_ = nullptr;        // dense coord-to-index grid
    int *gather_map_tmp_ = nullptr;    // temp gather_map [kv, max_out]
    int *out_coords_tmp_ = nullptr;    // temp out_coords [max_out, 4]
    int *out_grid_tmp_ = nullptr;      // temp output grid for strided dedup
    unsigned int* num_out_tmp_ = nullptr;
    int max_num_out_ = 0;
    int max_possible_out_ = 0;
    int max_c_ = 0;
    int max_grid_z_ = 0, max_grid_y_ = 0, max_grid_x_ = 0;
    float* l10_saved_feats_ = nullptr;
    unsigned int* l10_saved_coords_ = nullptr;
    int l10_saved_num_ = 0;
};

std::shared_ptr<SecondBackbone> create_second_backbone(const std::string& weights_path) {
    auto impl = std::make_shared<SecondBackboneImpl>();
    if (!impl->load_weights(weights_path)) {
        impl.reset();
    }
    return impl;
}

}  // namespace lidar
}  // namespace pointpillar
