#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) printf("[TRT] %s\n", msg);
    }
};

int main(int argc, char** argv) {
    const char* plan_file = argc > 1 ? argv[1] : "../model/second_2d_backbone.plan";
    printf("Testing engine: %s\n", plan_file);

    // Read plan
    std::ifstream f(plan_file, std::ios::binary | std::ios::ate);
    if (!f) { printf("Cannot open %s\n", plan_file); return 1; }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<char> data(size);
    f.read(data.data(), size);
    printf("Plan size: %zu bytes\n", size);

    // Create TRT objects
    Logger logger;
    IRuntime* runtime = createInferRuntime(logger);
    if (!runtime) { printf("createInferRuntime failed\n"); return 1; }
    printf("Runtime created\n");

    ICudaEngine* engine = runtime->deserializeCudaEngine(data.data(), size);
    if (!engine) { printf("deserializeCudaEngine failed\n"); return 1; }
    printf("Engine deserialized. IOs: %d\n", engine->getNbIOTensors());

    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        TensorIOMode mode = engine->getTensorIOMode(name);
        DataType dtype = engine->getTensorDataType(name);
        Dims dims = engine->getTensorShape(name);
        printf("  IO[%d]: %s mode=%s dtype=%d shape=[", i, name,
               mode == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT", (int)dtype);
        for (int j = 0; j < dims.nbDims; j++) printf("%s%ld", j ? "," : "", (long)dims.d[j]);
        printf("]\n");
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) { printf("createExecutionContext failed\n"); return 1; }
    printf("Context created\n");

    // Allocate device memory
    cudaSetDevice(0);
    float *d_bev, *d_cls, *d_box, *d_dir;
    cudaMalloc(&d_bev, 1*256*200*176*sizeof(float));
    cudaMalloc(&d_cls, 1*200*176*18*sizeof(float));
    cudaMalloc(&d_box, 1*200*176*42*sizeof(float));
    cudaMalloc(&d_dir, 1*200*176*12*sizeof(float));
    printf("Device memory allocated\n");

    // Set tensor addresses
    context->setTensorAddress("bev_features", d_bev);
    context->setTensorAddress("cls", d_cls);
    context->setTensorAddress("box", d_box);
    context->setTensorAddress("dir", d_dir);
    printf("Tensor addresses set\n");

    // Create stream and fill input with non-zero data
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<float> host_bev(1*256*200*176, 1.0f);
    cudaMemcpy(d_bev, host_bev.data(), host_bev.size()*sizeof(float), cudaMemcpyHostToDevice);
    printf("Input data uploaded\n");

    printf("Running enqueueV3...\n");
    bool ok = context->enqueueV3(stream);
    printf("enqueueV3 returned: %d\n", ok);

    cudaStreamSynchronize(stream);
    printf("Stream synced\n");

    // Check outputs
    std::vector<float> cls(1*200*176*18);
    std::vector<float> box(1*200*176*42);
    std::vector<float> dir(1*200*176*12);
    cudaMemcpy(cls.data(), d_cls, cls.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(box.data(), d_box, box.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dir.data(), d_dir, dir.size()*sizeof(float), cudaMemcpyDeviceToHost);

    auto stats = [](const std::vector<float>& v, const char* name) {
        float mn=v[0], mx=v[0], sum=0;
        for (auto x : v) { mn = std::min(mn, x); mx = std::max(mx, x); sum += x; }
        printf("%s: %.2fM min=%.4f max=%.4f mean=%.6f\n", name, v.size()/1e6, mn, mx, sum/v.size());
    };
    stats(cls, "cls");
    stats(box, "box");
    stats(dir, "dir");

    cudaStreamDestroy(stream);
    cudaFree(d_bev);
    cudaFree(d_cls);
    cudaFree(d_box);
    cudaFree(d_dir);
    delete context;
    delete engine;
    delete runtime;
    printf("Done.\n");
    return 0;
}
