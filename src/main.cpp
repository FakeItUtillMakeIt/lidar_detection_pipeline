// src/main.cpp
#include <iostream>
#include <string>
#include <csignal>
#include <fstream>

#include <nlohmann/json.hpp>

#include "lidar_core/core/pipeline.h"
#include "lidar_core/core/async_pipeline.h"
#include "lidar_core/nodes/i_source_node.h"
#include "lidar_core/nodes/i_infer_node.h"
#include "lidar_core/nodes/i_output_node.h"
#include "nodes/source/bin_source_node.h"

static volatile bool g_running = true;

static void signalHandler(int) { g_running = false; }

static void printHelp() {
    std::cout << "Usage: lidar_app [options]\n"
              << "Options:\n"
              << "  --config <path>    JSON config file path\n"
              << "  --async            Enable async pipeline (reader/infer in parallel)\n"
              << "  --queue <size>     Async queue size (default: 3)\n"
              << "  --help             Show this help\n";
}

// 同步模式
static int runSync(const nlohmann::json& config) {
    std::string pipeline_id = config["pipeline"]["id"];
    auto pipeline = std::make_shared<lidar_core::core::Pipeline>(pipeline_id);

    if (!pipeline->buildFromJson(config)) {
        std::cerr << "Error: Failed to build pipeline" << std::endl;
        return 1;
    }

    if (!pipeline->start()) {
        std::cerr << "Error: Failed to start pipeline" << std::endl;
        return 1;
    }

    std::cout << "Pipeline ready (sync mode). Processing..." << std::endl;

    auto source_node = std::dynamic_pointer_cast<lidar_core::nodes::ISourceNode>(
        pipeline->getNode("source"));
    auto infer_node = std::dynamic_pointer_cast<lidar_core::nodes::IInferNode>(
        pipeline->getNode("infer"));

    if (!source_node || !infer_node) {
        std::cerr << "Error: Missing required nodes" << std::endl;
        pipeline->stop();
        return 1;
    }

    uint64_t frame_count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    while (g_running && pipeline->isRunning()) {
        auto cloud_packet = std::make_shared<lidar_core::core::PointCloudPacket>();
        if (!std::dynamic_pointer_cast<lidar_core::nodes::BinSourceNode>(source_node)->readNext(cloud_packet)) {
            break;
        }
        infer_node->pushData(cloud_packet);
        frame_count++;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\nDone. Processed " << frame_count << " frames in " << total_sec << "s";
    if (total_sec > 0) {
        std::cout << " (" << (frame_count / total_sec) << " FPS)";
    }
    std::cout << std::endl;

    pipeline->stop();
    return 0;
}

// 异步模式
static int runAsync(const nlohmann::json& config, size_t queue_size) {
    std::string pipeline_id = config["pipeline"]["id"];
    auto async_pipeline = std::make_shared<lidar_core::core::AsyncPipeline>(pipeline_id, queue_size);

    if (!async_pipeline->buildFromJson(config)) {
        std::cerr << "Error: Failed to build async pipeline" << std::endl;
        return 1;
    }

    if (!async_pipeline->start()) {
        std::cerr << "Error: Failed to start async pipeline" << std::endl;
        return 1;
    }

    std::cout << "Pipeline ready (async mode, queue=" << queue_size << "). Processing..." << std::endl;

    // 等待处理完成
    while (g_running && async_pipeline->isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    async_pipeline->stop();

    uint64_t frames = async_pipeline->getProcessedFrames();
    double fps = async_pipeline->getFPS();

    std::cout << "\nDone. Processed " << frames << " frames";
    if (fps > 0) {
        std::cout << " (" << fps << " FPS)";
    }
    std::cout << std::endl;

    return 0;
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);

    std::string config_path = "../config/pipeline.json";
    bool async_mode = false;
    size_t queue_size = 3;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--async") == 0) {
            async_mode = true;
        } else if (strcmp(argv[i], "--queue") == 0 && i + 1 < argc) {
            queue_size = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printHelp();
            return 0;
        }
    }

    // Read config file
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Error: Cannot open config file: " << config_path << std::endl;
        return 1;
    }

    nlohmann::json config;
    try {
        config_file >> config;
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid JSON config: " << e.what() << std::endl;
        return 1;
    }

    if (!config.contains("pipeline") || !config["pipeline"].contains("id")) {
        std::cerr << "Error: Missing 'pipeline.id' in config" << std::endl;
        return 1;
    }

    if (async_mode) {
        return runAsync(config, queue_size);
    } else {
        return runSync(config);
    }
}
