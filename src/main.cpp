// src/main.cpp
#include <iostream>
#include <string>
#include <csignal>
#include <fstream>

#include <nlohmann/json.hpp>

#include "lidar_core/core/pipeline.h"
#include "lidar_core/nodes/i_source_node.h"
#include "lidar_core/nodes/i_infer_node.h"
#include "lidar_core/nodes/i_output_node.h"
#include "src/nodes/source/bin_source_node.h"

static volatile bool g_running = true;

static void signalHandler(int) { g_running = false; }

static void printHelp() {
    std::cout << "Usage: lidar_app [options]\n"
              << "Options:\n"
              << "  --config <path>    JSON config file path\n"
              << "  --help             Show this help\n";
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);

    std::string config_path = "../config/pipeline.json";

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
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

    // Get pipeline config
    if (!config.contains("pipeline") || !config["pipeline"].contains("id")) {
        std::cerr << "Error: Missing 'pipeline.id' in config" << std::endl;
        return 1;
    }

    std::string pipeline_id = config["pipeline"]["id"];
    auto pipeline = std::make_shared<lidar_core::core::Pipeline>(pipeline_id);

    // Build pipeline from config
    if (!pipeline->buildFromJson(config)) {
        std::cerr << "Error: Failed to build pipeline" << std::endl;
        return 1;
    }

    // Start pipeline
    if (!pipeline->start()) {
        std::cerr << "Error: Failed to start pipeline" << std::endl;
        return 1;
    }

    std::cout << "Pipeline ready. Processing..." << std::endl;

    // Get source and infer nodes
    auto source_node = std::dynamic_pointer_cast<lidar_core::nodes::ISourceNode>(
        pipeline->getNode("source"));
    auto infer_node = std::dynamic_pointer_cast<lidar_core::nodes::IInferNode>(
        pipeline->getNode("infer"));

    if (!source_node || !infer_node) {
        std::cerr << "Error: Missing required nodes" << std::endl;
        pipeline->stop();
        return 1;
    }

    // Process frames
    uint64_t frame_count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    while (g_running && pipeline->isRunning()) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        // Read point cloud
        auto cloud_packet = std::make_shared<lidar_core::core::PointCloudPacket>();
        if (!std::dynamic_pointer_cast<lidar_core::nodes::BinSourceNode>(source_node)->readNext(cloud_packet)) {
            break;
        }

        // Run inference
        infer_node->pushData(cloud_packet);

        auto frame_end = std::chrono::high_resolution_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

        frame_count++;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\nDone. Processed " << frame_count << " frames in " << total_sec << "s";
    if (total_sec > 0) {
        std::cout << " (" << (frame_count / total_sec) << " FPS)";
    }
    std::cout << std::endl;

    // Shutdown
    pipeline->stop();

    return 0;
}
