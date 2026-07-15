// src/main.cpp
#include <cinttypes>
#include <iostream>
#include <string>
#include <csignal>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "lidar_core/core/pipeline.h"
#include "lidar_core/core/async_pipeline.h"
#include "lidar_core/nodes/i_source_node.h"
#include "lidar_core/nodes/i_infer_node.h"
#include "lidar_core/nodes/i_output_node.h"
#include "src/nodes/source/bin_source_node.h"

#include "3rd_party/log_mgr/log_mgr.h"

static volatile bool g_running = true;

static void signalHandler(int) { g_running = false; }

static void printHelp() {
    LOG_INFO_FMT("[Main] Usage: lidar_app [options]");
    LOG_INFO_FMT("[Main] Options:");
    LOG_INFO_FMT("[Main]   --config <path>    JSON config file path");
    LOG_INFO_FMT("[Main]   --async            Enable async pipeline (reader/infer in parallel)");
    LOG_INFO_FMT("[Main]   --queue <size>     Async queue size (default: 3)");
    LOG_INFO_FMT("[Main]   --help             Show this help");
}

// 同步模式
static int runSync(const nlohmann::json& config) {
    std::string pipeline_id = config["pipeline"]["id"];
    auto pipeline = std::make_shared<lidar_core::core::Pipeline>(pipeline_id);

    if (!pipeline->buildFromJson(config)) {
        LOG_ERROR_FMT("[Main] Failed to build pipeline from config");
        return 1;
    }

    if (!pipeline->start()) {
        LOG_ERROR_FMT("[Main] Failed to start pipeline");
        return 1;
    }

    LOG_INFO_FMT("[Main] Pipeline ready (sync mode). Processing...");

    auto source_node = std::dynamic_pointer_cast<lidar_core::nodes::ISourceNode>(
        pipeline->getNode("source"));
    auto infer_node = std::dynamic_pointer_cast<lidar_core::nodes::IInferNode>(
        pipeline->getNode("infer"));

    if (!source_node || !infer_node) {
        LOG_ERROR_FMT("[Main] Missing required nodes");
        pipeline->stop();
        return 1;
    }

    auto bin_source = std::dynamic_pointer_cast<lidar_core::nodes::BinSourceNode>(source_node);

    uint64_t frame_count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    // idle timeout: 连续 5 秒无数据则退出
    const int kIdleLimit = 500;
    int idle_count = 0;

    while (g_running && pipeline->isRunning()) {
        auto cloud_packet = std::make_shared<lidar_core::core::PointCloudPacket>();
        if (bin_source && !bin_source->readNext(cloud_packet)) {
            if (++idle_count >= kIdleLimit) {
                LOG_INFO_FMT("[Main] No data for {} consecutive polls, exiting", idle_count);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        idle_count = 0;  // 有数据则重置空闲计数
        infer_node->pushData(cloud_packet);
        frame_count++;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    LOG_INFO_FMT("[Main] Done. Processed {} frames in {}s", frame_count, total_sec);
    if (total_sec > 0) {
        double fps = static_cast<double>(frame_count) / total_sec;
        LOG_INFO_FMT("[Main] FPS: {}", fps);
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
        LOG_ERROR_FMT("[Main] Failed to build async pipeline");
        return 1;
    }

    if (!async_pipeline->start()) {
        LOG_ERROR_FMT("[Main] Failed to start async pipeline");
        return 1;
    }

    LOG_INFO_FMT("[Main] Pipeline ready (async mode, queue={}). Processing...", queue_size);

    // 等待处理完成
    while (g_running && async_pipeline->isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    async_pipeline->stop();

    uint64_t frames = async_pipeline->getProcessedFrames();
    double fps = async_pipeline->getFPS();

    LOG_INFO_FMT("[Main] Done. Processed {} frames", frames);
    if (fps > 0) {
        LOG_INFO_FMT("[Main] FPS: {}", fps);
    }

    return 0;
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);

    // 设置日志
    LogManager::Config log_config;
    log_config.log_dir = "./logs";
    log_config.log_name = "lidar_detection_pipeline";
    log_config.level = LogManager::Level::S_INFO;
    log_config.max_file_size = 5 * 1024 * 1024; // 5MB
    log_config.max_files = 5;
    log_config.console_output = true;
    log_config.async_mode = false;
    log_config.queue_size = 16384;
    log_config.flush_interval = 2;

    if (!LogManager::get_instance().initialize(log_config))
    {
        std::cerr << "Failed to initialize logger!" << std::endl;
        return -1;
    }

    std::string config_path = "../config/pipeline.json";
    // 也尝试从可执行文件所在目录的父目录查找
    {
        std::string exe_dir(argv[0]);
        auto pos = exe_dir.find_last_of("/\\");
        if (pos != std::string::npos) {
            exe_dir = exe_dir.substr(0, pos);
            std::string alt_path = exe_dir + "/../config/pipeline.json";
            std::ifstream test(alt_path);
            if (test.is_open()) {
                config_path = alt_path;
            }
        }
    }
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
        LOG_ERROR_FMT("[Main] Cannot open config file: {}", config_path);
        return 1;
    }

    nlohmann::json config;
    try {
        config_file >> config;
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[Main] Invalid JSON config: {}", e.what());
        return 1;
    }

    if (!config.contains("pipeline") || !config["pipeline"].contains("id")) {
        LOG_ERROR_FMT("[Main] Missing 'pipeline.id' in config");
        return 1;
    }

    if (async_mode) {
        return runAsync(config, queue_size);
    } else {
        return runSync(config);
    }
}
