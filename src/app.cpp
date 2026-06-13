#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <csignal>

#include "pipeline/types.hpp"
#include "pipeline/reader.hpp"
#include "pipeline/engine.hpp"
#include "pipeline/output.hpp"
#ifdef WITH_OPENCV
#include "pipeline/bev_visualizer.hpp"
#endif

static volatile bool g_running = true;

static void signalHandler(int) { g_running = false; }

static void printHelp() {
    std::cout << "Usage: lidar_app [options]\n"
              << "Options:\n"
              << "  --input <path>           Input path (bin file/dir or pcap)\n"
              << "  --input-type <type>      Input type: bin (default), velodyne\n"
              << "  --model <path>           TRT engine path\n"
              << "  --output <path>          Output directory\n"
              << "  --output-types <types>   Output types: file,vis,callback (comma-separated)\n"
              << "  --async                  Enable async inference\n"
              << "  --workers <n>            Number of async workers (default: 2)\n"
              << "  --score-thresh <f>       Score threshold (default: 0.1)\n"
              << "  --timer                  Show per-frame timing\n"
              << "  --help                   Show this help\n";
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);

    pipeline::ReaderConfig reader_config;
    pipeline::EngineConfig engine_config;
    pipeline::OutputConfig output_config;

    std::vector<pipeline::OutputType> output_types;
    bool timer = false;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            reader_config.input_path = argv[++i];
        } else if (strcmp(argv[i], "--input-type") == 0 && i + 1 < argc) {
            std::string type = argv[++i];
            if (type == "velodyne") reader_config.type = pipeline::ReaderType::VELODYNE_UDP;
            else reader_config.type = pipeline::ReaderType::BIN_FILE;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            engine_config.model_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_config.output_dir = argv[++i];
        } else if (strcmp(argv[i], "--output-types") == 0 && i + 1 < argc) {
            std::string types = argv[++i];
            size_t pos = 0;
            while ((pos = types.find(',')) != std::string::npos) {
                std::string t = types.substr(0, pos);
                if (t == "file") output_types.push_back(pipeline::OutputType::FILE);
                else if (t == "vis") output_types.push_back(pipeline::OutputType::BEV_VISUALIZER);
                else if (t == "callback") output_types.push_back(pipeline::OutputType::CALLBACK);
                types.erase(0, pos + 1);
            }
            if (types == "file") output_types.push_back(pipeline::OutputType::FILE);
            else if (types == "vis") output_types.push_back(pipeline::OutputType::BEV_VISUALIZER);
            else if (types == "callback") output_types.push_back(pipeline::OutputType::CALLBACK);
        } else if (strcmp(argv[i], "--async") == 0) {
            engine_config.async_mode = true;
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            engine_config.num_workers = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--score-thresh") == 0 && i + 1 < argc) {
            engine_config.score_thresh = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--timer") == 0) {
            timer = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printHelp();
            return 0;
        }
    }

    // Validate
    if (reader_config.input_path.empty()) {
        std::cerr << "Error: --input is required" << std::endl;
        printHelp();
        return 1;
    }

    // Default output: file
    if (output_types.empty()) {
        output_types.push_back(pipeline::OutputType::FILE);
    }

    // Create reader
    auto reader = pipeline::createReader(reader_config);
    if (!reader || !reader->open()) {
        std::cerr << "Failed to open input: " << reader_config.input_path << std::endl;
        return 1;
    }

    // Create engine
    pipeline::PointPillarsEngine engine(engine_config);
    if (!engine.init()) {
        std::cerr << "Failed to init engine" << std::endl;
        return 1;
    }

    // Create outputs
    std::vector<std::shared_ptr<pipeline::DetectionOutput>> outputs;
#ifdef WITH_OPENCV
    pipeline::BEVVisualizer* vis_ptr = nullptr;
#endif

    for (auto ot : output_types) {
        pipeline::OutputConfig cfg;
        cfg.type = ot;
        cfg.output_dir = output_config.output_dir;
        auto out = pipeline::createOutput(cfg);
        if (out) {
            outputs.push_back(out);
#ifdef WITH_OPENCV
            if (ot == pipeline::OutputType::BEV_VISUALIZER) {
                vis_ptr = dynamic_cast<pipeline::BEVVisualizer*>(out.get());
            }
#endif
        }
    }

    std::cout << "Pipeline ready. Processing..." << std::endl;

    // Process frames
    uint64_t frame_count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    pipeline::PointCloud cloud;
    while (g_running && reader->read(cloud)) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        if (engine_config.async_mode) {
            // Async: dispatch and continue
            engine.detectAsyncWithVis(
                cloud,
                [&outputs](const std::vector<pipeline::Detection>& dets, uint64_t fid) {
                    for (auto& out : outputs) {
                        out->write(dets, fid);
                    }
                },
                nullptr
            );
        } else {
            // Sync
            auto detections = engine.detect(cloud);

#ifdef WITH_OPENCV
            if (vis_ptr) {
                vis_ptr->setPointCloud(cloud);
            }
#endif
            for (auto& out : outputs) {
                out->write(detections, cloud.frame_id);
            }
        }

        auto frame_end = std::chrono::high_resolution_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

        if (timer) {
            std::cout << "Frame " << cloud.frame_id << ": " << frame_ms << " ms" << std::endl;
        }

        frame_count++;
        cloud.points.clear();
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\nDone. Processed " << frame_count << " frames in " << total_sec << "s";
    if (total_sec > 0) {
        std::cout << " (" << (frame_count / total_sec) << " FPS)";
    }
    std::cout << std::endl;

    // Shutdown
    engine.shutdown();
    for (auto& out : outputs) out->close();
    reader->close();

    return 0;
}
