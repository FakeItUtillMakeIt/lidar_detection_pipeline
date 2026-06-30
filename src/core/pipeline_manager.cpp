// src/core/pipeline_manager.cpp
#include "3rd_party/log_mgr/log_mgr.h"
#include "lidar_core/core/pipeline.h"
#include <nlohmann/json.hpp>
#include <iostream>

// 节点工厂头文件
#include "nodes/registry/node_factory.h"

// 节点头文件
#include "lidar_core/nodes/i_source_node.h"
#include "lidar_core/nodes/i_infer_node.h"
#include "lidar_core/nodes/i_output_node.h"
#include "lidar_core/nodes/i_tracker_node.h"
#include "lidar_core/nodes/i_attribute_node.h"
#include "lidar_core/nodes/i_planner_node.h"

namespace lidar_core {
namespace core {

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::buildFromJson(const nlohmann::json& config) {
    try {
        // 1. 解析节点配置并创建实例
        if (!config.contains("nodes") || !config["nodes"].is_array()) {
            LOG_ERROR_FMT("[Pipeline {}] Invalid config: missing 'nodes' array", id_);
            return false;
        }

        for (const auto& node_cfg : config["nodes"]) {
            std::string id = node_cfg.value("id", "");
            std::string type = node_cfg.value("type", "");
            nlohmann::json params = node_cfg.value("params", nlohmann::json::object());

            LOG_INFO_FMT("[Pipeline {}] Node: id:{}, type:{}", id_, id, type);

            if (id.empty() || type.empty()) {
                LOG_ERROR_FMT("[Pipeline {}] Node missing 'id' or 'type'", id_);
                return false;
            }

            auto node = NodeFactory::instance().create(type, params);
            if (!node) {
                LOG_ERROR_FMT("[Pipeline {}] Failed to create node type: {}", id_, type);
                return false;
            }

            // 根据节点类型设置参数
            if (type.find("source") != std::string::npos) {
                auto source_node = std::dynamic_pointer_cast<nodes::ISourceNode>(node);
                if (source_node) {
                    source_node->setInputPath(params.value("input_path", ""));
                    source_node->setInputType(params.value("input_type", "bin"));
                }
            }

            if (type.find("infer") != std::string::npos) {
                auto infer_node = std::dynamic_pointer_cast<nodes::IInferNode>(node);
                if (infer_node) {
                    infer_node->setModelParams(
                        params.value("model_path", ""),
                        params.value("score_thresh", 0.1f),
                        params.value("nms_thresh", 0.01f)
                    );
                    if (!infer_node->loadModel(infer_node->getModelPath())) {
                        LOG_ERROR_FMT("[Pipeline {}] Failed to load model", id_);
                        return false;
                    }
                }
            }

            if (type.find("output") != std::string::npos || type.find("visualizer") != std::string::npos) {
                auto output_node = std::dynamic_pointer_cast<nodes::IOutputNode>(node);
                if (output_node) {
                    output_node->setOutputDir(params.value("output_dir", "./out"));
                    output_node->setOutputType(params.value("output_type", "file"));
                }
            }

            if (type.find("tracker") != std::string::npos) {
                auto tracker_node = std::dynamic_pointer_cast<nodes::ITrackerNode>(node);
                if (tracker_node) {
                    // 设置跟踪器类型
                    std::string tracker_type = params.value("tracker_type", "ocsort");
                    if (tracker_type == "ocsort") {
                        tracker_node->setTrackerType(nodes::TrackerType::OCSORT);
                    }

                    // 设置OCSort配置
                    nodes::OCSortConfig ocsort_config;
                    if (params.contains("ocsort_config")) {
                        auto& cfg = params["ocsort_config"];
                        ocsort_config.det_thresh = cfg.value("det_thresh", 0.3f);
                        ocsort_config.max_age = cfg.value("max_age", 30);
                        ocsort_config.min_hits = cfg.value("min_hits", 3);
                        ocsort_config.iou_threshold = cfg.value("iou_threshold", 0.3f);
                        ocsort_config.delta_t = cfg.value("delta_t", 3);
                        ocsort_config.inertia = cfg.value("inertia", 0.2f);
                        ocsort_config.use_byte = cfg.value("use_byte", false);
                    }
                    tracker_node->setOCSortConfig(ocsort_config);
                }
            }

            if (type.find("attribute") != std::string::npos) {
                auto attr_node = std::dynamic_pointer_cast<nodes::IAttributeNode>(node);
                if (attr_node) {
                    attr_node->setTimeInterval(params.value("dt", 0.1f));
                }
            }

            if (type.find("planner") != std::string::npos){
                auto plan_node = std::dynamic_pointer_cast<nodes::IPlannerNode>(node);
                if (plan_node){
                    PlanningConfig planing_config;
                    planing_config.max_speed=params.value("max_speed",8.0);
                    planing_config.max_accel=params.value("max_accel",2.0);
                    planing_config.goal_x=params.value("goal_x",100.0);
                    planing_config.goal_y=params.value("goal_y",0.0);
                    plan_node->setPlannerConfig(planing_config);
                }
            }

            nodes_[id] = node;
            LOG_INFO_FMT("[Pipeline {}] Created node: {}, type: {}", id_, id, type);
        }

        // 2. 建立边连接
        if (config.contains("edges") && config["edges"].is_array()) {
            for (const auto& edge : config["edges"]) {
                std::string from = edge.value("from", "");
                std::string to = edge.value("to", "");

                auto it_from = nodes_.find(from);
                auto it_to = nodes_.find(to);
                if (it_from == nodes_.end() || it_to == nodes_.end()) {
                    LOG_ERROR_FMT("[Pipeline {}] Invalid edge: {} -> {}", id_, from, to);
                    return false;
                }

                it_from->second->addDownstream(it_to->second);
                LOG_INFO_FMT("[Pipeline {}] Connected: {} -> {}", id_, from, to);
            }
        }

        // 3. 为每个节点设置管道弱引用
        auto weak_self = weak_from_this();
        for (auto& [_, node] : nodes_) {
            node->setPipeline(weak_self);
        }

        return true;
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[Pipeline {}] Exception during build: {}", id_, e.what());
        return false;
    }
}

bool Pipeline::start() {
    if (running_)
        return true;

    bool all_started = true;
    for (auto& [name, node] : nodes_) {
        LOG_INFO_FMT("[Pipeline {}] Starting node: {}", id_, name);
        if (!node->start()) {
            LOG_ERROR_FMT("[Pipeline {}] Failed to start node: {}", id_, name);
            all_started = false;
            break;
        }
    }

    if (all_started) {
        running_ = true;
        LOG_INFO_FMT("[Pipeline {}] All nodes started successfully", id_);
    } else {
        stop();
    }
    return all_started;
}

void Pipeline::stop() {
    if (!running_)
        return;
    running_ = false;

    for (auto& [name, node] : nodes_) {
        LOG_INFO_FMT("[Pipeline {}] Stopping node: {}", id_, name);
        node->stop();
    }
    LOG_INFO_FMT("[Pipeline {}] Pipeline stopped", id_);
}

std::shared_ptr<Node> Pipeline::getNode(const std::string& name) {
    auto it = nodes_.find(name);
    if (it != nodes_.end()) {
        return it->second;
    }
    return nullptr;
}

bool Pipeline::isRunning() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& node : nodes_) {
        if (node.second->isRunning()) {
            return true;
        }
    }
    return false;
}

} // namespace core
} // namespace lidar_core
