// src/nodes/planner/potential_field_planner.cpp
#include "3rd_party/log_mgr/log_mgr.h"
#include "potential_field_planner.h"
#include "node_factory.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace lidar_core {
namespace nodes {

PotentialFieldPlanner::PotentialFieldPlanner() : IPlannerNode("PotentialFieldPlanner") {
}

PotentialFieldPlanner::~PotentialFieldPlanner() {
    stop();
}

void PotentialFieldPlanner::setPlannerType(PlannerType type) {
    planner_type_ = type;
}

PlannerType PotentialFieldPlanner::getPlannerType() const {
    return planner_type_;
}

void PotentialFieldPlanner::setPlannerConfig(const core::PlanningConfig& config) {
    config_ = config;
}

const core::PlanningConfig& PotentialFieldPlanner::getPlannerConfig() const {
    return config_;
}

const core::Trajectory& PotentialFieldPlanner::getLatestTrajectory() const {
    return latest_trajectory_;
}

bool PotentialFieldPlanner::isPlanningFeasible() const {
    return latest_trajectory_.is_feasible;
}

bool PotentialFieldPlanner::start() {
    if (running_)
        return true;

    running_ = true;
    LOG_INFO_FMT("[PotentialFieldPlanner] Started, goal: ({}, {})", 
                 config_.goal_x, config_.goal_y);
    return true;
}

void PotentialFieldPlanner::stop() {
    if (!running_)
        return;
    running_ = false;
    latest_trajectory_.points.clear();
    LOG_INFO_FMT("[PotentialFieldPlanner] Stopped");
}

void PotentialFieldPlanner::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto det_packet = std::dynamic_pointer_cast<core::DetectionPacket>(packet);
    if (!det_packet) {
        LOG_ERROR_FMT("[PotentialFieldPlanner] Invalid packet type");
        return;
    }

    // 使用原点作为自车位置 (实际应用中应从传感器获取)
    core::EgoState ego_state;
    ego_state.x = 0.0f;
    ego_state.y = 0.0f;
    ego_state.heading = 0.0f;
    ego_state.speed = 0.0f;
    ego_state.timestamp_ns = packet->timestamp_ns;

    // 生成规划轨迹
    latest_trajectory_ = generateTrajectory(det_packet->detections, ego_state);

    // 创建规划数据包
    auto plan_packet = std::make_shared<core::PlanningPacket>();
    plan_packet->frame_id = packet->frame_id;
    plan_packet->timestamp_ns = packet->timestamp_ns;
    plan_packet->obstacles = det_packet->detections;
    plan_packet->cloud_points = det_packet->cloud_points;
    plan_packet->is_feasible = latest_trajectory_.is_feasible;
    plan_packet->planning_status = latest_trajectory_.status;
    plan_packet->planning_timestamp_ns = packet->timestamp_ns;

    // 转换轨迹格式
    for (const auto& pt : latest_trajectory_.points) {
        core::PlanningPacket::PathPoint plan_pt;
        plan_pt.x = pt.x;
        plan_pt.y = pt.y;
        plan_pt.z = pt.z;
        plan_pt.heading = pt.heading;
        plan_pt.curvature = pt.curvature;
        plan_pt.speed = pt.speed;
        plan_pt.acceleration = pt.acceleration;
        plan_pt.relative_time = pt.relative_time;
        plan_packet->trajectory.push_back(plan_pt);
    }

    // 广播给下游节点
    broadcast(plan_packet);
}

core::Trajectory PotentialFieldPlanner::generateTrajectory(
    const std::vector<core::Detection>& obstacles,
    const core::EgoState& ego_state) {
    
    core::Trajectory trajectory;
    std::vector<core::PathPoint> path_points;

    float current_x = ego_state.x;
    float current_y = ego_state.y;
    float goal_x = config_.goal_x;
    float goal_y = config_.goal_y;

    // 梯度下降法寻找路径
    for (int i = 0; i < pf_params_.max_iterations; ++i) {
        // 计算合力
        float fx = 0.0f, fy = 0.0f;
        
        // 引力
        float afx = 0.0f, afy = 0.0f;
        computeAttractiveForce(afx, afy, current_x, current_y, goal_x, goal_y);
        fx += afx;
        fy += afy;

        // 斥力
        float rfx = 0.0f, rfy = 0.0f;
        computeRepulsiveForce(rfx, rfy, current_x, current_y, obstacles);
        fx += rfx;
        fy += rfy;

        // 检查是否到达目标点
        float dist_to_goal = distanceBetweenPoints(current_x, current_y, goal_x, goal_y);
        if (dist_to_goal < config_.planning_resolution) {
            trajectory.is_feasible = true;
            trajectory.status = "Reached goal";
            break;
        }

        // 检查碰撞
        if (checkCollision(current_x, current_y, obstacles)) {
            trajectory.is_feasible = false;
            trajectory.status = "Collision detected";
            break;
        }

        // 梯度下降更新位置
        float force_magnitude = std::sqrt(fx * fx + fy * fy);
        if (force_magnitude < pf_params_.convergence_threshold) {
            trajectory.is_feasible = false;
            trajectory.status = "Converged to local minimum";
            break;
        }

        // 归一化并更新位置
        float norm_fx = fx / force_magnitude;
        float norm_fy = fy / force_magnitude;
        
        current_x += pf_params_.step_size * norm_fx;
        current_y += pf_params_.step_size * norm_fy;

        // 添加路径点
        core::PathPoint pt;
        pt.x = current_x;
        pt.y = current_y;
        pt.z = 0.0f;
        pt.heading = std::atan2(norm_fy, norm_fx);
        pt.relative_time = static_cast<float>(i) * pf_params_.step_size / config_.max_speed;
        path_points.push_back(pt);
    }

    // 如果达到最大迭代次数但未到达目标
    if (!trajectory.is_feasible && trajectory.status.empty()) {
        trajectory.is_feasible = false;
        trajectory.status = "Max iterations reached";
    }

    // 计算轨迹速度
    computeTrajectorySpeed(path_points);

    // 平滑轨迹
    smoothTrajectory(path_points);

    trajectory.points = path_points;
    
    // 计算轨迹总长度和时间
    if (!path_points.empty()) {
        trajectory.total_time = path_points.back().relative_time;
        trajectory.total_length = 0.0f;
        for (size_t i = 1; i < path_points.size(); ++i) {
            trajectory.total_length += distanceBetweenPoints(
                path_points[i-1].x, path_points[i-1].y,
                path_points[i].x, path_points[i].y
            );
        }
    }

    return trajectory;
}

void PotentialFieldPlanner::computeAttractiveForce(
    float& fx, float& fy,
    float x, float y,
    float goal_x, float goal_y) {
    
    float dx = goal_x - x;
    float dy = goal_y - y;
    float dist = std::sqrt(dx * dx + dy * dy);
    
    if (dist > 0.001f) {
        // 线性引力 (距离远时力恒定)
        fx = pf_params_.attractive_gain * dx / dist;
        fy = pf_params_.attractive_gain * dy / dist;
    } else {
        fx = 0.0f;
        fy = 0.0f;
    }
}

void PotentialFieldPlanner::computeRepulsiveForce(
    float& fx, float& fy,
    float x, float y,
    const std::vector<core::Detection>& obstacles) {
    
    fx = 0.0f;
    fy = 0.0f;

    for (const auto& obs : obstacles) {
        float dist = distanceBetweenPoints(x, y, obs.x, obs.y);
        
        // 考虑障碍物尺寸
        float obs_radius = std::max(obs.w, obs.l) / 2.0f;
        float effective_dist = dist - obs_radius;
        
        if (effective_dist < pf_params_.obstacle_influence_dist && effective_dist > 0.001f) {
            // 斥力方向：远离障碍物
            float dir_x = (x - obs.x) / dist;
            float dir_y = (y - obs.y) / dist;
            
            // 斥力大小：与距离成反比
            float repulsive_force = pf_params_.repulsive_gain * 
                (1.0f / effective_dist - 1.0f / pf_params_.obstacle_influence_dist) /
                (effective_dist * effective_dist);
            
            fx += repulsive_force * dir_x;
            fy += repulsive_force * dir_y;
        }
    }
}

bool PotentialFieldPlanner::checkCollision(
    float x, float y,
    const std::vector<core::Detection>& obstacles) {
    
    for (const auto& obs : obstacles) {
        float dist = distanceBetweenPoints(x, y, obs.x, obs.y);
        float obs_radius = std::max(obs.w, obs.l) / 2.0f;
        float safety_radius = obs_radius + config_.safety_margin;
        
        if (dist < safety_radius) {
            return true;
        }
    }
    return false;
}

void PotentialFieldPlanner::computeTrajectorySpeed(
    std::vector<core::PathPoint>& trajectory) {
    
    if (trajectory.size() < 2) return;

    // 假设匀速运动
    for (size_t i = 0; i < trajectory.size(); ++i) {
        trajectory[i].speed = config_.max_speed;
        trajectory[i].acceleration = 0.0f;
        
        // 计算曲率
        if (i > 0 && i < trajectory.size() - 1) {
            float dx1 = trajectory[i].x - trajectory[i-1].x;
            float dy1 = trajectory[i].y - trajectory[i-1].y;
            float dx2 = trajectory[i+1].x - trajectory[i].x;
            float dy2 = trajectory[i+1].y - trajectory[i].y;
            
            float cross = dx1 * dy2 - dy1 * dx2;
            float dot = dx1 * dx2 + dy1 * dy2;
            float curvature = std::abs(cross) / (std::sqrt(dx1*dx1 + dy1*dy1) * 
                                                 std::sqrt(dx2*dx2 + dy2*dy2) + 0.001f);
            trajectory[i].curvature = curvature;
        }
    }
}

void PotentialFieldPlanner::smoothTrajectory(
    std::vector<core::PathPoint>& trajectory) {
    
    if (trajectory.size() < 3) return;

    // 简单的移动平均平滑
    std::vector<core::PathPoint> smoothed = trajectory;
    int window_size = 3;

    for (size_t i = 1; i < trajectory.size() - 1; ++i) {
        smoothed[i].x = (trajectory[i-1].x + trajectory[i].x + trajectory[i+1].x) / 3.0f;
        smoothed[i].y = (trajectory[i-1].y + trajectory[i].y + trajectory[i+1].y) / 3.0f;
    }

    trajectory = smoothed;
}

float PotentialFieldPlanner::distanceBetweenPoints(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

float PotentialFieldPlanner::normalizeAngle(float angle) {
    while (angle > M_PI) angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}

// 注册节点
REGISTER_NODE("potential_field_planner", PotentialFieldPlanner)

} // namespace nodes
} // namespace lidar_core