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
    // 从config更新势场参数
    pf_params_.attractive_gain = config.attractive_gain;
    pf_params_.repulsive_gain = config.repulsive_gain;
    pf_params_.obstacle_influence_dist = config.obstacle_influence_dist;
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

    // 自车状态 (相对坐标系下始终在原点)
    core::EgoState ego_state;
    ego_state.x = 0.0f;
    ego_state.y = 0.0f;
    ego_state.heading = 0.0f;
    ego_state.speed = 0.0f;
    ego_state.timestamp_ns = packet->timestamp_ns;
    ego_state.dt = config_.planning_resolution / config_.max_speed;  // 估算时间步长

    // 在相对坐标系模式下，障碍物坐标已经是相对自车的
    // 在全局坐标系模式下，需要将障碍物转换到自车坐标系
    std::vector<core::Detection> obstacles = det_packet->detections;
    
    if (!config_.use_relative_frame) {
        // 全局坐标系模式：将障碍物从全局坐标转换到自车局部坐标
        // 这里假设自车位置已知，实际应用中应从传感器获取
        // 暂时保持障碍物不变
    }

    // 生成规划轨迹
    latest_trajectory_ = generateTrajectory(obstacles, ego_state);

    LOG_INFO_FMT("[Planner] Generated trajectory: points={}, feasible={}, status={}", 
                 latest_trajectory_.points.size(), 
                 latest_trajectory_.is_feasible,
                 latest_trajectory_.status);

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

    // 检查初始点是否在边界内
    if (config_.enable_boundary && isOutsideBoundary(current_x, current_y)) {
        trajectory.is_feasible = false;
        trajectory.status = "Start point outside boundary";
        LOG_WARN_FMT("[Planner] Start point ({}, {}) outside boundary", current_x, current_y);
        return trajectory;
    }

    LOG_INFO_FMT("[Planner] Generating trajectory: start=({},{}), goal=({},{}), boundary={}",
                 current_x, current_y, goal_x, goal_y, config_.enable_boundary);

    // 梯度下降法寻找路径
    for (int i = 0; i < pf_params_.max_iterations; ++i) {
        // 计算合力
        float fx = 0.0f, fy = 0.0f;
        
        // 引力
        float afx = 0.0f, afy = 0.0f;
        computeAttractiveForce(afx, afy, current_x, current_y, goal_x, goal_y);
        fx += afx;
        fy += afy;

        // 斥力 (障碍物)
        float rfx = 0.0f, rfy = 0.0f;
        computeRepulsiveForce(rfx, rfy, current_x, current_y, obstacles);
        fx += rfx;
        fy += rfy;

        // 边界斥力 (道路边界)
        if (config_.enable_boundary) {
            float bfx = 0.0f, bfy = 0.0f;
            computeBoundaryForce(bfx, bfy, current_x, current_y);
            fx += bfx;
            fy += bfy;
        }

        // 检查是否到达目标点
        float dist_to_goal = distanceBetweenPoints(current_x, current_y, goal_x, goal_y);
        if (dist_to_goal < config_.planning_resolution * 10.0f) {  // 放宽到10倍分辨率
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

        // 检查是否超出边界
        if (config_.enable_boundary && isOutsideBoundary(current_x, current_y)) {
            LOG_WARN_FMT("[Planner] Outside boundary at ({}, {})", current_x, current_y);
            trajectory.is_feasible = false;
            trajectory.status = "Outside boundary";
            break;
        }

        // 梯度下降更新位置
        float force_magnitude = std::sqrt(fx * fx + fy * fy);
        
        // 调试：打印力信息（每50步打印一次）
        if (i % 50 == 0 || i < 3) {
            LOG_INFO_FMT("[Planner] Iter {}: pos=({:.2f},{:.2f}), force=({:.2f},{:.2f}), mag={:.2f}", 
                         i, current_x, current_y, fx, fy, force_magnitude);
        }
        
        if (force_magnitude < pf_params_.convergence_threshold) {
            // 判断是否已经走了足够远（接近目标）
            float progress = distanceBetweenPoints(ego_state.x, ego_state.y, current_x, current_y);
            float total_dist = distanceBetweenPoints(ego_state.x, ego_state.y, goal_x, goal_y);
            
            LOG_INFO_FMT("total_dist:{}, progress:{}",total_dist,progress);
            if (total_dist > 0 && progress / total_dist > 0.8f) {
                // 已经走了80%以上的距离，视为可行
                trajectory.is_feasible = true;
                trajectory.status = "Near goal (local minimum accepted)";
                
            } else {
                trajectory.is_feasible = false;
                trajectory.status = "Converged to local minimum";
            }
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
        // 检查进度：如果已经走了大部分距离，视为可行
        float progress = distanceBetweenPoints(ego_state.x, ego_state.y, current_x, current_y);
        float total_dist = distanceBetweenPoints(ego_state.x, ego_state.y, goal_x, goal_y);
        
        if (total_dist > 0 && progress / total_dist > 0.8f) {
            trajectory.is_feasible = true;
            trajectory.status = "Max iterations reached but near goal";
        } else {
            trajectory.is_feasible = false;
            trajectory.status = "Max iterations reached";
        }
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
        LOG_INFO_FMT("[Planner] Trajectory: points={}, length={:.1f}m, end=({:.1f},{:.1f})", 
                     path_points.size(), trajectory.total_length,
                     path_points.back().x, path_points.back().y);
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
        // 二次引力：力与距离成正比，接近目标时力变小
        fx = pf_params_.attractive_gain * dx;
        fy = pf_params_.attractive_gain * dy;
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

void PotentialFieldPlanner::computeBoundaryForce(
    float& fx, float& fy,
    float x, float y) {
    
    fx = 0.0f;
    fy = 0.0f;

    // 检查x范围
    if (x < config_.boundary_start_x || x > config_.boundary_end_x) {
        return;  // 不在边界范围内
    }

    // 左边界斥力
    float dist_to_left = config_.boundary_left - y;
    if (dist_to_left > 0 && dist_to_left < config_.boundary_influence_dist) {
        // 力的大小与距离成反比，越近力越大
        float repulsive_force = config_.boundary_repulsive_gain * 
            (config_.boundary_influence_dist - dist_to_left) / 
            (dist_to_left * config_.boundary_influence_dist);
        fy -= repulsive_force;  // 向下推（远离左边界）
    }

    // 右边界斥力
    float dist_to_right = y - config_.boundary_right;
    if (dist_to_right > 0 && dist_to_right < config_.boundary_influence_dist) {
        // 力的大小与距离成反比，越近力越大
        float repulsive_force = config_.boundary_repulsive_gain * 
            (config_.boundary_influence_dist - dist_to_right) / 
            (dist_to_right * config_.boundary_influence_dist);
        fy += repulsive_force;  // 向上推（远离右边界）
    }
}

bool PotentialFieldPlanner::isOutsideBoundary(float x, float y) {
    // 检查x范围
    if (x < config_.boundary_start_x || x > config_.boundary_end_x) {
        return false;  // 不在边界范围内，不检查
    }

    // 检查y边界（包含膨胀半径）
    float inflated_left = config_.boundary_left + config_.boundary_inflation;
    float inflated_right = config_.boundary_right - config_.boundary_inflation;

    bool outside = (y > inflated_left || y < inflated_right);
    if (outside) {
        LOG_WARN_FMT("[Planner] Point ({}, {}) outside boundary: left={}, right={}", 
                     x, y, inflated_left, inflated_right);
    }
    return outside;
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