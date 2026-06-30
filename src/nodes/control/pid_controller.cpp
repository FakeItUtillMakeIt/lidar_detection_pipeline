// src/nodes/control/pid_controller.cpp
#include "3rd_party/log_mgr/log_mgr.h"
#include "pid_controller.h"
#include "node_factory.h"
#include <cmath>
#include <algorithm>

namespace lidar_core {
namespace nodes {

PIDController::PIDController() : IControlNode("PIDController") {
}

PIDController::~PIDController() {
    stop();
}

void PIDController::setControllerType(ControllerType type) {
    controller_type_ = type;
}

ControllerType PIDController::getControllerType() const {
    return controller_type_;
}

void PIDController::setPIDParams(const PIDParams& params) {
    pid_params_ = params;
}

const PIDParams& PIDController::getPIDParams() const {
    return pid_params_;
}

void PIDController::updateEgoState(const core::EgoState& state) {
    ego_state_ = state;
}

const core::ControlCommand& PIDController::getLatestCommand() const {
    return latest_command_;
}

void PIDController::reset() {
    speed_pid_state_ = PIDState();
    heading_pid_state_ = PIDState();
    lateral_pid_state_ = PIDState();
    current_point_index_ = 0;
    trajectory_received_ = false;
    latest_command_ = core::ControlCommand();
}

bool PIDController::start() {
    if (running_)
        return true;

    reset();
    running_ = true;
    LOG_INFO_FMT("[PIDController] Started, kp_speed={}, kp_heading={}", 
                 pid_params_.kp_speed, pid_params_.kp_heading);
    return true;
}

void PIDController::stop() {
    if (!running_)
        return;
    running_ = false;
    reset();
    LOG_INFO_FMT("[PIDController] Stopped");
}

void PIDController::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto plan_packet = std::dynamic_pointer_cast<core::PlanningPacket>(packet);
    if (!plan_packet) {
        LOG_ERROR_FMT("[PIDController] Invalid packet type, expected PlanningPacket");
        return;
    }

    if (plan_packet->trajectory.empty()) {
        LOG_WARN_FMT("[PIDController] Empty trajectory, sending stop command");
        latest_command_.throttle = 0.0f;
        latest_command_.brake = 1.0f;
        latest_command_.steering = 0.0f;
        latest_command_.is_valid = false;
        latest_command_.status = "No trajectory";
        latest_command_.timestamp_ns = packet->timestamp_ns;
        broadcast(plan_packet);
        return;
    }

    // 查找最近轨迹点
    int nearest_idx = findNearestPoint(plan_packet->trajectory, 
                                       ego_state_.x, ego_state_.y);
    
    // 如果到达轨迹末端，使用最后一个点
    if (nearest_idx >= static_cast<int>(plan_packet->trajectory.size()) - 1) {
        nearest_idx = static_cast<int>(plan_packet->trajectory.size()) - 1;
    }

    const auto& target_point = plan_packet->trajectory[nearest_idx];
    float dt = ego_state_.dt > 0 ? ego_state_.dt : pid_params_.dt;

    // 计算目标速度 (从轨迹点获取)
    float target_speed = target_point.speed;

    // 计算目标航向
    float target_heading = target_point.heading;

    // 纵向控制 (速度控制)
    longitudinalControl(target_speed, ego_state_.speed, dt);

    // 横向控制 (航向控制)
    float lateral_error = computeLateralError(target_point, 
                                             ego_state_.x, ego_state_.y, 
                                             ego_state_.heading);
    float heading_error = computeHeadingError(target_heading, ego_state_.heading);
    
    lateralControl(target_heading, ego_state_.heading, 
                   lateral_error, 0.0f, dt);

    // 更新控制指令
    latest_command_.timestamp_ns = packet->timestamp_ns;
    latest_command_.is_valid = plan_packet->is_feasible;
    latest_command_.status = plan_packet->is_feasible ? "Tracking" : "Infeasible trajectory";

    LOG_DEBUG_FMT("[PIDController] target_speed={:.2f}, current_speed={:.2f}, throttle={:.2f}, brake={:.2f}, steering={:.2f}",
                  target_speed, ego_state_.speed, latest_command_.throttle, 
                  latest_command_.brake, latest_command_.steering);

    // 广播给下游节点
    broadcast(plan_packet);
}

float PIDController::computePID(float error, float& integral, float& prev_error,
                                float kp, float ki, float kd, float dt) {
    // 积分项 (带限幅)
    integral += error * dt;
    integral = clamp(integral, -pid_params_.max_integral, pid_params_.max_integral);
    
    // 微分项
    float derivative = (error - prev_error) / dt;
    prev_error = error;
    
    // PID输出
    return kp * error + ki * integral + kd * derivative;
}

void PIDController::longitudinalControl(float target_speed, float current_speed, float dt) {
    float speed_error = target_speed - current_speed;
    
    // 计算PID输出
    float pid_output = computePID(speed_error, 
                                  speed_pid_state_.integral,
                                  speed_pid_state_.prev_error,
                                  pid_params_.kp_speed,
                                  pid_params_.ki_speed,
                                  pid_params_.kd_speed,
                                  dt);
    
    // 转换为油门/刹车
    if (pid_output >= 0) {
        // 加速
        latest_command_.throttle = clamp(pid_output, 0.0f, 1.0f);
        latest_command_.brake = 0.0f;
    } else {
        // 减速
        latest_command_.throttle = 0.0f;
        latest_command_.brake = clamp(-pid_output, 0.0f, 1.0f);
    }
}

void PIDController::lateralControl(float target_heading, float current_heading,
                                   float target_lateral_offset, float current_lateral_offset, 
                                   float dt) {
    // 航向误差
    float heading_error = computeHeadingError(target_heading, current_heading);
    
    // 横向误差
    float lateral_error = target_lateral_offset - current_lateral_offset;
    
    // 组合航向和横向控制
    float heading_pid = computePID(heading_error,
                                   heading_pid_state_.integral,
                                   heading_pid_state_.prev_error,
                                   pid_params_.kp_heading,
                                   pid_params_.ki_heading,
                                   pid_params_.kd_heading,
                                   dt);
    
    float lateral_pid = computePID(lateral_error,
                                   lateral_pid_state_.integral,
                                   lateral_pid_state_.prev_error,
                                   pid_params_.kp_lateral,
                                   pid_params_.ki_lateral,
                                   pid_params_.kd_lateral,
                                   dt);
    
    // 组合输出 (航向控制为主，横向控制为辅)
    float steering_output = heading_pid + lateral_pid;
    
    // 限制转向范围 [-1, 1]
    latest_command_.steering = clamp(steering_output, -1.0f, 1.0f);
}

int PIDController::findNearestPoint(const std::vector<core::PlanningPacket::PathPoint>& trajectory,
                                   float ego_x, float ego_y) {
    if (trajectory.empty()) return 0;

    float min_dist = std::numeric_limits<float>::max();
    int nearest_idx = 0;

    // 从当前点开始搜索，提高效率
    for (size_t i = current_point_index_; i < trajectory.size(); ++i) {
        float dx = trajectory[i].x - ego_x;
        float dy = trajectory[i].y - ego_y;
        float dist = std::sqrt(dx * dx + dy * dy);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_idx = static_cast<int>(i);
        }
    }

    // 更新当前点索引 (允许回溯)
    current_point_index_ = nearest_idx;
    
    return nearest_idx;
}

float PIDController::computeLateralError(const core::PlanningPacket::PathPoint& target_point,
                                        float ego_x, float ego_y, float ego_heading) {
    // 计算从自车到目标点的向量
    float dx = target_point.x - ego_x;
    float dy = target_point.y - ego_y;
    
    // 旋转到自车坐标系
    float cos_h = std::cos(-ego_heading);
    float sin_h = std::sin(-ego_heading);
    
    float local_x = dx * cos_h - dy * sin_h;
    float local_y = dx * sin_h + dy * cos_h;
    
    // 横向误差 (y方向)
    return local_y;
}

float PIDController::computeHeadingError(float target_heading, float current_heading) {
    float error = target_heading - current_heading;
    
    // 归一化到 [-π, π]
    while (error > M_PI) error -= 2.0f * M_PI;
    while (error < -M_PI) error += 2.0f * M_PI;
    
    return error;
}

float PIDController::clamp(float value, float min_val, float max_val) {
    return std::max(min_val, std::min(max_val, value));
}

// 注册节点
REGISTER_NODE("pid_controller", PIDController)

} // namespace nodes
} // namespace lidar_core