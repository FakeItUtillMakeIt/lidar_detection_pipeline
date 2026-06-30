// src/nodes/control/pid_controller.h
#pragma once

#include "lidar_core/nodes/i_control_node.h"
#include <memory>

namespace lidar_core {
namespace nodes {

class PIDController : public IControlNode {
public:
    PIDController();
    ~PIDController() override;

    // IControlNode 接口
    void setControllerType(ControllerType type) override;
    ControllerType getControllerType() const override;
    void setPIDParams(const PIDParams& params) override;
    const PIDParams& getPIDParams() const override;
    void updateEgoState(const core::EgoState& state) override;
    const core::ControlCommand& getLatestCommand() const override;
    void reset() override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    // PID计算
    float computePID(float error, float& integral, float& prev_error, 
                     float kp, float ki, float kd, float dt);

    // 纵向控制 (速度控制)
    void longitudinalControl(float target_speed, float current_speed, float dt);

    // 横向控制 (航向/转向控制)
    void lateralControl(float target_heading, float current_heading, 
                       float target_lateral_offset, float current_lateral_offset, float dt);

    // 查找最近轨迹点
    int findNearestPoint(const std::vector<core::PlanningPacket::PathPoint>& trajectory,
                        float ego_x, float ego_y);

    // 计算横向偏差
    float computeLateralError(const core::PlanningPacket::PathPoint& target_point,
                             float ego_x, float ego_y, float ego_heading);

    // 计算航向偏差
    float computeHeadingError(float target_heading, float current_heading);

    // 限制输出范围
    float clamp(float value, float min_val, float max_val);

    ControllerType controller_type_ = ControllerType::PID;
    PIDParams pid_params_;
    core::EgoState ego_state_;
    core::ControlCommand latest_command_;

    // PID状态
    struct PIDState {
        float integral = 0.0f;
        float prev_error = 0.0f;
    };
    PIDState speed_pid_state_;
    PIDState heading_pid_state_;
    PIDState lateral_pid_state_;

    // 轨迹跟踪状态
    int current_point_index_ = 0;
    bool trajectory_received_ = false;
};

} // namespace nodes
} // namespace lidar_core