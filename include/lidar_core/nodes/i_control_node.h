// include/lidar_core/nodes/i_control_node.h
#pragma once

#include "lidar_core/core/node.h"
#include "lidar_core/core/planning_types.hpp"
#include <string>

namespace lidar_core {
namespace nodes {

// 控制器类型
enum class ControllerType {
    PID = 0,           // PID控制器
    MPC = 1,           // 模型预测控制
    LQR = 2,           // 线性二次调节器
};

// PID参数
struct PIDParams {
    float kp_speed = 1.0f;        // 速度PID比例增益
    float ki_speed = 0.1f;        // 速度PID积分增益
    float kd_speed = 0.05f;       // 速度PID微分增益
    
    float kp_heading = 1.0f;      // 航向PID比例增益
    float ki_heading = 0.0f;      // 航向PID积分增益
    float kd_heading = 0.1f;      // 航向PID微分增益
    
    float kp_lateral = 0.5f;      // 横向PID比例增益
    float ki_lateral = 0.0f;      // 横向PID积分增益
    float kd_lateral = 0.05f;     // 横向PID微分增益
    
    float max_integral = 1.0f;    // 积分限幅
    float dt = 0.1f;              // 控制周期 (s)
};

// 控制节点接口
class IControlNode : public core::Node {
public:
    using core::Node::Node;

    // 设置控制器类型
    virtual void setControllerType(ControllerType type) = 0;
    virtual ControllerType getControllerType() const = 0;

    // 设置PID参数
    virtual void setPIDParams(const PIDParams& params) = 0;
    virtual const PIDParams& getPIDParams() const = 0;

    // 更新自车状态
    virtual void updateEgoState(const core::EgoState& state) = 0;

    // 获取最新控制指令
    virtual const core::ControlCommand& getLatestCommand() const = 0;

    // 重置控制器
    virtual void reset() = 0;
};

} // namespace nodes
} // namespace lidar_core