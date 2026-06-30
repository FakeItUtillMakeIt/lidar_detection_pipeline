#ifndef __PLANNING_TYPES_HPP__
#define __PLANNING_TYPES_HPP__

#include <cstdint>
#include <string>
#include <vector>

namespace lidar_core {
namespace core {

// ============================================================
// Path Point - 规划轨迹点
// ============================================================
struct PathPoint {
    float x = 0.0f;           // x位置 (m)
    float y = 0.0f;           // y位置 (m)
    float z = 0.0f;           // z位置 (m)
    float heading = 0.0f;     // 航向角 (rad)
    float curvature = 0.0f;   // 曲率 (1/m)
    float speed = 0.0f;       // 速度 (m/s)
    float acceleration = 0.0f; // 加速度 (m/s²)
    float relative_time = 0.0f; // 相对时间 (s)
};

// ============================================================
// Trajectory - 规划轨迹
// ============================================================
struct Trajectory {
    std::vector<PathPoint> points;    // 轨迹点序列
    float total_time = 0.0f;          // 轨迹总时间 (s)
    float total_length = 0.0f;        // 轨迹总长度 (m)
    bool is_feasible = false;         // 轨迹是否可行
    std::string status;               // 规划状态信息
};

// ============================================================
// Planning Config - 规划器配置
// ============================================================
struct PlanningConfig {
    // 坐标系模式
    bool use_relative_frame = true;      // true: 相对坐标系(自车始终在原点), false: 全局坐标系
    
    // 速度与加速度约束
    float max_speed = 10.0f;           // 最大速度 (m/s)
    float max_accel = 3.0f;            // 最大加速度 (m/s²)
    float max_decel = 5.0f;            // 最大减速度 (m/s²)
    float max_lateral_accel = 2.0f;    // 最大横向加速度 (m/s²)
    
    // 规划参数
    float planning_horizon = 5.0f;     // 规划时域 (s)
    float planning_resolution = 0.1f;  // 路径点间距 (m)
    int max_planning_points = 100;     // 最大路径点数量
    
    // 障碍物参数
    float obstacle_inflation = 0.5f;   // 障碍物膨胀半径 (m)
    float safety_margin = 1.0f;        // 安全距离 (m)
    
    // 势场参数
    float attractive_gain = 1.0f;      // 引力增益 (增大以避免局部最小值)
    float repulsive_gain = 10.0f;      // 斥力增益 (减小以平衡引力)
    float obstacle_influence_dist = 5.0f; // 障碍物影响距离 (m)
    
    // 边界限制 (道路边界)
    bool enable_boundary = false;       // 是否启用边界限制
    float boundary_left = 3.75f;       // 左边界距离 (m)，默认3.75m (一条车道)
    float boundary_right = -3.75f;     // 右边界距离 (m)
    float boundary_start_x = 0.0f;     // 边界起始x (m)
    float boundary_end_x = 200.0f;     // 边界结束x (m)
    float boundary_inflation = 0.3f;   // 边界膨胀半径 (m)
    float boundary_repulsive_gain = 50.0f; // 边界斥力增益
    float boundary_influence_dist = 2.0f;  // 边界影响距离 (m)
    
    // 目标点 (相对坐标系下相对自车, 全局坐标系下绝对位置)
    float goal_x = 100.0f;            // 目标点x (m)
    float goal_y = 0.0f;              // 目标点y (m)
    
    // 控制参数
    bool enable_control = true;        // 是否启用控制输出
    float control_frequency = 10.0f;   // 控制频率 (Hz)
};

// ============================================================
// Ego Vehicle State - 自车状态
// ============================================================
struct EgoState {
    // 位姿
    float x = 0.0f;           // x位置 (m)
    float y = 0.0f;           // y位置 (m)
    float z = 0.0f;           // z位置 (m)
    float heading = 0.0f;     // 航向角 (rad)
    
    // 运动状态
    float speed = 0.0f;       // 速度 (m/s)
    float acceleration = 0.0f; // 加速度 (m/s²)
    float steer_angle = 0.0f; // 转向角 (rad)
    float yaw_rate = 0.0f;    // 航向角速度 (rad/s)
    
    // 车辆参数
    float wheelbase = 2.7f;   // 轴距 (m) - 用于运动学模型
    
    // 时间戳
    uint64_t timestamp_ns = 0; // 时间戳 (ns)
    float dt = 0.1f;          // 时间间隔 (s)
};

// ============================================================
// Control Command - 控制指令
// ============================================================
struct ControlCommand {
    float throttle = 0.0f;    // 油门 [0, 1]
    float brake = 0.0f;       // 刹车 [0, 1]
    float steering = 0.0f;    // 转向 [-1, 1] (左负右正)
    
    // 控制状态
    bool is_valid = false;    // 控制指令是否有效
    std::string status;       // 控制状态信息
    uint64_t timestamp_ns = 0; // 时间戳 (ns)
};

} // namespace core
} // namespace lidar_core

#endif // __PLANNING_TYPES_HPP__