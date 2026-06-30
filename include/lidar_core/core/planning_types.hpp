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
    float max_speed = 10.0f;           // 最大速度 (m/s)
    float max_accel = 3.0f;            // 最大加速度 (m/s²)
    float max_decel = 5.0f;            // 最大减速度 (m/s²)
    float max_lateral_accel = 2.0f;    // 最大横向加速度 (m/s²)
    float planning_horizon = 5.0f;     // 规划时域 (s)
    float planning_resolution = 0.1f;  // 路径点间距 (m)
    float obstacle_inflation = 0.5f;   // 障碍物膨胀半径 (m)
    float safety_margin = 1.0f;        // 安全距离 (m)
    float goal_x = 100.0f;            // 目标点x (m)
    float goal_y = 0.0f;              // 目标点y (m)
};

// ============================================================
// Ego Vehicle State - 自车状态
// ============================================================
struct EgoState {
    float x = 0.0f;           // x位置 (m)
    float y = 0.0f;           // y位置 (m)
    float z = 0.0f;           // z位置 (m)
    float heading = 0.0f;     // 航向角 (rad)
    float speed = 0.0f;       // 速度 (m/s)
    float acceleration = 0.0f; // 加速度 (m/s²)
    float steer_angle = 0.0f; // 转向角 (rad)
    uint64_t timestamp_ns = 0; // 时间戳 (ns)
};

} // namespace core
} // namespace lidar_core

#endif // __PLANNING_TYPES_HPP__