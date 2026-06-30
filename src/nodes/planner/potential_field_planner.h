// src/nodes/planner/potential_field_planner.h
// 势场法路径规划
#pragma once

#include "lidar_core/nodes/i_planner_node.h"
#include <vector>
#include <memory>

namespace lidar_core {
namespace nodes {

// 势场参数
struct PotentialFieldParams {
    float attractive_gain = 1.0f;       // 引力增益 (增大以避免局部最小值)
    float repulsive_gain = 10.0f;       // 斥力增益 (减小以平衡引力)
    float obstacle_influence_dist = 5.0f; // 障碍物影响距离 (m)
    float step_size = 0.5f;            // 梯度下降步长 (m)
    int max_iterations = 200;          // 最大迭代次数
    float convergence_threshold = 0.1f; // 收敛阈值（降低以避免误判）
};

class PotentialFieldPlanner : public IPlannerNode {
public:
    PotentialFieldPlanner();
    ~PotentialFieldPlanner() override;

    // IPlannerNode 接口
    void setPlannerType(PlannerType type) override;
    PlannerType getPlannerType() const override;
    void setPlannerConfig(const core::PlanningConfig& config) override;
    const core::PlanningConfig& getPlannerConfig() const override;
    const core::Trajectory& getLatestTrajectory() const override;
    bool isPlanningFeasible() const override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    // 核心规划算法
    core::Trajectory generateTrajectory(
        const std::vector<core::Detection>& obstacles,
        const core::EgoState& ego_state);

    // 计算引力 (目标点吸引)
    void computeAttractiveForce(
        float& fx, float& fy,
        float x, float y,
        float goal_x, float goal_y);

    // 计算斥力 (障碍物排斥)
    void computeRepulsiveForce(
        float& fx, float& fy,
        float x, float y,
        const std::vector<core::Detection>& obstacles);

    // 计算边界斥力 (道路边界排斥)
    void computeBoundaryForce(
        float& fx, float& fy,
        float x, float y);

    // 检查是否在边界外
    bool isOutsideBoundary(float x, float y);

    // 检查碰撞
    bool checkCollision(
        float x, float y,
        const std::vector<core::Detection>& obstacles);

    // 计算轨迹点速度 (考虑运动学约束)
    void computeTrajectorySpeed(
        std::vector<core::PathPoint>& trajectory);

    // 平滑轨迹
    void smoothTrajectory(
        std::vector<core::PathPoint>& trajectory);

    // 辅助函数
    float distanceBetweenPoints(float x1, float y1, float x2, float y2);
    float normalizeAngle(float angle);

    PlannerType planner_type_ = PlannerType::POTENTIAL_FIELD;
    core::PlanningConfig config_;
    PotentialFieldParams pf_params_;
    core::Trajectory latest_trajectory_;
    core::EgoState ego_state_;
};

} // namespace nodes
} // namespace lidar_core