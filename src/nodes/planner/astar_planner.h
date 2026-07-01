// src/nodes/planner/astar_planner.h
// A*路径规划算法
#pragma once

#include "lidar_core/nodes/i_planner_node.h"
#include <vector>
#include <memory>
#include <queue>
#include <unordered_map>

namespace lidar_core {
namespace nodes {

// A*网格节点
struct GridNode {
    int x;          // 网格x坐标
    int y;          // 网格y坐标
    float g;        // 从起点到当前节点的代价
    float h;        // 从当前节点到终点的启发式代价
    float f;        // f = g + h
    int parent_x;   // 父节点x
    int parent_y;   // 父节点y
    
    bool operator>(const GridNode& other) const {
        return f > other.f;
    }
};

// A*规划器参数
struct AStarParams {
    float grid_resolution = 0.5f;    // 网格分辨率 (m)
    int max_grid_x = 300;           // x方向最大网格数
    int max_grid_y = 50;            // y方向最大网格数
    float obstacle_inflation = 0.5f; // 障碍物膨胀半径 (m)
    bool use_diagonal = true;       // 是否允许对角线移动
};

class AStarPlanner : public IPlannerNode {
public:
    AStarPlanner();
    ~AStarPlanner() override;

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

    // A*搜索算法
    bool astarSearch(
        int start_x, int start_y,
        int goal_x, int goal_y,
        const std::vector<std::vector<bool>>& obstacle_grid,
        std::vector<std::pair<int, int>>& path);

    // 计算启发式距离 (八边形距离)
    float heuristic(int x1, int y1, int x2, int y2);

    // 创建障碍物网格
    void createObstacleGrid(
        std::vector<std::vector<bool>>& grid,
        const std::vector<core::Detection>& obstacles);

    // 将网格路径转换为轨迹
    core::Trajectory convertToTrajectory(
        const std::vector<std::pair<int, int>>& grid_path,
        const core::EgoState& ego_state);

    // 平滑轨迹
    void smoothTrajectory(std::vector<core::PathPoint>& trajectory);

    // 辅助函数
    float distanceBetweenPoints(float x1, float y1, float x2, float y2);
    void gridToWorld(int gx, int gy, float& wx, float& wy);
    void worldToGrid(float wx, float wy, int& gx, int& gy);

    PlannerType planner_type_ = PlannerType::LATTICE;
    core::PlanningConfig config_;
    AStarParams astar_params_;
    core::Trajectory latest_trajectory_;
};

} // namespace nodes
} // namespace lidar_core
