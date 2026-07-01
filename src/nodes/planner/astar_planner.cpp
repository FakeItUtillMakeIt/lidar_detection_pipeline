// src/nodes/planner/astar_planner.cpp
// A*路径规划算法实现
#include "3rd_party/log_mgr/log_mgr.h"
#include "astar_planner.h"
#include "node_factory.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <set>

namespace lidar_core {
namespace nodes {

AStarPlanner::AStarPlanner() : IPlannerNode("AStarPlanner") {
}

AStarPlanner::~AStarPlanner() {
    stop();
}

void AStarPlanner::setPlannerType(PlannerType type) {
    planner_type_ = type;
}

PlannerType AStarPlanner::getPlannerType() const {
    return planner_type_;
}

void AStarPlanner::setPlannerConfig(const core::PlanningConfig& config) {
    config_ = config;
    // 从config更新参数
    astar_params_.obstacle_inflation = config.obstacle_inflation;
}

const core::PlanningConfig& AStarPlanner::getPlannerConfig() const {
    return config_;
}

const core::Trajectory& AStarPlanner::getLatestTrajectory() const {
    return latest_trajectory_;
}

bool AStarPlanner::isPlanningFeasible() const {
    return latest_trajectory_.is_feasible;
}

bool AStarPlanner::start() {
    if (running_)
        return true;

    running_ = true;
    LOG_INFO_FMT("[AStarPlanner] Started, grid_resolution={}, goal=({},{})", 
                 astar_params_.grid_resolution, config_.goal_x, config_.goal_y);
    return true;
}

void AStarPlanner::stop() {
    if (!running_)
        return;
    running_ = false;
    latest_trajectory_.points.clear();
    LOG_INFO_FMT("[AStarPlanner] Stopped");
}

void AStarPlanner::pushData(std::shared_ptr<core::BasePacket> packet) {
    auto det_packet = std::dynamic_pointer_cast<core::DetectionPacket>(packet);
    if (!det_packet) {
        LOG_ERROR_FMT("[AStarPlanner] Invalid packet type");
        return;
    }

    // 自车状态 (相对坐标系下始终在原点)
    core::EgoState ego_state;
    ego_state.x = 0.0f;
    ego_state.y = 0.0f;
    ego_state.heading = 0.0f;
    ego_state.speed = 0.0f;
    ego_state.timestamp_ns = packet->timestamp_ns;

    // 障碍物
    std::vector<core::Detection> obstacles = det_packet->detections;

    // 生成规划轨迹
    latest_trajectory_ = generateTrajectory(obstacles, ego_state);

    LOG_INFO_FMT("[AStarPlanner] Generated trajectory: points={}, feasible={}, status={}", 
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

core::Trajectory AStarPlanner::generateTrajectory(
    const std::vector<core::Detection>& obstacles,
    const core::EgoState& ego_state) {
    
    core::Trajectory trajectory;

    LOG_INFO_FMT("[AStarPlanner] Generating trajectory: start=({},{}), goal=({},{}), obstacles={}",
                 ego_state.x, ego_state.y, config_.goal_x, config_.goal_y, obstacles.size());

    // 创建障碍物网格
    std::vector<std::vector<bool>> obstacle_grid;
    createObstacleGrid(obstacle_grid, obstacles);

    // 转换起点和终点到网格坐标
    int start_gx, start_gy;
    int goal_gx, goal_gy;
    worldToGrid(ego_state.x, ego_state.y, start_gx, start_gy);
    worldToGrid(config_.goal_x, config_.goal_y, goal_gx, goal_gy);

    // 限制在网格范围内
    goal_gx = std::min(goal_gx, astar_params_.max_grid_x - 1);
    goal_gy = std::min(goal_gy, astar_params_.max_grid_y - 1);

    LOG_INFO_FMT("[AStarPlanner] Grid: start=({},{}), goal=({},{}), grid_size={}x{}", 
                 start_gx, start_gy, goal_gx, goal_gy, 
                 astar_params_.max_grid_x, astar_params_.max_grid_y);

    // A*搜索
    std::vector<std::pair<int, int>> grid_path;
    bool found = astarSearch(start_gx, start_gy, goal_gx, goal_gy, obstacle_grid, grid_path);

    if (!found || grid_path.empty()) {
        trajectory.is_feasible = false;
        trajectory.status = "No path found";
        LOG_WARN_FMT("[AStarPlanner] No path found");
        return trajectory;
    }

    LOG_INFO_FMT("[AStarPlanner] Found path with {} points", grid_path.size());

    // 转换为轨迹
    trajectory = convertToTrajectory(grid_path, ego_state);
    trajectory.is_feasible = true;
    trajectory.status = "Path found";

    return trajectory;
}

bool AStarPlanner::astarSearch(
    int start_x, int start_y,
    int goal_x, int goal_y,
    const std::vector<std::vector<bool>>& obstacle_grid,
    std::vector<std::pair<int, int>>& path) {
    
    // 优先队列 (最小堆)
    std::priority_queue<GridNode, std::vector<GridNode>, std::greater<GridNode>> open_list;
    
    // 已访问节点
    std::vector<std::vector<bool>> closed_list(
        astar_params_.max_grid_x, 
        std::vector<bool>(astar_params_.max_grid_y, false));
    
    // g值表
    std::vector<std::vector<float>> g_score(
        astar_params_.max_grid_x, 
        std::vector<float>(astar_params_.max_grid_y, std::numeric_limits<float>::max()));
    
    // 父节点表 (用于回溯路径)
    std::vector<std::vector<std::pair<int, int>>> parent(
        astar_params_.max_grid_x, 
        std::vector<std::pair<int, int>>(astar_params_.max_grid_y, {-1, -1}));

    // 起点
    float h = heuristic(start_x, start_y, goal_x, goal_y);
    GridNode start_node = {start_x, start_y, 0.0f, h, h, -1, -1};
    open_list.push(start_node);
    g_score[start_x][start_y] = 0.0f;

    // 8方向移动 (包括对角线)
    int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    float cost[] = {1.414f, 1.0f, 1.414f, 1.0f, 1.0f, 1.414f, 1.0f, 1.414f};
    
    // 如果不允许对角线，只用4方向
    if (!astar_params_.use_diagonal) {
        dx[0] = 0; dy[0] = -1; cost[0] = 1.0f;
        dx[1] = -1; dy[1] = 0; cost[1] = 1.0f;
        dx[2] = 0; dy[2] = 1; cost[2] = 1.0f;
        dx[3] = 1; dy[3] = 0; cost[3] = 1.0f;
    }

    int iterations = 0;
    int max_iterations = astar_params_.max_grid_x * astar_params_.max_grid_y;

    while (!open_list.empty() && iterations < max_iterations) {
        iterations++;
        
        GridNode current = open_list.top();
        open_list.pop();

        // 到达目标
        if (current.x == goal_x && current.y == goal_y) {
            // 回溯路径
            int cx = goal_x, cy = goal_y;
            while (cx != -1 && cy != -1) {
                path.push_back({cx, cy});
                auto p = parent[cx][cy];
                cx = p.first;
                cy = p.second;
            }
            std::reverse(path.begin(), path.end());
            return true;
        }

        // 已访问过
        if (closed_list[current.x][current.y]) {
            continue;
        }
        closed_list[current.x][current.y] = true;

        // 探索邻居
        int directions = astar_params_.use_diagonal ? 8 : 4;
        for (int i = 0; i < directions; i++) {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];

            // 检查边界
            if (nx < 0 || nx >= astar_params_.max_grid_x || 
                ny < 0 || ny >= astar_params_.max_grid_y) {
                continue;
            }

            // 检查障碍物
            if (obstacle_grid[nx][ny]) {
                continue;
            }

            // 已访问
            if (closed_list[nx][ny]) {
                continue;
            }

            // 计算新的g值
            float new_g = current.g + cost[i];
            
            if (new_g < g_score[nx][ny]) {
                g_score[nx][ny] = new_g;
                float h = heuristic(nx, ny, goal_x, goal_y);
                float f = new_g + h;
                
                GridNode neighbor = {nx, ny, new_g, h, f, current.x, current.y};
                open_list.push(neighbor);
                parent[nx][ny] = {current.x, current.y};
            }
        }
    }

    return false;  // 未找到路径
}

float AStarPlanner::heuristic(int x1, int y1, int x2, int y2) {
    // 八边形距离 (Octile distance)
    int dx = std::abs(x1 - x2);
    int dy = std::abs(y1 - y2);
    return std::max(dx, dy) + (std::sqrt(2.0f) - 1.0f) * std::min(dx, dy);
}

void AStarPlanner::createObstacleGrid(
    std::vector<std::vector<bool>>& grid,
    const std::vector<core::Detection>& obstacles) {
    
    // 初始化网格 (全部为false)
    grid.assign(astar_params_.max_grid_x, 
                std::vector<bool>(astar_params_.max_grid_y, false));

    // 将障碍物标记到网格
    for (const auto& obs : obstacles) {
        // 计算障碍物的边界框
        float cos_a = cos(obs.rt);
        float sin_a = sin(obs.rt);
        float half_l = obs.l / 2.0f + astar_params_.obstacle_inflation;
        float half_w = obs.w / 2.0f + astar_params_.obstacle_inflation;

        // 获取四个角点
        float corners_x[4], corners_y[4];
        corners_x[0] = obs.x - half_l * cos_a - half_w * sin_a;
        corners_y[0] = obs.y - half_l * sin_a + half_w * cos_a;
        corners_x[1] = obs.x + half_l * cos_a - half_w * sin_a;
        corners_y[1] = obs.y + half_l * sin_a + half_w * cos_a;
        corners_x[2] = obs.x + half_l * cos_a + half_w * sin_a;
        corners_y[2] = obs.y + half_l * sin_a - half_w * cos_a;
        corners_x[3] = obs.x - half_l * cos_a + half_w * sin_a;
        corners_y[3] = obs.y - half_l * sin_a - half_w * cos_a;

        // 找到边界框范围
        float min_x = *std::min_element(corners_x, corners_x + 4);
        float max_x = *std::max_element(corners_x, corners_x + 4);
        float min_y = *std::min_element(corners_y, corners_y + 4);
        float max_y = *std::max_element(corners_y, corners_y + 4);

        // 转换为网格坐标
        int gx_min, gy_min, gx_max, gy_max;
        worldToGrid(min_x, min_y, gx_min, gy_min);
        worldToGrid(max_x, max_y, gx_max, gy_max);

        // 标记网格
        for (int gx = gx_min; gx <= gx_max; gx++) {
            for (int gy = gy_min; gy <= gy_max; gy++) {
                if (gx >= 0 && gx < astar_params_.max_grid_x &&
                    gy >= 0 && gy < astar_params_.max_grid_y) {
                    grid[gx][gy] = true;
                }
            }
        }
    }

    // 添加边界限制
    if (config_.enable_boundary) {
        int left_gx, left_gy, right_gx, right_gy;
        worldToGrid(0.0f, config_.boundary_left, left_gx, left_gy);
        worldToGrid(0.0f, config_.boundary_right, right_gx, right_gy);

        // 标记边界外的区域
        for (int gx = 0; gx < astar_params_.max_grid_x; gx++) {
            for (int gy = 0; gy < astar_params_.max_grid_y; gy++) {
                float wy;
                float wx;
                gridToWorld(gx, gy, wx, wy);
                if (wy > config_.boundary_left || wy < config_.boundary_right) {
                    grid[gx][gy] = true;
                }
            }
        }
    }
}

core::Trajectory AStarPlanner::convertToTrajectory(
    const std::vector<std::pair<int, int>>& grid_path,
    const core::EgoState& ego_state) {
    
    core::Trajectory trajectory;
    std::vector<core::PathPoint> path_points;

    // 将网格路径转换为世界坐标
    for (size_t i = 0; i < grid_path.size(); i++) {
        float wx, wy;
        gridToWorld(grid_path[i].first, grid_path[i].second, wx, wy);

        core::PathPoint pt;
        pt.x = wx;
        pt.y = wy;
        pt.z = 0.0f;

        // 计算航向角
        if (i > 0) {
            float prev_wx, prev_wy;
            gridToWorld(grid_path[i-1].first, grid_path[i-1].second, prev_wx, prev_wy);
            pt.heading = std::atan2(wy - prev_wy, wx - prev_wx);
        } else {
            pt.heading = ego_state.heading;
        }

        pt.curvature = 0.0f;
        pt.speed = config_.max_speed;
        pt.acceleration = 0.0f;
        pt.relative_time = static_cast<float>(i) * astar_params_.grid_resolution / config_.max_speed;

        path_points.push_back(pt);
    }

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

void AStarPlanner::smoothTrajectory(std::vector<core::PathPoint>& trajectory) {
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

float AStarPlanner::distanceBetweenPoints(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

void AStarPlanner::gridToWorld(int gx, int gy, float& wx, float& wy) {
    wx = gx * astar_params_.grid_resolution;
    wy = (gy - astar_params_.max_grid_y / 2) * astar_params_.grid_resolution;
}

void AStarPlanner::worldToGrid(float wx, float wy, int& gx, int& gy) {
    gx = static_cast<int>(wx / astar_params_.grid_resolution);
    gy = static_cast<int>(wy / astar_params_.grid_resolution) + astar_params_.max_grid_y / 2;
}

// 注册节点
REGISTER_NODE("astar_planner", AStarPlanner)

} // namespace nodes
} // namespace lidar_core
