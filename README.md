# Lidar Detection Pipeline

基于 TensorRT 的 PointPillars 3D 目标检测部署管线，支持 x86 (RTX 3090)。

## 项目结构

```
lidar_detection_pipeline/
├── src/
│   ├── main.cpp                    # 入口：JSON config → Pipeline
│   ├── core/                       # CUDA 检测核心（由 src/nodes/infer/ 引用）
│   │   ├── common/
│   │   │   ├── tensorrt.cpp/.hpp   # TensorRT 引擎封装
│   │   │   ├── check.hpp           # CUDA 错误检查
│   │   │   ├── launch.cuh          # 核函数启动辅助
│   │   │   └── dtype.hpp           # 数据类型
│   │   └── nodes/
│   │       ├── i_source_node.h     # 源节点接口
│   │       ├── i_infer_node.h      # 推理节点接口
│   │       └── i_output_node.h     # 输出节点接口
│   ├── nodes/
│   │   ├── infer/
│   │   │   ├── lidar-voxelization.cu/.hpp  # GPU 体素化（FP16→10特征）
│   │   │   ├── lidar-backbone.cu/.hpp      # TRT 引擎推理 + FP16→FP32
│   │   │   ├── lidar-postprocess.cu/.hpp   # Anchor 解码 + NMS（GPU）
│   │   │   ├── pointpillar-scatter.cu      # (已废弃) 旧版 PPScatter 插件
│   │   │   ├── detector.cpp/.hpp           # Detector 接口封装
│   │   │   ├── engine.hpp/.cpp             # PointPillarsEngine（同步/异步）
│   │   │   ├── detection_infer_node.h/.cpp # 推理节点实现
│   │   │   └── second-*                    # SECOND 管线（独立，不用于 PointPillar）
│   │   ├── source/
│   │   │   ├── bin_source_node.h/.cpp      # KITTI .bin 文件读取
│   │   │   └── bin_reader.cpp              # .bin 解析
│   │   ├── output/
│   │   │   ├── file_output_node.h/.cpp     # 文本文件输出
│   │   │   └── bev_visualizer_node.h/.cpp  # OpenCV BEV 可视化
│   │   ├── track/                 # 跟踪 (OC-SORT)
│   │   ├── attribute/             # 属性计算
│   │   ├── planner/               # 路径规划
│   │   ├── control/               # 控制
│   │   └── registry/              # 节点工厂注册
│   ├── config/                    # JSON 管线配置
│   ├── model/                     # TRT 引擎文件
│   ├── data/                      # KITTI 测试数据（000000.bin ~ 000009.bin）
│   ├── out/                       # 检测输出
│   └── tools/                     # 模型导出/验证工具
```

## 架构

### 管线（Pipeline）模式

基于配置文件驱动，JSON 定义节点（nodes）和数据流（edges）。支持两种执行模式：

- **同步模式**：主线程逐帧读取→推理→输出
- **异步模式**：读取/推理/输出分线程并行

启动入口：

```bash
./lidar_app --config ../config/pipeline.json       # 同步
./lidar_app --config ../config/pipeline.json --async  # 异步
```

### 数据流

```
.bin (KITTI 格式: x,y,z,intensity)
  ↓ BinSourceNode (CPU)
PointCloudPacket
  ↓ DetectionInferNode (GPU)
    ├── Voxelization (CUDA):  xyz→10 特征（含偏移量）
    ├── TRT Engine (FP32):    PFN→Scatter→2D Backbone→Head
    └── PostProcess (CUDA):   Anchor 解码 + NMS
Detection[]
  ↓ FileOutputNode / BEVVisualizerNode
.txt / bev_*.png
```

### 管线配置示例

`pipeline_simple.json`:
```json
{
    "pipeline": {"id": "lidar_detection_simple"},
    "nodes": [
        {"id": "source", "type": "bin_source",
         "params": {"input_path": "../data", "input_type": "bin"}},
        {"id": "infer", "type": "detection_infer",
         "params": {"model_path": "../model/pointpillar.plan",
                    "score_thresh": 0.3, "nms_thresh": 0.01}},
        {"id": "file_output", "type": "file_output",
         "params": {"output_dir": "./out/simple", "output_type": "file"}}
    ],
    "edges": [
        {"from": "source", "to": "infer"},
        {"from": "infer", "to": "file_output"}
    ]
}
```

### 支持的节点类型

| 类型 | 功能 | 配置参数 |
|------|------|----------|
| `bin_source` | 读取 KITTI .bin 文件/目录 | `input_path`, `input_type` |
| `detection_infer` | PointPillar 推理 | `model_path`, `score_thresh`, `nms_thresh` |
| `file_output` | 输出检测结果的 .txt 文件 | `output_dir`, `output_type` |
| `bev_visualizer` | BEV 俯视图可视化 | `output_dir` |
| `tracker` | OC-SORT 跟踪 | `det_thresh`, `max_age`, `iou_threshold` |
| `attribute` | 属性计算 | `dt` |

## 环境依赖

- CUDA 13.1 + cuDNN
- TensorRT 11.0+
- g++-9
- OpenCV 4.x（BEV 可视化必需）
- CMake ≥ 3.18
- nlohmann-json（submodule）

## 编译

```bash
# 设置环境变量
export CUDA_Inc=/usr/local/cuda-13.1/include/
export CUDA_Lib=/usr/local/cuda-13.1/lib64/
export TensorRT_Inc=/usr/include/x86_64-linux-gnu/
export TensorRT_Lib=/usr/lib/x86_64-linux-gnu/

# 编译
cd build
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 \
      -DCMAKE_CUDA_ARCHITECTURES=86 ..
make -j$(nproc)
```

可选选项：
```bash
cmake .. -DWITH_ROS2=ON     # ROS2 输出
cmake .. -DWITH_OPENCV=OFF  # 禁用 BEV 可视化
```

## 运行

```bash
cd build
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 简单检测（同步）
./lidar_app --config ../config/pipeline_simple.json

# 含可视化 + 跟踪
./lidar_app --config ../config/pipeline_final.json

# 异步模式
./lidar_app --config ../config/pipeline.json --async
```

## 模型转换

当前模型使用 OpenPCDet 官方权重 `pointpillar_7728.pth`，导出为端到端 ONNX 后构建 TRT 引擎。

### 导出流程

```bash
# 1. 导出 ONNX（PFN + Scatter + 2D Backbone + Head）
cd /home/sevnce/lj/project/OpenPCDet/tools
python3 ../path/to/lidar_detection_pipeline/tools/export_pointpillar_trt.py

# 2. 构建 TRT 引擎
trtexec \
    --onnx=pointpillar_7728.onnx \
    --saveEngine=pointpillar_7728.engine \
    --memPoolSize=workspace:4096

# 3. 替换引擎
cp pointpillar_7728.engine pointpillar.engine
cp pointpillar_7728.engine pointpillar.plan
```

### ONNX 模型规格

| 项目 | 说明 |
|------|------|
| 输入 | `voxels[40000,32,10]`, `voxel_idxs[40000,4]` |
| 输出 | `cls_preds[1,248,216,18]`, `box_preds[1,248,216,42]`, `dir_cls_preds[1,248,216,12]` |
| 节点数 | 173（全部标准 ONNX op，无自定义插件） |
| 大小 | 71MB（FP32） |

10 个特征构造（匹配 OpenPCDet `USE_ABSLOTE_XYZ=True`）：
```
x, y, z, intensity,
x-μ_x, y-μ_y, z-μ_z,        # f_cluster
x-cx, y-cy, z-cz             # f_center
```

### 历史说明

旧版模型曾使用 `onnx_graphsurgeon` 图修改 + `PPScatter` 自定义 TRT 插件（`pointpillar-scatter.cu`），并在 C++ 中用 SECOND 3D 稀疏卷积作为 Backbone。当前架构已改为**单一 PointPillar TRT 引擎**，包含完整的 PFN→Scatter→2D Backbone→Head，无需插件和图手术。

## 检测参数

### 点云参数

| 参数 | 值 |
|------|-----|
| 点云范围 X | [0, 69.12] m |
| 点云范围 Y | [-39.68, 39.68] m |
| 点云范围 Z | [-3, 1] m |
| 体素大小 XY | 0.16 m |
| 体素大小 Z | 4 m |
| 最大体素数 | 40000 |
| 每体素最多点数 | 32 |

### 类别与锚框

| 类别 ID | 名称 | 锚框 (w,l,h) | 锚框朝向 |
|---------|------|--------------|----------|
| 0 | Car | 3.9, 1.6, 1.56 | 0.0, π/2 |
| 1 | Pedestrian | 0.8, 0.6, 1.73 | 0.0, π/2 |
| 2 | Cyclist | 1.76, 0.6, 1.73 | 0.0, π/2 |

- 共 6 个锚框（2 朝向 × 3 类别）
- 锚框底部高度偏移：[-1.78, -0.6, -0.6]

## 输出格式

检测结果写入 `.txt` 文件，每行格式：
```
x y z w l h rt id score -1 0 0 0 0
```
其中 `(x,y,z)` 为中心坐标，`(w,l,h)` 为尺寸，`rt` 为朝向角（弧度），`id` 为类别 ID，`score` 为置信度。

## BEV 可视化

`bev_visualizer_node` 生成俯视 BEV 图像（PNG），显示点云（灰度）和检测框（彩色），z > -1.5m 的框高亮显示。

## 性能

| 阶段 | 延迟 | 说明 |
|------|------|------|
| Voxelization + Copy | ~1ms | GPU FP16→FP32 转换 |
| TRT Inference | ~4.5ms | FP32，RTX 3090 |
| PostProcess | <1ms | Anchor 解码 + NMS |
| **合计** | **~6ms/帧** | ~170 FPS |
