# Lidar Detection Pipeline

基于 TensorRT 的 PointPillars 3D 目标检测部署，支持 x86 (RTX 3090) 和 Jetson Orin。

## 项目结构

```
lidar_detection_pipeline/
├── src/
│   ├── common/              # TensorRT 封装、数据类型定义
│   ├── core/                # CUDA 检测核心（Voxelization + Backbone + PostProcess）
│   │   ├── lidar-voxelization.cu   # 体素化 + 特征生成
│   │   ├── lidar-backbone.cu       # TRT backbone + FP32 转换
│   │   ├── lidar-postprocess.cu    # 解码 + NMS
│   │   ├── pointpillar-scatter.cu  # PPScatter TRT 插件
│   │   ├── detector.cpp            # Detector 接口封装
│   │   └── detector.hpp
│   └── pipeline/            # 管线层
│       ├── types.hpp        # PointCloud, Detection, Config 定义
│       ├── reader.hpp/cpp   # 输入接口（文件/UDP）
│       ├── bin_reader.cpp   # KITTI .bin 文件读取
│       ├── velodyne_reader.cpp # Velodyne UDP（预留）
│       ├── engine.hpp/cpp   # 推理引擎（支持异步）
│       ├── output.hpp/cpp   # 输出接口工厂
│       ├── file_writer.cpp  # 文件输出
│       ├── callback_output.cpp # 回调输出
│       └── bev_visualizer.cpp   # OpenCV BEV 可视化
├── model/                   # TRT 引擎文件
├── data/                    # KITTI 测试数据
├── out/                     # 检测输出
└── build/                   # 编译输出
```

## 环境依赖

- CUDA 13.1 + cuDNN
- TensorRT 11.0+
- g++-9（CUDA 13.1 兼容性要求）
- OpenCV 4.x（可选，BEV 可视化）
- ROS2 Humble/Foxy（可选）

## 编译

```bash
# 设置环境变量
export CUDA_Inc=/usr/local/cuda-13.1/include/
export CUDA_Lib=/usr/local/cuda-13.1/lib64/
export TensorRT_Inc=/usr/include/x86_64-linux-gnu/
export TensorRT_Lib=/usr/lib/x86_64-linux-gnu/

# 编译
cd build
cmake ..
make -j$(nproc)
```

可选编译选项：
```bash
cmake .. -DWITH_ROS2=ON     # 启用 ROS2 输出
cmake .. -DWITH_OPENCV=OFF  # 禁用 BEV 可视化
```

## 运行

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

./lidar_app \
    --input ../data \
    --model ../model/pointpillar.plan \
    --output-types file \
    --timer
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input <path>` | 输入路径（.bin 文件或目录） | 必填 |
| `--input-type <type>` | 输入类型：`bin`, `velodyne` | `bin` |
| `--model <path>` | TRT 引擎路径 | `../model/pointpillar.plan` |
| `--output <path>` | 输出目录 | `../out` |
| `--output-types <types>` | 输出类型，逗号分隔：`file`, `vis`, `callback` | `file` |
| `--async` | 启用异步推理 | 关闭 |
| `--workers <n>` | 异步工作线程数 | 2 |
| `--score-thresh <f>` | 置信度阈值 | 0.1 |
| `--timer` | 显示每帧耗时 | 关闭 |

## 模型转换流程

### 1. 导出 ONNX

```bash
cd /path/to/OpenPCDet
python tool/export_onnx.py \
    --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --ckpt checkpoint/pointpillar_7728.pth \
    --data_path ../CUDA-PointPillars/data \
    --out_dir ../CUDA-PointPillars/model
```

- 加载 OpenPCDet PointPillars 权重（`.pth`）
- `ModelWrapper` 包装推理链：VFE → map_to_bev → backbone_2d → conv_cls/box/dir_cls
- 输出原始卷积结果（raw conv deltas），不做 sigmoid/exp/anchor 解码
- 输出：`pointpillar_raw.onnx`

### 2. ONNX 图修改

```bash
cd /path/to/CUDA-PointPillars
python tool/modify_onnx.py
```

使用 `onnx_graphsurgeon` 进行图手术：

**后处理裁剪** (`simplify_postprocess`)：
- 截断 conv 后的输出处理节点
- 输出变为 3 个 raw 变量：`cls_preds[1,248,216,18]`、`box_preds[1,248,216,42]`、`dir_cls_preds[1,248,216,12]`

**前处理简化** (`simplify_preprocess`)：
- 插入 VFE 的 linear+BN+ReLU 操作
- 插入 `PPScatterPlugin` 节点（自定义 TRT 插件）
- 输入变为：`voxels[10000,32,4]`、`voxel_idxs[10000,4]`、`voxel_num[1]`

最终输出：`pointpillar.onnx`（~20MB）

### 3. 构建 TRT 引擎

```bash
trtexec \
    --onnx=./model/pointpillar.onnx \
    --fp16 \
    --plugins=build/libpointpillar_core.so \
    --saveEngine=./model/pointpillar.plan \
    --inputIOFormats=fp16:chw,int32:chw,int32:chw
```

最终输出：`pointpillar.plan`（~30MB）

### 转换流程图

```
.pth (OpenPCDet 权重)
  ↓ export_onnx.py
pointpillar_raw.onnx
  ↓ modify_onnx.py (onnx_graphsurgeon)
pointpillar.onnx (嵌入 PPScatter 插件)
  ↓ trtexec --fp16 --plugins=libpointpillar_core.so
pointpillar.plan (TRT 引擎)
```

### BUILD
```
build:
    export CUDA_Inc=/usr/local/cuda-13.1/include/ && export CUDA_Lib=/usr/local/cuda-13.1/lib64/ && export TensorRT_Inc=/usr/include/x86_64-linux-gnu/ && export TensorRT_Lib=/usr/lib/x86_64-linux-gnu/ && cd /home/sevnce/lj/project/lidar_detection_pipeline/build && rm -rf * && cmake .. && make -j$(nproc) 2>&1
```

### RUN
```
run:
    ./lidar_app --input ../data --model ../model/pointpillar.plan --output-types file --output ../out --timer 2>&1 
```

## 数据流

```
.bin (KITTI 格式: x,y,z,intensity)
  ↓ BinFileReader (CPU)
PointCloud {x,y,z,intensity}
  ↓ Voxelization (GPU, FP16)
voxels[10000,32,10], voxel_idxs[10000,4], voxel_num[1]
  ↓ TRT Backbone (GPU, FP16)
cls_preds[1,248,216,18], box_preds[1,248,216,42], dir_cls_preds[1,248,216,12]
  ↓ PostProcess (GPU, FP32)
BoundingBox[] (x,y,z,w,l,h,rt,id,score)
  ↓ DetectionResult 输出
.txt / BEV 可视化 / 回调
```

## 类别与锚框

| 类别 ID | 名称 | 锚框 (w,l,h) | 锚框朝向 |
|---------|------|--------------|----------|
| 0 | Car | 3.9, 1.6, 1.56 | 0.0, π/2 |
| 1 | Pedestrian | 0.8, 0.6, 1.73 | 0.0, π/2 |
| 2 | Cyclist | 1.76, 0.6, 1.73 | 0.0, π/2 |

- 共 6 个锚框（2 个朝向 × 3 个类别）
- 体素大小：0.16m × 0.16m × 4m
- 点云范围：[0, -39.68, -3] ~ [69.12, 39.68, 1]（米）
- 每个体素最多 32 个点

## 性能

- **推理延迟**：~7ms/帧（RTX 3090, FP16, 含预处理）
- **Python vs C++ 精度对比**：99.2% 召回率（125 Python / 126 C++ / 124 匹配）
