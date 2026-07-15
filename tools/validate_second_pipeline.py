#!/usr/bin/env python3
"""
Validate SECOND pipeline: compare PyTorch model BEV with TRT engine output.
"""
import os, sys, warnings, logging
import numpy as np
import torch

sys.path.insert(0, '/home/sevnce/lj/project/OpenPCDet/tools')

def build_model(config_path, checkpoint_path, device='cuda'):
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.datasets import DatasetTemplate
    class DummyDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names):
            super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=False)
        def __len__(self): return 1
        def __getitem__(self, idx): return {}
    cfg_from_yaml_file(config_path, cfg)
    dataset = DummyDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=checkpoint_path, logger=logging.getLogger(), to_cpu=True)
    model.eval().to(device)
    return model, cfg


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    config_path = '/home/sevnce/lj/project/OpenPCDet/tools/cfgs/kitti_models/second.yaml'
    ckpt_path = '/home/sevnce/lj/project/lidar_detection_pipeline/model/SECOND/second_7862.pth'
    plan_path = '/home/sevnce/lj/project/lidar_detection_pipeline/model/second_2d_backbone.plan'

    model, cfg = build_model(config_path, ckpt_path, device)
    print(f'[INFO] Model loaded')

    # Load one KITTI point cloud
    bin_path = '/home/sevnce/lj/project/lidar_detection_pipeline/data/000000.bin'
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    print(f'[INFO] Loaded {bin_path}: {points.shape[0]} points')

    # Get model params
    pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    voxel_size = cfg.DATA_CONFIG.DATA_PROCESSOR[2]['VOXEL_SIZE']
    grid_size = [round((pc_range[3]-pc_range[0])/voxel_size[0]),
                 round((pc_range[4]-pc_range[1])/voxel_size[1]),
                 round((pc_range[5]-pc_range[2])/voxel_size[2])]
    print(f'[INFO] Range: {pc_range}')
    print(f'[INFO] Voxel size: {voxel_size}')
    print(f'[INFO] Grid size: {grid_size}')

    # Voxelize using OpenPCDet's built-in voxelization
    from pcdet.datasets.processor.data_processor import DataProcessor
    # Use the model's voxelization directly
    input_dict = {'points': torch.from_numpy(points).float().to(device)}
    example = {}
    
    # Manually voxelize
    pts = points
    vx = ((pts[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int64)
    vy = ((pts[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int64)
    vz = ((pts[:, 2] - pc_range[2]) / voxel_size[2]).astype(np.int64)
    vx = np.clip(vx, 0, grid_size[0]-1)
    vy = np.clip(vy, 0, grid_size[1]-1)
    vz = np.clip(vz, 0, grid_size[2]-1)
    
    max_voxels = 60000
    max_points = 5
    
    voxel_id = vz * grid_size[1] * grid_size[0] + vy * grid_size[0] + vx
    
    # Group by voxel (capped at max_points per voxel, max_voxels total)
    from collections import defaultdict
    voxel_dict = defaultdict(list)
    for i in range(len(pts)):
        vid = int(voxel_id[i])
        if len(voxel_dict[vid]) < max_points:
            voxel_dict[vid].append(i)
    
    # Select first max_voxels voxels
    sorted_vids = sorted(voxel_dict.keys())[:max_voxels]
    num_voxels = len(sorted_vids)
    
    voxels_np = np.zeros((num_voxels, max_points, 4), dtype=np.float32)
    num_points_np = np.zeros((num_voxels,), dtype=np.int32)
    coords_np = np.zeros((num_voxels, 4), dtype=np.int32)
    coords_float_np = np.zeros((num_voxels, 4), dtype=np.float32)
    
    for idx, vid in enumerate(sorted_vids):
        pt_indices = voxel_dict[vid]
        npts = min(len(pt_indices), max_points)
        for j in range(npts):
            voxels_np[idx, j] = pts[pt_indices[j]]
        num_points_np[idx] = npts
        tmp_vid = vid
        vz_c = tmp_vid // (grid_size[1] * grid_size[0])
        tmp_vid %= (grid_size[1] * grid_size[0])
        vy_c = tmp_vid // grid_size[0]
        vx_c = tmp_vid % grid_size[0]
        coords_np[idx] = [0, vz_c, vy_c, vx_c]
    
    print(f'[INFO] Voxelization: {num_voxels} voxels')
    
    # Run full model
    with torch.no_grad():
        batch_dict = {
            'voxels': torch.from_numpy(voxels_np).float().to(device),
            'voxel_num_points': torch.from_numpy(num_points_np).int().to(device),
            'voxel_coords': torch.from_numpy(coords_np).int().to(device),
            'batch_size': 1,
        }
        batch_dict = model(batch_dict)
    
    # Get BEV features
    spatial_features = batch_dict['spatial_features']  # (1, 256, 200, 176)
    print(f'[INFO] BEV features shape: {spatial_features.shape}')
    bev_np = spatial_features.cpu().numpy()
    print(f'[INFO] BEV range: [{bev_np.min():.4f}, {bev_np.max():.4f}]  mean={bev_np.mean():.6f}  nz={np.count_nonzero(bev_np)}/{bev_np.size}')
    
    # Get model outputs
    if 'batch_cls_preds' in batch_dict:
        cls_preds = batch_dict['batch_cls_preds']
        print(f'[INFO] PyTorch cls: min={cls_preds.min():.4f} max={cls_preds.max():.4f} mean={cls_preds.mean():.6f}')
    
    # Run TRT engine on the same BEV features
    print('\n[INFO] Running TRT 2D backbone...')
    import tensorrt as trt
    
    with open(plan_path, 'rb') as f:
        plan_data = f.read()
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_data)
    context = engine.create_execution_context()
    
    # Use torch CUDA tensors for TRT (data_ptr() gives device pointer)
    bev_tensor = torch.from_numpy(bev_np).contiguous().to(device)
    
    cls_shape = tuple(engine.get_tensor_shape('cls'))
    box_shape = tuple(engine.get_tensor_shape('box'))
    dir_shape = tuple(engine.get_tensor_shape('dir'))
    print(f'[INFO] TRT cls shape: {cls_shape}')
    
    cls_tensor = torch.zeros(cls_shape, dtype=torch.float32, device=device)
    box_tensor = torch.zeros(box_shape, dtype=torch.float32, device=device)
    dir_tensor = torch.zeros(dir_shape, dtype=torch.float32, device=device)
    
    context.set_tensor_address('bev_features', bev_tensor.data_ptr())
    context.set_tensor_address('cls', cls_tensor.data_ptr())
    context.set_tensor_address('box', box_tensor.data_ptr())
    context.set_tensor_address('dir', dir_tensor.data_ptr())
    
    context.execute_async_v3(0)
    torch.cuda.synchronize()
    
    cls_out = cls_tensor.cpu().numpy()
    box_out = box_tensor.cpu().numpy()
    dir_out = dir_tensor.cpu().numpy()
    
    print(f'[INFO] TRT cls range: [{cls_out.min():.4f}, {cls_out.max():.4f}] mean={cls_out.mean():.6f}')
    print(f'[INFO] TRT box range: [{box_out.min():.4f}, {box_out.max():.4f}]')
    print(f'[INFO] TRT dir range: [{dir_out.min():.4f}, {dir_out.max():.4f}]')
    
    print('\n[INFO] Validation complete!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    warnings.filterwarnings('ignore')
    main()
