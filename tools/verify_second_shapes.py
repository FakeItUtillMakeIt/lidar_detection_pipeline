#!/usr/bin/env python3
"""Run a forward pass through the full SECOND model to verify all shapes match."""
import os, sys, warnings, logging
import torch
import numpy as np

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
    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_path = 'cfgs/kitti_models/second.yaml'
ckpt_path = '/home/sevnce/lj/project/lidar_detection_pipeline/model/SECOND/second_7862.pth'
model = build_model(config_path, ckpt_path, device)

bb3d = model.backbone_3d
bb2d = model.backbone_2d
dense_head = model.dense_head

# Build a realistic input
import spconv.pytorch as spconv
sparse_shape = model.dataset.grid_size[::-1]
batch_size = 1

N = 5000
coords = torch.zeros(N, 4, dtype=torch.int32, device=device)
coords[:, 1] = torch.randint(0, sparse_shape[0], (N,), device=device)
coords[:, 2] = torch.randint(0, sparse_shape[1], (N,), device=device)
coords[:, 3] = torch.randint(0, sparse_shape[2], (N,), device=device)
coords = torch.unique(coords, dim=0)
N = coords.shape[0]
features = torch.randn(N, 4, device=device)

sp_tensor = spconv.SparseConvTensor(features, coords, list(sparse_shape), batch_size)

# Run forward through backbone_3d
batch_dict = {'voxel_features': features, 'voxel_coords': coords, 'batch_size': batch_size}
with torch.no_grad():
    batch_dict = bb3d(batch_dict)

enc = batch_dict['encoded_spconv_tensor']
print(f'Backbone 3D output: features={enc.features.shape}, indices={enc.indices.shape}, spatial={enc.spatial_shape}')

# HeightCompression (simulate)
dense = enc.dense()
print(f'Dense: {dense.shape}')
N, C, D, H, W = dense.shape
spatial_features = dense.view(N, C * D, H, W)
print(f'HeightCompression: {spatial_features.shape}  (N={N}, C*D={C*D}, H={H}, W={W})')

# 2D backbone
batch_dict['spatial_features'] = spatial_features
batch_dict = bb2d(batch_dict)
sf2d = batch_dict['spatial_features_2d']
print(f'2D backbone output: {sf2d.shape}')

# Dense head
cls = dense_head.conv_cls(sf2d)
box = dense_head.conv_box(sf2d)
dir = dense_head.conv_dir_cls(sf2d)
print(f'Head output: cls={cls.shape}, box={box.shape}, dir={dir.shape}')
