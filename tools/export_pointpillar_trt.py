#!/usr/bin/env python3
"""
Export OpenPCDet PointPillar model (pointpillar_7728.pth) to ONNX,
then convert to TensorRT engine.

Pipeline expects ONNX with:
  Input:  voxels[40000,32,10], voxel_idxs[40000,4], voxel_num[1]
  Output: cls_preds[1,248,216,18], box_preds[1,248,216,42], dir_cls_preds[1,248,216,12]

The C++ voxelization pre-computes 10 features per point matching OpenPCDet's
PillarVFE internal feature construction (USE_ABSLOTE_XYZ=True):
  [x,y,z,i, x-mean_x, y-mean_y, z-mean_z, x-pillar_cx, y-pillar_cy, z-pillar_cz]

Usage:
  cd /home/sevnce/lj/project/OpenPCDet/tools
  python3 /path/to/export_pointpillar_trt.py
"""

import os, sys, warnings, logging, subprocess
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/sevnce/lj/project/OpenPCDet/tools')

NUM_VOXELS = 40000
MAX_POINTS_PER_VOXEL = 32
NUM_FEATURES = 10

OUTPUT_DIR = '/home/sevnce/lj/project/lidar_detection_pipeline/model/pointpillar'


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


class PointPillarExport(nn.Module):
    """
    Takes pre-computed 10-feature voxels (from C++ voxelization),
    skips offset computation, runs PFN -> Scatter -> 2D Backbone -> Head.
    """
    def __init__(self, pillar_vfe, scatter, backbone_2d, dense_head):
        super().__init__()
        self.pfn_layers = pillar_vfe.pfn_layers
        self.scatter = scatter
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head

    def forward(self, voxels, voxel_idxs, voxel_num):
        # voxels: [B, 32, 10]  (B = max_voxels, pre-computed offsets)
        # voxel_idxs: [B, 4]   (batch, z, y, x)
        # voxel_num: [1]       scalar = number of valid voxels (unused here)

        # Create mask: detect padded points (x,y,z all ~zero -> invalid)
        norm = voxels[:, :, :3].norm(dim=-1, keepdim=True)
        mask = (norm > 1e-8).float()
        features = voxels * mask

        # PFN layers: each output [B, 1, 64]
        for pfn in self.pfn_layers:
            features = pfn(features)

        # Squeeze dim=1 explicitly (avoid dynamic Squeeze for TRT)
        features = features.squeeze(dim=1)

        # Scatter to pseudo-image
        batch_dict = {'pillar_features': features, 'voxel_coords': voxel_idxs}
        batch_dict = self.scatter(batch_dict)
        spatial_features = batch_dict['spatial_features']

        # 2D Backbone
        batch_dict = self.backbone_2d({'spatial_features': spatial_features})
        spatial_features_2d = batch_dict['spatial_features_2d']

        # Detection Head
        cls_preds = self.dense_head.conv_cls(spatial_features_2d)
        box_preds = self.dense_head.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        if self.dense_head.conv_dir_cls is not None:
            dir_cls_preds = self.dense_head.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = torch.zeros(1, 248, 216, 12, dtype=torch.float32, device=cls_preds.device)

        return cls_preds, box_preds, dir_cls_preds


def export():
    config_path = 'cfgs/kitti_models/pointpillar.yaml'
    ckpt_path = '/home/sevnce/lj/project/OpenPCDet/ckpt/pointpillar_7728.pth'
    onnx_path = os.path.join(OUTPUT_DIR, 'pointpillar_7728.onnx')
    engine_path = os.path.join(OUTPUT_DIR, 'pointpillar_7728.engine')
    plan_path = os.path.join(OUTPUT_DIR, 'pointpillar_7728.plan')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    model = build_model(config_path, ckpt_path, device)
    print(f'[INFO] Model loaded')

    wrapper = PointPillarExport(
        model.vfe, model.map_to_bev_module, model.backbone_2d, model.dense_head
    ).to(device).eval()
    print(f'[INFO] Wrapper created')

    # Create dummy input matching C++ voxelization output
    B = NUM_VOXELS
    dummy_voxels = torch.zeros(B, MAX_POINTS_PER_VOXEL, NUM_FEATURES, device=device)
    dummy_voxel_idxs = torch.zeros(B, 4, dtype=torch.int32, device=device)
    dummy_voxel_num = torch.tensor([100], dtype=torch.int32, device=device)

    # Fill 100 valid voxels with synthetic data
    for vi in range(100):
        npts = max(1, min(32, vi % 32 + 5))
        for pi in range(npts):
            dummy_voxels[vi, pi, 0] = float(np.random.uniform(0, 69.12))
            dummy_voxels[vi, pi, 1] = float(np.random.uniform(-39.68, 39.68))
            dummy_voxels[vi, pi, 2] = float(np.random.uniform(-3, 1))
            dummy_voxels[vi, pi, 3] = float(np.random.uniform(0, 30))
            for f in range(4, 10):
                dummy_voxels[vi, pi, f] = float(np.random.randn() * 0.5)
        dummy_voxel_idxs[vi, 0] = 0
        dummy_voxel_idxs[vi, 1] = 0
        dummy_voxel_idxs[vi, 2] = vi % 496
        dummy_voxel_idxs[vi, 3] = vi % 432

    # Test forward
    with torch.no_grad():
        try:
            out = wrapper(dummy_voxels, dummy_voxel_idxs, dummy_voxel_num)
            print(f'[INFO] Test forward:  cls={out[0].shape}  box={out[1].shape}  dir={out[2].shape}')
        except Exception as e:
            print(f'[ERROR] Test failed: {e}')
            import traceback; traceback.print_exc()
            return

    # Export to ONNX (batch_size=1, static shapes)
    print(f'[INFO] Exporting ONNX ...')
    torch.onnx.export(
        wrapper,
        (dummy_voxels, dummy_voxel_idxs, dummy_voxel_num),
        onnx_path,
        input_names=['voxels', 'voxel_idxs', 'voxel_num'],
        output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f'[INFO] ONNX exported: {onnx_path}')

    # Check ONNX
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    custom_ops = set()
    for node in onnx_model.graph.node:
        if node.domain:
            custom_ops.add(f'{node.domain}::{node.op_type}')
    print(f'[INFO] ONNX nodes: {len(onnx_model.graph.node)}, custom ops: {custom_ops}')

    if custom_ops:
        print(f'[WARN] Custom ops found - TRT may need plugins')

    # Convert to TRT engine (done separately to avoid timeout)
    print(f'[INFO] To convert to TRT, run:')
    print(f'  trtexec --onnx={onnx_path} --saveEngine={engine_path} --memPoolSize=workspace:4096')
    print(f'')


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    warnings.filterwarnings('ignore')
    export()
