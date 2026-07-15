#!/usr/bin/env python3
"""
Export SECOND's 2D backbone + detection head to ONNX.
This part uses only standard Conv2d/BN/ReLU ops and can be converted
to TensorRT directly without custom plugins.

The 3D sparse backbone will be implemented separately as a TRT plugin.

Usage:
  cd /home/sevnce/lj/project/OpenPCDet/tools && \\
  PYTHONPATH=/usr/lib/python3.10/dist-packages python3 \\
    /path/to/export_second_2d_backbone.py
"""

import os, sys, warnings, logging
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/sevnce/lj/project/OpenPCDet/tools')


# ======================================================================
# Wrapper for 2D backbone + head
# ======================================================================

class Second2DBackboneHead(nn.Module):
    """
    2D backbone (BaseBEVBackbone) + detection head (AnchorHeadSingle).
    Input:  BEV features (1, 256, H, W)  [from HeightCompression]
    Output: cls (1, H, W, 18), box (1, H, W, 42), dir (1, H, W, 12)
    """
    def __init__(self, backbone_2d, dense_head):
        super().__init__()
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head

    def forward(self, spatial_features):
        batch_dict = {'spatial_features': spatial_features}
        batch_dict = self.backbone_2d(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']

        cls_preds = self.dense_head.conv_cls(spatial_features_2d)
        box_preds = self.dense_head.conv_box(spatial_features_2d)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        if self.dense_head.conv_dir_cls is not None:
            dir_cls_preds = self.dense_head.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = torch.zeros(1, dtype=torch.float32, device=cls_preds.device)

        return cls_preds, box_preds, dir_cls_preds


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


def export():
    config_path = 'cfgs/kitti_models/second.yaml'
    ckpt_path = '/home/sevnce/lj/project/lidar_detection_pipeline/model/SECOND/second_7862.pth'
    output_path = '/home/sevnce/lj/project/lidar_detection_pipeline/model/second_2d_backbone.onnx'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    model = build_model(config_path, ckpt_path, device)
    print(f'[INFO] Model loaded')

    # Create wrapper for 2D backbone + head
    wrapper = Second2DBackboneHead(model.backbone_2d, model.dense_head).to(device).eval()
    print(f'[INFO] Wrapper created')

    # Create dummy BEV input
    # After HeightCompression: (1, 256, H, W) where H=200, W=176
    H, W = 200, 176
    dummy_bev = torch.randn(1, 256, H, W, device=device).float()

    # Test forward
    with torch.no_grad():
        try:
            out = wrapper(dummy_bev)
            print(f'[INFO] Test forward: cls={out[0].shape}, box={out[1].shape}, dir={out[2].shape}')
        except Exception as e:
            print(f'[ERROR] Test failed: {e}')
            import traceback
            traceback.print_exc()
            return

    # Export
    print(f'[INFO] Exporting ONNX to {output_path} ...')
    torch.onnx.export(
        wrapper,
        dummy_bev,
        output_path,
        input_names=['bev_features'],
        output_names=['cls', 'box', 'dir'],
        dynamic_axes={
            'bev_features': {2: 'H', 3: 'W'},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f'[INFO] ONNX exported: {output_path}')

    # Check
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    custom_ops = set()
    for node in onnx_model.graph.node:
        if node.domain:
            custom_ops.add(f'{node.domain}::{node.op_type}')
    print(f'[INFO] ONNX nodes: {len(onnx_model.graph.node)}')
    if custom_ops:
        print(f'[INFO] Custom ops: {custom_ops}')
    else:
        print(f'[INFO] All standard ops - ready for TRT conversion')
    print('[INFO] Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    warnings.filterwarnings('ignore')
    export()
