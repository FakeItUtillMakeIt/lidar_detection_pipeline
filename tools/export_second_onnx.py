#!/usr/bin/env python3
"""
Export SECOND (OpenPCDet) to ONNX with custom spconv ops.

Each spconv op (SubMConv3d, SparseConv3d, SparseToDense) is wrapped in a
torch.autograd.Function with ONNX symbolic, producing custom ONNX ops
that will be implemented as TensorRT plugins.

Usage:
  cd /home/sevnce/lj/project/OpenPCDet/tools && \\
  PYTHONPATH=/usr/lib/python3.10/dist-packages python3 \\
    /path/to/lidar_detection_pipeline/tools/export_second_onnx.py
"""

import os, sys, warnings, logging
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/sevnce/lj/project/OpenPCDet/tools')


def generate_valid_coords(N, spatial_shape, device='cuda'):
    """Generate N unique, sorted sparse coordinates within spatial_shape."""
    import random
    Z, Y, X = spatial_shape
    coords_set = set()
    max_attempts = N * 10
    attempts = 0
    while len(coords_set) < N and attempts < max_attempts:
        z = random.randint(0, Z - 1)
        y = random.randint(0, Y - 1)
        x = random.randint(0, X - 1)
        coords_set.add((0, z, y, x))
        attempts += 1
    _N = len(coords_set)
    coords = torch.tensor(sorted(list(coords_set)), dtype=torch.int32, device=device)
    return coords, _N


# ======================================================================
# Custom autograd Functions with ONNX symbolic for spconv ops
# ======================================================================

class SparseSubMConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indices, spatial_shape, batch_size,
                weight, bias, kernel_size, padding):
        import spconv.pytorch as spconv
        sp_tensor = spconv.SparseConvTensor(features, indices, list(spatial_shape), batch_size)
        conv = spconv.SubMConv3d(
            features.shape[1], weight.shape[0],
            kernel_size=list(kernel_size), padding=list(padding),
            bias=bias is not None, indice_key='subm_onnx_export'
        ).to(features.device)
        conv.weight.data.copy_(weight)
        if bias is not None:
            conv.bias.data.copy_(bias)
        out = conv(sp_tensor)
        return out.features, out.indices

    @staticmethod
    def symbolic(g, features, indices, spatial_shape, batch_size,
                 weight, bias, kernel_size, padding):
        return g.op('spconv::SubMConv3d',
                    features, indices, weight, bias,
                    spatial_shape_i=list(spatial_shape),
                    batch_size_i=batch_size,
                    kernel_size_i=list(kernel_size),
                    padding_i=list(padding))


class SparseConv3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indices, spatial_shape, batch_size,
                weight, bias, kernel_size, stride, padding):
        import spconv.pytorch as spconv
        sp_tensor = spconv.SparseConvTensor(features, indices, list(spatial_shape), batch_size)
        conv = spconv.SparseConv3d(
            features.shape[1], weight.shape[0],
            kernel_size=list(kernel_size), stride=list(stride), padding=list(padding),
            bias=bias is not None, indice_key='spconv_onnx_export'
        ).to(features.device)
        conv.weight.data.copy_(weight)
        if bias is not None:
            conv.bias.data.copy_(bias)
        out = conv(sp_tensor)
        return out.features, out.indices

    @staticmethod
    def symbolic(g, features, indices, spatial_shape, batch_size,
                 weight, bias, kernel_size, stride, padding):
        return g.op('spconv::SparseConv3d',
                    features, indices, weight, bias,
                    spatial_shape_i=list(spatial_shape),
                    batch_size_i=batch_size,
                    kernel_size_i=list(kernel_size),
                    stride_i=list(stride),
                    padding_i=list(padding))


class SparseToDenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indices, spatial_shape, batch_size):
        import spconv.pytorch as spconv
        sp_tensor = spconv.SparseConvTensor(features, indices, list(spatial_shape), batch_size)
        return sp_tensor.dense()

    @staticmethod
    def symbolic(g, features, indices, spatial_shape, batch_size):
        return g.op('spconv::SparseToDense',
                    features, indices,
                    spatial_shape_i=list(spatial_shape),
                    batch_size_i=batch_size)


# ======================================================================
# ONNX-friendly VoxelBackBone8x
# ======================================================================

class ONNXSparseConvLayer(nn.Module):
    """Single sparse conv layer for ONNX export."""
    def __init__(self, conv_layer):
        super().__init__()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.is_subm = 'subm' in (getattr(conv_layer, 'indice_key', '') or '')
        self.kernel_size = conv_layer.kernel_size
        self.stride = getattr(conv_layer, 'stride', [1, 1, 1])
        self.padding = getattr(conv_layer, 'padding', [0, 0, 0])

        self.register_buffer('conv_weight', conv_layer.weight.data.clone())
        if getattr(conv_layer, 'bias', None) is not None:
            self.register_buffer('conv_bias', conv_layer.bias.data.clone())
        else:
            self.conv_bias = None

    def forward(self, features, indices, spatial_shape, batch_size):
        if self.is_subm:
            return SparseSubMConvFunction.apply(
                features, indices, spatial_shape, batch_size,
                self.conv_weight, self.conv_bias,
                self.kernel_size, self.padding)
        else:
            return SparseConv3DFunction.apply(
                features, indices, spatial_shape, batch_size,
                self.conv_weight, self.conv_bias,
                self.kernel_size, self.stride, self.padding)


class ONNXBatchNorm1d(nn.Module):
    def __init__(self, bn_layer):
        super().__init__()
        self.register_buffer('bn_weight', bn_layer.weight.data.clone())
        self.register_buffer('bn_bias', bn_layer.bias.data.clone())
        self.register_buffer('running_mean', bn_layer.running_mean.data.clone())
        self.register_buffer('running_var', bn_layer.running_var.data.clone())
        self.eps = bn_layer.eps

    def forward(self, features, indices):
        # (features - mean) / sqrt(var + eps) * weight + bias
        return ((features - self.running_mean) /
                torch.sqrt(self.running_var + self.eps)) * self.bn_weight + self.bn_bias, indices


class ONNXBlock(nn.Module):
    """A block of sparse conv → BN → ReLU."""
    def __init__(self, spconv_seq_module):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.conv_flags = []  # 'conv', 'bn', 'relu'
        self._unpack(spconv_seq_module)

    def _unpack(self, module):
        for child in module:
            if type(child).__name__ in ('SubMConv3d', 'SparseConv3d'):
                self.convs.append(ONNXSparseConvLayer(child))
                self.conv_flags.append('conv')
            elif isinstance(child, nn.BatchNorm1d):
                self.bns.append(ONNXBatchNorm1d(child))
                self.conv_flags.append('bn')
            elif isinstance(child, nn.ReLU):
                self.relus.append(nn.ReLU(inplace=False))
                self.conv_flags.append('relu')
            elif isinstance(child, nn.Sequential):
                self._unpack(child)

    def forward(self, features, indices, spatial_shape, batch_size):
        ci = bi = ri = 0
        for flag in self.conv_flags:
            if flag == 'conv':
                features, indices = self.convs[ci](features, indices, spatial_shape, batch_size)
                ci += 1
            elif flag == 'bn':
                features, indices = self.bns[bi](features, indices)
                bi += 1
            elif flag == 'relu':
                features = self.relus[ri](features)
                ri += 1
        return features, indices


class ONNXVoxelBackBone8x(nn.Module):
    """ONNX-friendly VoxelBackBone8x."""
    def __init__(self, original_backbone):
        super().__init__()
        self.sparse_shape = original_backbone.sparse_shape[:3]

        PAD1 = [1, 1, 1]
        PAD0 = [0, 0, 0]

        self.conv_input = ONNXBlock(original_backbone.conv_input)
        self.conv1 = ONNXBlock(original_backbone.conv1)
        self.conv2 = ONNXBlock(original_backbone.conv2)
        self.conv3 = ONNXBlock(original_backbone.conv3)
        self.conv4 = ONNXBlock(original_backbone.conv4)
        self.conv_out = ONNXBlock(original_backbone.conv_out)

    def forward(self, features, indices, batch_size=1):
        out_spatial = list(self.sparse_shape)

        # conv_input: SubM, same shape
        features, indices = self.conv_input(features, indices, out_spatial, batch_size)
        # conv1: SubM, same shape
        features, indices = self.conv1(features, indices, out_spatial, batch_size)

        # conv2: SparseConv stride 2 → downsample spatial
        f2, i2 = self.conv2(features, indices, out_spatial, batch_size)
        features, indices = f2, i2

        # conv3: SparseConv stride 2
        f3, i3 = self.conv3(features, indices, out_spatial, batch_size)
        features, indices = f3, i3

        # conv4: SparseConv stride 2
        f4, i4 = self.conv4(features, indices, out_spatial, batch_size)
        features, indices = f4, i4

        # conv_out: SparseConv stride (2,1,1) kernel (3,1,1) padding 0
        f_out, i_out = self.conv_out(features, indices, out_spatial, batch_size)
        return f_out, i_out


# ======================================================================
# Wrapped model
# ======================================================================

class SecondONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.onnx_backbone = ONNXVoxelBackBone8x(model.backbone_3d)
        self.backbone_2d = model.backbone_2d
        self.dense_head = model.dense_head

    def forward(self, voxels, num_points, coords):
        batch_size = 1
        # MeanVFE
        voxel_features = voxels[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(num_points.view(-1, 1).float(), min=1.0)
        voxel_features = voxel_features / normalizer

        # 3D sparse backbone
        features, indices = self.onnx_backbone(voxel_features, coords, batch_size)

        # SparseToDense
        if indices.numel() > 0:
            d = indices[:, 1].max().item() + 1
            h = indices[:, 2].max().item() + 1
            w = indices[:, 3].max().item() + 1
        else:
            d, h, w = 2, 200, 176

        out_spatial = (int(d), int(h), int(w))
        dense = SparseToDenseFunction.apply(features, indices, out_spatial, batch_size)

        # HeightCompression
        N, C, D, H, W = dense.shape
        spatial_features = dense.view(N, C * D, H, W)

        # 2D backbone
        batch_dict = {'spatial_features': spatial_features}
        batch_dict = self.backbone_2d(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']

        # Detection head
        cls_preds = self.dense_head.conv_cls(spatial_features_2d)
        box_preds = self.dense_head.conv_box(spatial_features_2d)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        if self.dense_head.conv_dir_cls is not None:
            dir_cls_preds = self.dense_head.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = torch.zeros(1, device=cls_preds.device)

        return cls_preds, box_preds, dir_cls_preds


# ======================================================================
# Export
# ======================================================================

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
    output_path = '/home/sevnce/lj/project/lidar_detection_pipeline/model/second_raw.onnx'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')

    model = build_model(config_path, ckpt_path, device)
    print(f'[INFO] Model loaded. grid_size: {model.dataset.grid_size}')

    wrapper = SecondONNXWrapper(model).to(device).eval()
    print(f'[INFO] Wrapper created')

    # Create valid test input
    N = 400
    max_points = 5
    C = 4
    spatial_shape = wrapper.onnx_backbone.sparse_shape  # (Z, Y, X)

    dummy_voxels = torch.randn(N, max_points, C, device=device).float()
    dummy_num_points = torch.randint(1, max_points + 1, (N,), device=device).int()
    dummy_coords, actual_N = generate_valid_coords(N, spatial_shape, device)

    # Trim tensors to match actual number of unique coords
    dummy_voxels = dummy_voxels[:actual_N]
    dummy_num_points = dummy_num_points[:actual_N]

    print(f'[INFO] Test input: {actual_N} voxels, features={C}, max_points={max_points}')
    print(f'[INFO] Coords range: z=[{dummy_coords[:,1].min()},{dummy_coords[:,1].max()}], '
          f'y=[{dummy_coords[:,2].min()},{dummy_coords[:,2].max()}], '
          f'x=[{dummy_coords[:,3].min()},{dummy_coords[:,3].max()}]')

    # Test forward
    with torch.no_grad():
        try:
            out = wrapper(dummy_voxels, dummy_num_points, dummy_coords)
            print(f'[INFO] Test forward OK: cls={out[0].shape}, box={out[1].shape}, dir={out[2].shape}')
        except Exception as e:
            print(f'[ERROR] Test forward failed: {e}')
            import traceback
            traceback.print_exc()
            return

    # Export
    print(f'[INFO] Exporting ONNX to {output_path} ...')
    torch.onnx.export(
        wrapper,
        (dummy_voxels, dummy_num_points, dummy_coords),
        output_path,
        input_names=['voxels', 'num_points', 'coords'],
        output_names=['cls', 'box', 'dir'],
        dynamic_axes={
            'voxels': {0: 'num_voxels'},
            'num_points': {0: 'num_voxels'},
            'coords': {0: 'num_voxels'},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f'[INFO] ONNX exported: {output_path}')

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    custom_ops = set()
    for node in onnx_model.graph.node:
        if node.domain:
            custom_ops.add(f'{node.domain}::{node.op_type}')
    print(f'[INFO] ONNX verified: {len(onnx_model.graph.node)} nodes')
    if custom_ops:
        print(f'[INFO] Custom ops: {custom_ops}')
    print('[INFO] Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    warnings.filterwarnings('ignore')
    export()
