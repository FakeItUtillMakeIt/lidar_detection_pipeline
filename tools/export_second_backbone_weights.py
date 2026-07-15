#!/usr/bin/env python3
"""
Export SECOND 3D backbone (VoxelBackBone8x) weights + pre-computed forward maps.

Output: binary weight file with JSON header consumed by C++ second-backbone.cu

For each layer we export:
  - type (SubMConv / SparseConv)
  - weight, bias, BN params
  - gather_map: [num_kernel * num_out] int32 — for each (k, i_out), input index to gather
  - out_coords: [num_out, 4] int32 — output coordinates (identity for SubM)

Usage:
  cd /home/sevnce/lj/project/OpenPCDet/tools && \\
  PYTHONPATH=/usr/lib/python3.10/dist-packages python3 \\
    /path/to/export_second_backbone_weights.py
"""

import os, sys, warnings, logging, struct, json
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
    return model


def build_forward_map(subm, indice_data, num_in, num_out, device):
    """
    Build a uniform forward gather_map from spconv indice data.

    Returns:
        gather_map: int32 [kv, num_out] — gather_indices[k, i] = input index to read
        out_coords: int32 [num_out, 4] — output coordinates
    """
    import spconv.pytorch.core as spcore
    kv = indice_data.ksize[0] * indice_data.ksize[1] * indice_data.ksize[2]
    # kv might be given explicitly; derive from data

    if isinstance(indice_data, spcore.IndiceData):
        ip = indice_data.indice_pairs  # SubM: [2, kv, N], regular: [kv, N_in]
        if subm:
            # ip shape: [2, kv, N]
            kv_actual = ip.shape[1]
            gather_map = torch.full((kv_actual, num_out), -1, dtype=torch.int32, device=device)
            for k in range(kv_actual):
                valid = ip[0, k] >= 0
                out_idx = ip[1, k, valid]  # output indices
                in_idx = ip[0, k, valid]   # input indices
                gather_map[k, out_idx] = in_idx
            out_coords = indice_data.out_indices.int()
        else:
            # ip shape: [kv, N_in] — maps input→output
            kv_actual = ip.shape[0]
            gather_map = torch.full((kv_actual, num_out), -1, dtype=torch.int32, device=device)
            for k in range(kv_actual):
                valid = ip[k] >= 0
                out_idx = ip[k, valid]
                in_idx = torch.where(valid)[0]
                gather_map[k, out_idx] = in_idx
            out_coords = indice_data.out_indices.int()

    elif isinstance(indice_data, spcore.ImplicitGemmIndiceData):
        if subm:
            pf = indice_data.pair_fwd  # [kv, num_in] = [kv, num_out]
            kv_actual = pf.shape[0]
            gather_map = pf.int()  # each entry is input index
            out_coords = indice_data.out_indices.int()
        else:
            pf = indice_data.pair_fwd  # [kv, num_out] — each entry is input index
            kv_actual = pf.shape[0]
            gather_map = pf.int()
            out_coords = indice_data.out_indices.int()

    return gather_map, out_coords


def extract_layer_info(bb3d, grid_size, device='cuda'):
    layers = []
    # Use bb3d.sparse_shape (6-element: [Z, Y, X, 1, 0, 0]) to match model internals
    sparse_shape = list(bb3d.sparse_shape)
    batch_size = 1
    spconv = __import__('spconv.pytorch', fromlist=[''])

    def extract_convs_from_block(seq_module):
        """Recursively find conv layers and their associated BN/ReLU."""
        results = []
        for child in seq_module:
            name = type(child).__name__
            if name in ('SubMConv3d', 'SparseConv3d'):
                # Find BN and ReLU after this conv within the same SparseSequential
                bn = None; relu = False
                child_list = list(seq_module)
                idx = child_list.index(child)
                for off in range(1, len(child_list) - idx):
                    c = child_list[idx + off]
                    if isinstance(c, torch.nn.BatchNorm1d):
                        bn = c
                    elif isinstance(c, torch.nn.ReLU):
                        relu = True
                    elif type(c).__name__ in ('SubMConv3d', 'SparseConv3d'):
                        break
                results.append((child, bn, relu))
            elif name == 'SparseSequential':
                results.extend(extract_convs_from_block(child))
        return results

    block_names = ['conv_input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv_out']
    current_spatial = list(sparse_shape)

    # Generate well-distributed coordinates covering full spatial extent
    N_init = 5000
    zs = torch.randint(0, max(sparse_shape[0], 1), (N_init,), device='cpu')
    ys = torch.randint(0, max(sparse_shape[1], 1), (N_init,), device='cpu')
    xs = torch.randint(0, max(sparse_shape[2], 1), (N_init,), device='cpu')
    dummy_coords = torch.stack([torch.zeros(N_init, dtype=torch.int32),
                                zs.int(), ys.int(), xs.int()], dim=1)
    dummy_coords = torch.unique(dummy_coords, dim=0).to(device)
    actual_n = dummy_coords.shape[0]
    dummy_features = torch.randn(actual_n, 4, device=device)

    sp_tensor = spconv.SparseConvTensor(
        features=dummy_features, indices=dummy_coords,
        spatial_shape=sparse_shape, batch_size=batch_size
    )
    current_tensor = sp_tensor

    for block_name in block_names:
        block = getattr(bb3d, block_name)
        convs = extract_convs_from_block(block)
        for conv, bn, relu in convs:
            is_subm = isinstance(conv, spconv.SubMConv3d)
            if not (is_subm or isinstance(conv, spconv.SparseConv3d)):
                continue

            ksize = list(conv.kernel_size)
            stride = list(conv.stride)
            padding = list(conv.padding)

            with torch.no_grad():
                out_tensor = conv(current_tensor)

            indice_data = None
            if conv.indice_key is not None:
                indice_data = out_tensor.find_indice_pair(conv.indice_key)
            if indice_data is None:
                raise RuntimeError(f'No indice_data for key={conv.indice_key}')

            num_in = current_tensor.features.shape[0]
            num_out = out_tensor.features.shape[0]

            gather_map, out_coords = build_forward_map(
                is_subm, indice_data, num_in, num_out, device)

            kv = gather_map.shape[0]
            layer = {
                'name': block_name,
                'is_subm': is_subm,
                'in_channels': conv.in_channels,
                'out_channels': conv.out_channels,
                'kernel_size': ksize,
                'stride': stride,
                'padding': padding,
                'weight': conv.weight.data.cpu().numpy(),
                # weight_stacked: (kv, C_out, C_in) contiguous, each kernel's weight in row-major
                # spconv uses channels-last: (C_out, kz, ky, kx, C_in)
                # Need to transpose to (kz, ky, kx, C_out, C_in) then flatten
                'weight_stacked': conv.weight.data.cpu().numpy().transpose(1, 2, 3, 0, 4).reshape(-1, conv.out_channels, conv.in_channels),
                'bias': conv.bias.data.cpu().numpy() if conv.bias is not None else None,
                'bn_weight': bn.weight.data.cpu().numpy() if bn else None,
                'bn_bias': bn.bias.data.cpu().numpy() if bn else None,
                'bn_running_mean': bn.running_mean.data.cpu().numpy() if bn else None,
                'bn_running_var': bn.running_var.data.cpu().numpy() if bn else None,
                'bn_eps': bn.eps if bn else 0.0,
                'has_relu': relu,
                'kv': kv,
                'num_out': num_out,
                'gather_map': gather_map.cpu().numpy().astype(np.int32),
                'out_coords': out_coords.cpu().numpy().astype(np.int32),
                'input_spatial_shape': list(current_spatial),
                'output_spatial_shape': list(out_tensor.spatial_shape[:3]) if hasattr(out_tensor, 'spatial_shape') else current_spatial,
            }
            layers.append(layer)
            dsp = 'same' if is_subm else f'{layer["output_spatial_shape"][:3]}'
            print(f'  {block_name} {"SubM" if is_subm else "Sparse"} '
                  f'{conv.in_channels}->{conv.out_channels} '
                  f'k{ksize} s{stride} p{padding} '
                  f'spatial {current_spatial[:3]}->{dsp} '
                  f'N_in={num_in} N_out={num_out} '
                  f'kv={kv} gather={gather_map.shape} '
                  f'BN={bn is not None} ReLU={relu}')

            current_tensor = out_tensor
            current_spatial = layer['output_spatial_shape']

    return layers


def convert_for_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_weights(layers, output_path):
    header = {'num_layers': len(layers), 'layer_info': []}
    offset = 0
    binary_data = b''

    for layer in layers:
        info = {
            'name': layer['name'],
            'is_subm': layer['is_subm'],
            'in_channels': layer['in_channels'],
            'out_channels': layer['out_channels'],
            'kernel_size': layer['kernel_size'],
            'stride': layer['stride'],
            'padding': layer['padding'],
            'has_bias': layer['bias'] is not None,
            'has_bn': layer['bn_weight'] is not None,
            'bn_eps': layer['bn_eps'],
            'has_relu': layer['has_relu'],
            'kv': layer['kv'],
            'num_out': layer['num_out'],
            'input_spatial_shape': layer['input_spatial_shape'],
            'output_spatial_shape': layer['output_spatial_shape'],
            'weight_offset': offset,
            'weight_size': layer['weight'].nbytes,
        }
        binary_data += layer['weight'].tobytes()
        offset += layer['weight'].nbytes

        # weight_stacked: contiguous per-kernel (kv, C_out, C_in)
        ws = layer['weight_stacked']
        info['weight_stacked_offset'] = offset
        info['weight_stacked_shape'] = list(ws.shape)
        info['weight_stacked_size'] = ws.nbytes
        binary_data += ws.tobytes()
        offset += ws.nbytes

        if layer['bias'] is not None:
            info['bias_offset'] = offset
            info['bias_size'] = layer['bias'].nbytes
            binary_data += layer['bias'].tobytes()
            offset += layer['bias'].nbytes

        if layer['bn_weight'] is not None:
            info['bn_weight_offset'] = offset
            info['bn_weight_size'] = layer['bn_weight'].nbytes
            binary_data += layer['bn_weight'].tobytes()
            offset += layer['bn_weight'].nbytes

            info['bn_bias_offset'] = offset
            info['bn_bias_size'] = layer['bn_bias'].nbytes
            binary_data += layer['bn_bias'].tobytes()
            offset += layer['bn_bias'].nbytes

            info['bn_mean_offset'] = offset
            info['bn_mean_size'] = layer['bn_running_mean'].nbytes
            binary_data += layer['bn_running_mean'].tobytes()
            offset += layer['bn_running_mean'].nbytes

            info['bn_var_offset'] = offset
            info['bn_var_size'] = layer['bn_running_var'].nbytes
            binary_data += layer['bn_running_var'].tobytes()
            offset += layer['bn_running_var'].nbytes

        # Compute kernel offsets from kernel_size, padding, stride
        kz, ky, kx = layer['kernel_size']
        pz, py, px = layer['padding'] if 'padding' in layer else [0, 0, 0]
        sz, sy, sx = layer['stride'] if 'stride' in layer else [1, 1, 1]
        offsets = []
        for kzi in range(kz):
            for kyi in range(ky):
                for kxi in range(kx):
                    # Input coord = out_coord * stride - padding + kernel_idx
                    dz = kzi - pz
                    dy = kyi - py
                    dx = kxi - px
                    offsets.append([dz, dy, dx])
        info['kernel_offsets'] = offsets
        info['stride'] = [sz, sy, sx]

        gm = layer['gather_map']
        info['gather_map_offset'] = offset
        info['gather_map_shape'] = list(gm.shape)
        info['gather_map_size'] = gm.nbytes
        binary_data += gm.tobytes()
        offset += gm.nbytes

        oc = layer['out_coords']
        info['out_coords_offset'] = offset
        info['out_coords_shape'] = list(oc.shape)
        info['out_coords_size'] = oc.nbytes
        binary_data += oc.tobytes()
        offset += oc.nbytes

        header['layer_info'].append(info)

    header_json = json.dumps(header, default=convert_for_json)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', len(header_json)))
        f.write(header_json.encode('utf-8'))
        f.write(binary_data)
    print(f'[INFO] Saved {len(layers)} layers to {output_path} ({offset / 1024 / 1024:.1f} MB)')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] Device: {device}')
    config_path = 'cfgs/kitti_models/second.yaml'
    ckpt_path = '/home/sevnce/lj/project/lidar_detection_pipeline/model/SECOND/second_7862.pth'
    output_path = '/home/sevnce/lj/project/lidar_detection_pipeline/model/second_3d_backbone.weights'

    model = build_model(config_path, ckpt_path, device)
    print(f'[INFO] Model loaded')
    bb3d = model.backbone_3d
    grid_size = model.dataset.grid_size
    print(f'[INFO] Grid size: {grid_size}  sparse_shape: {grid_size[::-1]}')

    layers = extract_layer_info(bb3d, grid_size, device)
    print(f'[INFO] Extracted {len(layers)} layers')
    save_weights(layers, output_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    warnings.filterwarnings('ignore')
    main()
