#!/usr/bin/env python3
"""
读取 KITTI .bin 点云，模拟成 VLP-16 UDP 包发送到本地管道。
用于离线测试 VLP-16 路径（source → infer → display）。

用法:
  # 单帧
  python tools/send_bin_as_velodyne.py data/kitti_seq00/000000.bin

  # 多帧循环（每帧间隔 0.1s）
  python tools/send_bin_as_velodyne.py data/kitti_seq00/ --loop --fps 10

  # 管道端 (另一终端, 需有 DISPLAY 环境变量)
  ./build/lidar_app config/pipeline_velodyne.json
"""

import argparse
import os
import socket
import struct
import time
import numpy as np

VLP16_PORT = 2368
PACKET_SIZE = 1206
BLOCKS_PER_PACKET = 12
HEADER_LEN = 42
BLOCK_DATA_LEN = 100
FOOTER_LEN = 6

# VLP-16 激光垂直角度 (度)，交错排列
VLP16_ANGLES_DEG = np.array([
    -15, 1, -13, 3, -11, 5, -9, 7,
    -7, 9, -5, 11, -3, 13, -1, 15
], dtype=np.float32)


def make_block(azimuth_raw, laser_meas):
    block = bytearray(BLOCK_DATA_LEN)
    struct.pack_into('<HH', block, 0, 0xEEFF, azimuth_raw & 0xFFFF)
    for i in range(16):
        m = laser_meas[i]
        if m is not None:
            dist, intensity = m
            struct.pack_into('<HB', block, 4 + i * 3, dist & 0xFFFF, intensity & 0xFF)
    return bytes(block)


def points_to_packets(points):
    if len(points) == 0:
        return [], 0

    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    # 方位角 (0~36000)
    az_rad = np.arctan2(y, x)
    az_rad[az_rad < 0] += 2 * np.pi
    az_raw = np.round(np.degrees(az_rad) * 100).astype(np.int32) % 36000

    # 距离
    dist = np.sqrt(x*x + y*y + z*z)
    mask = dist >= 0.01
    if not mask.any():
        return [], 0

    az_raw = az_raw[mask]
    dist = dist[mask]
    elev_deg = np.degrees(np.arcsin(z[mask] / dist))
    intensity = intensity[mask]

    # 全向量化激光ID分配
    diffs = np.abs(elev_deg[:, None] - VLP16_ANGLES_DEG[None, :])
    laser_ids = np.argmin(diffs, axis=1).astype(np.uint8)

    dist_2mm = np.clip(np.round(dist / 0.002), 0, 65535).astype(np.uint16)
    intensity_byte = np.clip(np.round(intensity * 255), 0, 255).astype(np.uint8)

    # 按方位角排序
    order = np.argsort(az_raw)
    az_sorted = az_raw[order]
    lid_sorted = laser_ids[order]
    dist_sorted = dist_2mm[order]
    int_sorted = intensity_byte[order]

    packets = []
    buf = bytearray(b'\x00' * HEADER_LEN)
    block_count = 0
    n = len(az_sorted)
    idx = 0
    AZ_WINDOW = 200  # 2° in VLP-16 units

    while idx < n:
        az_start = az_sorted[idx]
        end = np.searchsorted(az_sorted, az_start + AZ_WINDOW, side='right')

        # 窗口内对所有激光取最近点（完全向量化）
        win_lid = lid_sorted[idx:end]
        win_dist = dist_sorted[idx:end]
        win_int = int_sorted[idx:end]

        comb_order = np.lexsort((win_dist, win_lid))
        sorted_lid = win_lid[comb_order]
        sorted_dist = win_dist[comb_order]
        sorted_int = win_int[comb_order]

        _, first_idx = np.unique(sorted_lid, return_index=True)
        measurements = [None] * 16
        for fi in first_idx:
            lid = int(sorted_lid[fi])
            measurements[lid] = (int(sorted_dist[fi]), int(sorted_int[fi]))

        non_null = sum(m is not None for m in measurements)
        if non_null >= 2:
            buf.extend(make_block(int(az_start), measurements))
            block_count += 1

            if block_count == BLOCKS_PER_PACKET:
                ts = int(time.time() * 1_000_000) & 0xFFFFFFFF
                buf.extend(struct.pack('>I', ts) + b'\x00\x00')
                packets.append(bytes(buf))
                buf = bytearray(b'\x00' * HEADER_LEN)
                block_count = 0

        idx = end

    return packets, n


def send_file(sock, bin_path):
    t0 = time.perf_counter()
    data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    if len(data) == 0:
        print(f"[跳过] 空点云: {bin_path}")
        return

    packets, count = points_to_packets(data)
    if not packets:
        print(f"[跳过] 无有效点: {bin_path}")
        return

    for pkt in packets:
        try:
            sock.sendto(pkt, ('127.0.0.1', VLP16_PORT))
        except OSError as e:
            print(f"[错误] 发送失败: {e}")
            return

    elapsed = time.perf_counter() - t0
    print(f"[{elapsed:.2f}s] {os.path.basename(bin_path)}  {count} 点 → {len(packets)} 包")


def main():
    parser = argparse.ArgumentParser(description='将 KITTI .bin 点云模拟为 VLP-16 UDP 包')
    parser.add_argument('input', help='.bin 文件 或 包含 .bin 的目录')
    parser.add_argument('--port', type=int, default=VLP16_PORT,
                        help=f'目标端口 (默认 {VLP16_PORT})')
    parser.add_argument('--loop', action='store_true', help='循环模式（目录）')
    parser.add_argument('--fps', type=float, default=10, help='帧率 (默认 10)')
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    input_path = args.input
    if os.path.isfile(input_path):
        send_file(sock, input_path)
    elif os.path.isdir(input_path):
        bins = sorted(
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.endswith('.bin')
        )
        if not bins:
            print(f"[错误] 目录中无 .bin 文件: {input_path}")
            return

        target_interval = 1.0 / args.fps
        print(f"目录: {input_path}  ({len(bins)} 帧), "
              f"循环: {args.loop}, 目标帧率: {args.fps} fps ({target_interval:.3f}s/帧)")

        while True:
            for b in bins:
                t0 = time.perf_counter()
                send_file(sock, b)
                elapsed = time.perf_counter() - t0
                remaining = target_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    print(f"  ⚠ 处理耗时 {elapsed:.2f}s 已超过目标间隔 {target_interval:.2f}s")
            if not args.loop:
                break
            print("--- 循环重启 ---")
    else:
        print(f"[错误] 路径不存在: {input_path}")

    sock.close()


if __name__ == '__main__':
    main()
