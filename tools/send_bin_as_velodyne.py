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
BLOCK_DATA_LEN = 100  # 每个 block 100 字节
FOOTER_LEN = 6        # 4B timestamp + 2B factory

# VLP-16 激光垂直角度 (度)，交错排列
VLP16_ANGLES_DEG = np.array([
    -15, 1, -13, 3, -11, 5, -9, 7,
    -7, 9, -5, 11, -3, 13, -1, 15
], dtype=np.float32)


def find_best_laser(elev_deg):
    return int(np.argmin(np.abs(VLP16_ANGLES_DEG - elev_deg)))


def make_block(azimuth_raw, laser_meas):
    """
    laser_meas: 长度 16 的 list, 每项 (dist_2mm, intensity) 或 None
    返回 100 字节 block (flag + azimuth + 16×measurement)
    """
    block = bytearray(BLOCK_DATA_LEN)
    struct.pack_into('<HH', block, 0, 0xEEFF, azimuth_raw & 0xFFFF)
    for i in range(16):
        m = laser_meas[i]
        if m is not None:
            dist, intensity = m
            struct.pack_into('<HB', block, 4 + i * 3, dist & 0xFFFF, intensity & 0xFF)
    return bytes(block)


def points_to_packets(points):
    """
    将 N×4 点云转换为 VLP-16 UDP 包列表。
    返回 (packets, point_count)
    """
    if len(points) == 0:
        return [], 0

    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    # 方位角 (0~36000)
    az_raw = np.arctan2(y, x)
    az_raw[az_raw < 0] += 2 * np.pi
    az_raw = np.round(np.degrees(az_raw) * 100).astype(np.int32) % 36000

    # 距离和俯仰角
    dist = np.sqrt(x*x + y*y + z*z)
    mask = dist >= 0.01
    if not mask.any():
        return [], 0

    az_raw = az_raw[mask]
    dist = dist[mask]
    intensity = intensity[mask]
    x, y, z = x[mask], y[mask], z[mask]

    elev_deg = np.degrees(np.arcsin(z / dist))
    laser_ids = np.array([find_best_laser(e) for e in elev_deg])
    dist_2mm = np.round(dist / 0.002).astype(np.int32)
    dist_2mm = np.clip(dist_2mm, 0, 65535).astype(np.uint16)
    intensity_byte = np.clip(np.round(intensity * 255), 0, 255).astype(np.uint8)

    # 按方位角排序
    order = np.argsort(az_raw)
    az_raw, laser_ids = az_raw[order], laser_ids[order]
    dist_2mm, intensity_byte = dist_2mm[order], intensity_byte[order]

    # 每约 2° 为一组 → 一个 block
    AZ_WINDOW = 200  # 2° in VLP-16 units
    idx = 0
    n = len(az_raw)

    packets = []
    buf = bytearray(b'\x00' * HEADER_LEN)
    block_count = 0

    while idx < n:
        az_start = az_raw[idx]
        # 收集窗口内的点
        end = idx
        while end < n and az_raw[end] - az_start < AZ_WINDOW:
            end += 1
        chunk_slice = slice(idx, end)

        # 为 16 个激光各保留最近的一个点
        best = {}
        for j in range(idx, end):
            lid = int(laser_ids[j])
            d = int(dist_2mm[j])
            if lid not in best or d < best[lid][0]:
                best[lid] = (d, int(intensity_byte[j]))

        # 构建 block（至少命中 2 个激光才输出）
        if len(best) >= 2:
            laser_meas = [None] * 16
            for lid, (d, ib) in best.items():
                laser_meas[lid] = (d, ib)

            block_az = int(az_raw[idx])  # 块首方位角
            buf.extend(make_block(block_az, laser_meas))
            block_count += 1

            if block_count == BLOCKS_PER_PACKET:
                ts = int(time.time() * 1_000_000) & 0xFFFFFFFF
                buf.extend(struct.pack('>I', ts))
                buf.extend(b'\x00\x00')
                packets.append(bytes(buf))
                buf = bytearray(b'\x00' * HEADER_LEN)
                block_count = 0

        idx = end  # 跳到下一块

    return packets, n


def send_file(sock, bin_path):
    data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    if len(data) == 0:
        print(f"[跳过] 空点云: {bin_path}")
        return 0

    packets, count = points_to_packets(data)
    if not packets:
        print(f"[跳过] 无有效点: {bin_path}")
        return 0

    for pkt in packets:
        try:
            sock.sendto(pkt, ('127.0.0.1', VLP16_PORT))
        except OSError as e:
            print(f"[错误] 发送失败: {e}")
            return 0

    print(f"[OK] {os.path.basename(bin_path)}  {count} 点 → {len(packets)} 包")
    return len(packets)


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

        
        delay = 1.0 / args.fps
        print(f"目录: {input_path}  ({len(bins)} 帧), 循环: {args.loop}, 帧率: {args.fps} fps, 延迟: {delay:.3f}s)")
        while True:
            for b in bins:
                send_file(sock, b)
                time.sleep(delay)
            if not args.loop:
                break
            print("--- 循环重启 ---")
    else:
        print(f"[错误] 路径不存在: {input_path}")

    sock.close()


if __name__ == '__main__':
    main()
