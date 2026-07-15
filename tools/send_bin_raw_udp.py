#!/usr/bin/env python3
"""
读取 KITTI .bin 点云，通过 UDP 发送原始 XYZI 二进制数据。
采用批量限速避免内核缓冲溢出:
  - 每批 ~100 个包，包间 2ms 延迟 → 接收端有足够时间排空
  - 若 rmem_max 已调大 (sudo sysctl -w net.core.rmem_max=8388608) 则无需限速

用法:
  python tools/send_bin_raw_udp.py data/kitti_seq00/000000.bin
  python tools/send_bin_raw_udp.py data/kitti_seq00/ --loop --fps 10

接收端:
  ./build/lidar_app config/pipeline_raw_udp.json
"""

import argparse
import os
import socket
import struct
import time
import numpy as np

PORT = 2368
MAGIC = 0x41444350  # 'PCDA'
PTS_PER_PKT = 4000           # 4000 点/包 ≈ 64KB
BATCH_SIZE = 3               # 每批 3 包 ≈ 192KB（内核缓冲可容）
BATCH_DELAY = 0.001          # 批间 1ms 供接收端排空


def send_file(sock, bin_path, frame_id, target):
    t0 = time.perf_counter()
    data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    if len(data) == 0:
        print(f"[跳过] 空点云: {bin_path}")
        return

    total = len(data)
    offset = 0
    pkt_count = 0

    while offset < total:
        chunk = data[offset:offset + PTS_PER_PKT]
        n = len(chunk)
        header = struct.pack('<4I', MAGIC, frame_id, total, n)
        sock.sendto(header + chunk.tobytes(), target)
        pkt_count += 1
        offset += n

        # 每批后等接收端排空
        if pkt_count % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)

    elapsed = time.perf_counter() - t0
    print(f"[{elapsed:.2f}s] #{frame_id} {os.path.basename(bin_path)}  "
          f"{total} 点 → {pkt_count} 包 ({total//86} 包)")


def main():
    parser = argparse.ArgumentParser(description='发送原始 XYZI 点云 (UDP 限速)')
    parser.add_argument('input', help='.bin 文件 或 包含 .bin 的目录')
    parser.add_argument('--port', type=int, default=PORT,
                        help=f'目标端口 (默认 {PORT})')
    parser.add_argument('--loop', action='store_true', help='循环模式（目录）')
    parser.add_argument('--fps', type=float, default=10, help='帧率 (默认 10)')
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ('127.0.0.1', args.port)

    input_path = args.input
    if os.path.isfile(input_path):
        send_file(sock, input_path, 0, target)
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
              f"循环: {args.loop}, 目标: {args.fps} fps ({target_interval:.3f}s/帧)")

        frame_id = 0
        while True:
            for b in bins:
                t0 = time.perf_counter()
                send_file(sock, b, frame_id, target)
                frame_id += 1
                elapsed = time.perf_counter() - t0
                remaining = target_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                elif elapsed > target_interval * 1.5:
                    print(f"  ⚠ 处理耗时 {elapsed:.2f}s 超过目标 {target_interval:.2f}s")
            if not args.loop:
                break
            print("--- 循环重启 ---")
    else:
        print(f"[错误] 路径不存在: {input_path}")

    sock.close()


if __name__ == '__main__':
    main()
