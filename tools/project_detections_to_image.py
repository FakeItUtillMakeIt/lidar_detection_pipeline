#!/usr/bin/env python3
"""将3D检测框投影到相机图像上"""

import os
import sys
import glob
import numpy as np
import cv2


def parse_calib_velo_to_cam(path):
    """解析 calib_velo_to_cam.txt，返回 3x4 矩阵 [R|t]"""
    R, T = np.eye(3), np.zeros(3)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == 'R:':
                vals = list(map(float, parts[1:]))
                R = np.array(vals).reshape(3, 3)
            elif parts[0] == 'T:':
                T = np.array(list(map(float, parts[1:])))
    return np.hstack([R, T.reshape(3, 1)])


def parse_calib_cam_to_cam(path, cam_id='02'):
    """解析 calib_cam_to_cam.txt，返回 P_rect_xx (3x4) 和 R_rect_xx (3x3)"""
    P_rect = np.eye(3, 4)
    R_rect = np.eye(3)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            key = parts[0]
            if key == f'P_rect_{cam_id}:':
                P_rect = np.array(list(map(float, parts[1:]))).reshape(3, 4)
            elif key == f'R_rect_{cam_id}:':
                R_rect = np.array(list(map(float, parts[1:]))).reshape(3, 3)
    return P_rect, R_rect


def parse_detection_file(path):
    """解析检测结果文件，返回列表 [x,y,z,w,l,h,yaw,class_id,score,...]"""
    detections = []
    with open(path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 10:
                continue
            x, y, z, w, l, h, yaw = map(float, vals[:7])
            class_id = int(vals[7])
            score = float(vals[8])
            detections.append((x, y, z, w, l, h, yaw, class_id, score))
    return detections


def box_3d_corners(x, y, z, w, l, h, yaw):
    """计算3D框的8个角点 (激光雷达坐标系)"""
    # w=沿x(前), l=沿y(左), h=沿z(上)
    # 局部坐标: 中心在原点
    corners = np.array([
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
    ], dtype=np.float64)

    # 绕z轴旋转 (yaw)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [ cos_y, -sin_y, 0],
        [ sin_y,  cos_y, 0],
        [      0,       0, 1],
    ])
    corners = (R @ corners.T).T

    # 平移到中心
    corners += np.array([x, y, z])
    return corners


def project_lidar_to_image(points_3d, velo_to_cam, R_rect, P_rect):
    """
    将 N 个激光雷达3D点投影到图像像素坐标
    points_3d: (N, 3)
    返回 (N, 2) 像素坐标, 以及有效深度
    """
    # (3, 4) * (4, N) -> (3, N)
    pts_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])  # (N, 4)
    cam_pts = (velo_to_cam @ pts_homo.T).T  # (N, 3)

    # R_rect 矫正
    rect_pts = (R_rect @ cam_pts.T).T  # (N, 3)

    # P_rect 投影
    pts_homo2 = np.hstack([rect_pts, np.ones((len(rect_pts), 1))])  # (N, 4)
    proj = (P_rect @ pts_homo2.T).T  # (N, 3)

    depth = proj[:, 2]
    pixel = proj[:, :2] / depth[:, None]
    return pixel, depth


def get_class_name(class_id):
    names = {0: 'Car', 1: 'Ped', 2: 'Cyc'}
    return names.get(class_id, f'Cls{class_id}')


def get_class_color(class_id):
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255)}
    return colors.get(class_id, (255, 255, 255))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/sevnce/lj/project/lidar_detection_pipeline/data/test_1')
    parser.add_argument('--det_dir', default='/home/sevnce/lj/project/lidar_detection_pipeline/build/out/inference')
    parser.add_argument('--output_dir', default='/home/sevnce/lj/project/lidar_detection_pipeline/build/out/projection')
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--max_frames', type=int, default=0, help='0=全部')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取标定
    velo_to_cam = parse_calib_velo_to_cam(os.path.join(args.data_dir, 'calib_velo_to_cam.txt'))
    P_rect, R_rect = parse_calib_cam_to_cam(os.path.join(args.data_dir, 'calib_cam_to_cam.txt'), '02')

    # 组合完整投影矩阵: P_rect * R_rect * [R|t]
    # 可以分步投影，也可以合并
    # 完整外参 T_cam = [R|t]，然后 R_rect * T_cam -> 3x4
    # P_rect(3x4) * R_rect(3x3) * T_cam(3x4) 通过齐次坐标
    # 实际上：P_rect @ R_rect @ velo_to_cam 不能直接乘，因为维度不对
    # 正确做法：先 velo->cam，再 R_rect，再 P_rect
    # 但可以预乘: P_rect @ R_rect 是 3x3 * 3x4? 不行
    # 保持分步投影

    # 找到所有检测文件
    det_files = sorted(glob.glob(os.path.join(args.det_dir, '*.txt')))
    det_files = [f for f in det_files if os.path.basename(f).split('.')[0].isdigit()]

    # 找到所有图像文件
    image_dir = os.path.join(args.data_dir, 'image')

    count = 0
    for det_path in det_files:
        frame_id = os.path.basename(det_path).split('.')[0]

        # 对应的图像文件 (10位编号)
        img_name = f'{int(frame_id):010d}.png'
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f'[SKIP] 图像不存在: {img_path}')
            continue

        # 读取检测结果
        detections = parse_detection_file(det_path)

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f'[SKIP] 无法读取图像: {img_path}')
            continue

        valid_count = 0
        for det in detections:
            x, y, z, l, w, h, yaw, class_id, score = det
            if score < args.score_thresh:
                continue

            # 计算3D框的8个角点
            corners = box_3d_corners(x, y, z, w, l, h, yaw)

            # 投影到图像
            pixels, depths = project_lidar_to_image(corners, velo_to_cam, R_rect, P_rect)

            # 检查是否在图像范围内 (粗略)
            img_h, img_w = img.shape[:2]
            if np.all(depths <= 0):
                continue

            # 绘制投影后的2D框 (用凸包或直接连线)
            pts = pixels.astype(np.int32)

            # 过滤有效点
            valid_mask = (depths > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) & \
                         (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)

            if np.sum(valid_mask) < 4:
                continue

            # 连线顺序: 上面4点，下面4点，上下对应点
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 上面
                (4, 5), (5, 6), (6, 7), (7, 4),  # 下面
                (0, 4), (1, 5), (2, 6), (3, 7),  # 上下连线
            ]

            color = get_class_color(class_id)
            label = f'{get_class_name(class_id)} {score:.2f}'

            # 绘制边线
            for i, j in edges:
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[j])
                if valid_mask[i] and valid_mask[j]:
                    cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

            # 绘制底部中心点和标签
            center_2d = pixels[4:8].mean(axis=0)  # 底面中心
            if 0 <= center_2d[0] < img_w and 0 <= center_2d[1] < img_h:
                cx, cy = int(center_2d[0]), int(center_2d[1])
                cv2.circle(img, (cx, cy), 4, color, -1)
                cv2.putText(img, label, (cx - 30, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            valid_count += 1

        # 保存结果
        out_path = os.path.join(args.output_dir, f'proj_{frame_id}.png')
        cv2.imwrite(out_path, img)
        count += 1
        print(f'[{count}] Frame {frame_id}: {valid_count}/{len(detections)} 框已投影 -> {out_path}')

        if args.max_frames > 0 and count >= args.max_frames:
            break

    print(f'\n完成! 共处理 {count} 帧，结果保存在 {args.output_dir}')


if __name__ == '__main__':
    main()
