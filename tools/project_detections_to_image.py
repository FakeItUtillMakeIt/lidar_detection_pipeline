#!/usr/bin/env python3
"""将3D检测框投影到相机图像上 (KITTI标准流程)"""

import os
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
                R = np.array(list(map(float, parts[1:]))).reshape(3, 3)
            elif parts[0] == 'T:':
                T = np.array(list(map(float, parts[1:])))
    return np.hstack([R, T.reshape(3, 1)])


def parse_calib_cam_to_cam(path, cam_id='02'):
    """解析 calib_cam_to_cam.txt，返回 P_rect_xx (3x4) 和 R_rect_00 (3x3)

    KITTI标准:
      pixel = P_rect_02 * R_rect_00 * velo_to_cam * lidar_point
    """
    P_rect = np.eye(3, 4)
    R_rect_00 = np.eye(3)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            key = parts[0]
            if key == f'P_rect_{cam_id}:':
                P_rect = np.array(list(map(float, parts[1:]))).reshape(3, 4)
            elif key == 'R_rect_00:':
                R_rect_00 = np.array(list(map(float, parts[1:]))).reshape(3, 3)
    return P_rect, R_rect_00


def parse_detection_file(path):
    """解析检测结果文件

    文件格式: x y z w l h yaw class_id score track_id vx vy speed heading
      - w = x轴方向尺寸 (LiDAR前方, 车长 ≈ 3.9m for Car)
      - l = y轴方向尺寸 (LiDAR左方, 车宽 ≈ 1.6m for Car)
      - h = z轴方向尺寸 (LiDAR上方, 车高)
    """
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
    """计算3D框的8个角点 (LiDAR坐标系)

    参数:
      w: x轴方向半长 (LiDAR前方, 语义上的"长度/2")
      l: y轴方向半长 (LiDAR左方, 语义上的"宽度/2")
      h: z轴方向高度

    角点顺序 (0-3顶面, 4-7底面):
      0: front-left-top,  1: front-right-top,  2: rear-right-top,  3: rear-left-top
      4: front-left-bot,  5: front-right-bot,  6: rear-right-bot,  7: rear-left-bot
    """
    corners = np.array([
        [ w/2,  l/2,  h/2],
        [ w/2, -l/2,  h/2],
        [-w/2, -l/2,  h/2],
        [-w/2,  l/2,  h/2],
        [ w/2,  l/2, -h/2],
        [ w/2, -l/2, -h/2],
        [-w/2, -l/2, -h/2],
        [-w/2,  l/2, -h/2],
    ], dtype=np.float64)

    # 绕z轴旋转 (yaw), 逆时针为正
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
    """将 N 个激光雷达3D点投影到图像像素坐标

    标准KITTI投影:
      cam = velo_to_cam * lidar           # 3x4 @ (4, N) -> (3, N)
      rect = R_rect_00 * cam              # 3x3 @ (3, N) -> (3, N)
      img = P_rect_02 * [rect; 1]         # 3x4 @ (4, N) -> (3, N)
      u, v = img[0]/img[2], img[1]/img[2]
    """
    pts_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    cam_pts = (velo_to_cam @ pts_homo.T).T
    rect_pts = (R_rect @ cam_pts.T).T
    pts_homo2 = np.hstack([rect_pts, np.ones((len(rect_pts), 1))])
    proj = (P_rect @ pts_homo2.T).T
    depth = proj[:, 2]
    pixel = proj[:, :2] / depth[:, None]
    return pixel, depth


CLASS_NAMES = {0: 'Car', 1: 'Ped', 2: 'Cyc'}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='/home/sevnce/lj/project/lidar_detection_pipeline/data/test_1')
    parser.add_argument('--det_dir',
                        default='/home/sevnce/lj/project/lidar_detection_pipeline/build/out/inference')
    parser.add_argument('--output_dir',
                        default='/home/sevnce/lj/project/lidar_detection_pipeline/build/out/projection')
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--max_frames', type=int, default=0, help='0=全部')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 读取标定
    velo_to_cam = parse_calib_velo_to_cam(
        os.path.join(args.data_dir, 'calib_velo_to_cam.txt'))
    P_rect, R_rect_00 = parse_calib_cam_to_cam(
        os.path.join(args.data_dir, 'calib_cam_to_cam.txt'), '02')

    # 找到所有检测文件
    det_files = sorted(glob.glob(os.path.join(args.det_dir, '*.txt')))
    det_files = [f for f in det_files
                 if os.path.basename(f).split('.')[0].isdigit()]

    image_dir = os.path.join(args.data_dir, 'image')

    count = 0
    for det_path in det_files:
        frame_id = os.path.basename(det_path).split('.')[0]

        img_name = f'{int(frame_id):010d}.png'
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        detections = parse_detection_file(det_path)

        img = cv2.imread(img_path)
        if img is None:
            print(f'[SKIP] 无法读取图像: {img_path}')
            continue

        valid_count = 0
        for det in detections:
            x, y, z, w, l, h, yaw, class_id, score = det
            if score < args.score_thresh:
                continue

            corners = box_3d_corners(x, y, z, w, l, h, yaw)
            pixels, depths = project_lidar_to_image(
                corners, velo_to_cam, R_rect_00, P_rect)

            img_h, img_w = img.shape[:2]
            if np.all(depths <= 0):
                continue

            pts = pixels.astype(np.int32)
            valid_mask = ((depths > 0) &
                          (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) &
                          (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h))

            if np.sum(valid_mask) < 4:
                continue

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]

            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            label = f'{CLASS_NAMES.get(class_id, f"C{class_id}")} {score:.2f}'

            for i, j in edges:
                if valid_mask[i] and valid_mask[j]:
                    cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, 2,
                             cv2.LINE_AA)

            # 底面中心位置
            center_2d = pixels[4:8].mean(axis=0)
            if (0 <= center_2d[0] < img_w and
                    0 <= center_2d[1] < img_h):
                cx, cy = int(center_2d[0]), int(center_2d[1])
                cv2.circle(img, (cx, cy), 4, color, -1)
                cv2.putText(img, label, (cx - 30, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            valid_count += 1

        out_path = os.path.join(args.output_dir, f'proj_{frame_id}.png')
        cv2.imwrite(out_path, img)
        count += 1
        print(f'[{count}] Frame {frame_id}: {valid_count}/{len(detections)} '
              f'框已投影 -> {out_path}')

        if args.max_frames > 0 and count >= args.max_frames:
            break

    print(f'\n完成! 共处理 {count} 帧，结果保存在 {args.output_dir}')


if __name__ == '__main__':
    main()
