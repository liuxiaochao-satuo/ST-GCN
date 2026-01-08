#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从裁剪后的双杠子动作视频中提取关键点

功能：
1. 批量处理裁剪后的双杠子动作视频
2. 使用21点姿态估计模型提取关键点
3. 自动转换为ST-GCN所需的JSON格式
4. 支持根据文件名或目录结构自动识别动作标签

使用方法：
    # 单个视频
    python extract_keypoints_from_video.py --video_path <视频路径> --output_dir <输出目录> --action_label <动作标签>
    
    # 批量处理目录中的所有视频
    python extract_keypoints_from_video.py --video_dir <视频目录> --output_dir <输出目录> --auto_label
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 添加必要的路径
brain_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../brain'))
if brain_path not in sys.path:
    sys.path.insert(0, brain_path)

# 尝试导入MMPose相关模块
try:
    from algorithm.demo_mmpose import PoseDetector
    MMPOSE_AVAILABLE = True
except ImportError:
    print("警告: 无法导入PoseDetector，请确保brain/algorithm/demo_mmpose.py可用")
    MMPOSE_AVAILABLE = False

# 动作类别映射
ACTION_CLASSES = {
    'jumping': 0,
    'jump_to_leg_sit': 1,
    'front_swing': 2,
    'back_swing': 3,
    'front_swing_down': 4,
    'back_swing_down': 5,
}

# 动作标签的中文映射（用于从文件名自动识别）
ACTION_NAME_MAPPING = {
    'jumping': ['jumping', '起跳', 'jump'],
    'jump_to_leg_sit': ['jump_to_leg_sit', '跳上成支撑', 'jump_to', 'leg_sit', '支撑'],
    'front_swing': ['front_swing', '前摆', '前摆上', 'front'],
    'back_swing': ['back_swing', '后摆', '后摆上', 'back'],
    'front_swing_down': ['front_swing_down', '前摆下', 'front_down', '前下'],
    'back_swing_down': ['back_swing_down', '后摆下', 'back_down', '后下'],
}


def auto_detect_action_label(video_path: str) -> Optional[str]:
    """
    从视频文件名或路径自动识别动作标签
    
    Args:
        video_path: 视频文件路径
    
    Returns:
        动作标签名称，如果无法识别则返回None
    """
    video_name = Path(video_path).stem.lower()
    video_dir = Path(video_path).parent.name.lower()
    
    # 检查文件名和目录名
    for action_label, keywords in ACTION_NAME_MAPPING.items():
        for keyword in keywords:
            if keyword.lower() in video_name or keyword.lower() in video_dir:
                return action_label
    
    return None


def extract_keypoints_from_video(
    video_path: str,
    pose_detector: PoseDetector,
    action_label: str,
    output_dir: str,
    save_annotated_video: bool = False,
    device: str = 'cuda:0'
) -> bool:
    """
    从视频中提取关键点并保存为ST-GCN格式
    
    Args:
        video_path: 视频文件路径
        pose_detector: 姿态检测器实例
        action_label: 动作标签
        output_dir: 输出目录
        save_annotated_video: 是否保存标注后的视频
        device: 使用的设备（cuda:0或cpu）
    
    Returns:
        是否成功处理
    """
    if action_label not in ACTION_CLASSES:
        print(f"错误: 未知的动作标签 '{action_label}'")
        print(f"支持的动作标签: {list(ACTION_CLASSES.keys())}")
        return False
    
    label_index = ACTION_CLASSES[action_label]
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n处理视频: {Path(video_path).name}")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  动作标签: {action_label} ({label_index})")
    
    # 存储关键点序列
    sequence_info = []
    frame_count = 0
    
    # 处理视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 如果视频宽度太大，裁剪一半
        if width > 2000:
            width_n = width // 2
            frame = frame[:, :width_n, :]
        else:
            width_n = width
        
        # 使用姿态检测器提取关键点
        # 注意：这里需要根据实际的PoseDetector接口进行调整
        # 假设process_video方法会返回关键点数据
        
        # 由于demo_mmpose.py的接口可能不同，我们需要直接使用模型
        # 这里提供一个通用的接口，需要根据实际情况调整
        try:
            # 尝试使用MMPose直接处理单帧
            # 这里需要根据实际的模型接口进行调整
            keypoint_data = extract_keypoint_from_frame(frame, pose_detector, width_n, height)
            
            if keypoint_data:
                frame_data = {
                    'frame_index': frame_count,
                    'skeleton': [keypoint_data]
                }
                sequence_info.append(frame_data)
        
        except Exception as e:
            print(f"警告: 处理第 {frame_count} 帧时出错: {e}")
            # 添加空骨架（如果没有检测到关键点）
            frame_data = {
                'frame_index': frame_count,
                'skeleton': []
            }
            sequence_info.append(frame_data)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count}/{total_frames} 帧...", end='\r')
    
    cap.release()
    print(f"\n  处理完成，共提取 {frame_count} 帧")
    
    # 检查是否有有效数据
    valid_frames = sum(1 for f in sequence_info if len(f.get('skeleton', [])) > 0)
    if valid_frames == 0:
        print(f"警告: 视频 {video_path} 中没有检测到有效的关键点数据")
        return False
    
    print(f"  有效帧数: {valid_frames}/{frame_count}")
    
    # 创建视频信息字典
    video_info = {
        'data': sequence_info,
        'label': action_label,
        'label_index': label_index,
        'has_skeleton': valid_frames > 0,
        'total_frames': frame_count,
        'valid_frames': valid_frames,
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps
        }
    }
    
    # 保存JSON文件
    video_name = Path(video_path).stem
    output_filename = f"{video_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(video_info, f, indent=2, ensure_ascii=False)
    
    print(f"  已保存: {output_path}")
    
    return True


def extract_keypoint_from_frame(
    frame: np.ndarray,
    pose_detector: PoseDetector,
    frame_width: int,
    frame_height: int
) -> Optional[Dict]:
    """
    从单帧图像提取关键点（需要根据实际的PoseDetector接口调整）
    
    这是一个占位函数，需要根据实际的demo_mmpose.py接口进行调整
    """
    # 这里需要根据实际的PoseDetector接口进行实现
    # 由于demo_mmpose.py的接口可能不同，这里提供一个框架
    
    # 方法1: 如果PoseDetector有process_frame方法
    # try:
    #     result = pose_detector.process_frame(frame)
    #     if result and 'keypoints' in result:
    #         return convert_to_stgcn_format(result, frame_width, frame_height)
    # except:
    #     pass
    
    # 方法2: 使用demo_mmpose.py中已有的方法
    # 需要查看demo_mmpose.py的实际接口
    
    return None  # 占位符，需要实际实现


def convert_to_stgcn_format(
    keypoint_result: Dict,
    frame_width: int,
    frame_height: int,
    num_keypoints: int = 21
) -> Dict:
    """
    将关键点检测结果转换为ST-GCN格式
    
    Args:
        keypoint_result: 关键点检测结果
        frame_width: 图像宽度
        frame_height: 图像高度
        num_keypoints: 关键点数量（默认21）
    
    Returns:
        ST-GCN格式的骨架数据
    """
    if 'keypoints' not in keypoint_result:
        return None
    
    keypoints = np.array(keypoint_result['keypoints']).reshape(-1, 2)
    scores = np.array(keypoint_result.get('keypoint_scores', np.ones(len(keypoints))))
    
    # 确保有正确数量的关键点
    if len(keypoints) != num_keypoints:
        return None
    
    # 归一化坐标到[0, 1]范围
    coordinates = []
    score_list = []
    
    for i in range(num_keypoints):
        x = float(keypoints[i, 0]) / frame_width
        y = float(keypoints[i, 1]) / frame_height
        coordinates.extend([x, y])
        score_list.append(float(scores[i]))
    
    skeleton = {
        'pose': coordinates,
        'score': score_list
    }
    
    return skeleton


def batch_process_videos(
    video_dir: str,
    output_dir: str,
    pose_detector: Optional[PoseDetector] = None,
    auto_label: bool = True,
    device: str = 'cuda:0'
):
    """
    批量处理目录中的所有视频
    
    Args:
        video_dir: 视频目录路径
        output_dir: 输出目录
        pose_detector: 姿态检测器实例（如果为None则创建新的）
        auto_label: 是否自动识别动作标签
        device: 使用的设备
    """
    if not os.path.exists(video_dir):
        print(f"错误: 视频目录不存在: {video_dir}")
        return
    
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # 获取所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
        video_files.extend(Path(video_dir).glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"警告: 在目录 {video_dir} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 初始化姿态检测器
    if pose_detector is None and MMPOSE_AVAILABLE:
        print("初始化姿态检测器...")
        pose_detector = PoseDetector(debug=False)
        print("姿态检测器初始化完成")
    
    # 处理统计
    success_count = 0
    failed_count = 0
    
    # 处理每个视频
    for i, video_path in enumerate(sorted(video_files), 1):
        print(f"\n[{i}/{len(video_files)}] 处理视频: {video_path.name}")
        
        # 自动识别动作标签
        action_label = None
        if auto_label:
            action_label = auto_detect_action_label(str(video_path))
            if action_label:
                print(f"  自动识别动作标签: {action_label}")
            else:
                print(f"  警告: 无法自动识别动作标签，跳过此视频")
                print(f"  提示: 在文件名或目录名中包含以下关键词: {', '.join(['jumping', 'front_swing', 'back_swing', '前摆', '后摆'])}")
                failed_count += 1
                continue
        else:
            # 如果没有启用自动标签，需要手动指定
            # 这里可以从目录结构或其他方式获取
            print("  错误: 未启用自动标签识别，且未指定动作标签")
            failed_count += 1
            continue
        
        # 提取关键点
        if MMPOSE_AVAILABLE and pose_detector:
            success = extract_keypoints_from_video(
                str(video_path),
                pose_detector,
                action_label,
                output_dir,
                device=device
            )
            
            if success:
                success_count += 1
            else:
                failed_count += 1
        else:
            print("  错误: 姿态检测器不可用")
            failed_count += 1
    
    # 输出统计信息
    print(f"\n处理完成:")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"  总计: {len(video_files)}")
    print(f"\n输出目录: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从双杠子动作视频中提取关键点')
    parser.add_argument('--video_path', type=str, default=None,
                        help='单个视频文件路径')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='视频目录路径（批量处理）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--action_label', type=str, default=None,
                        choices=list(ACTION_CLASSES.keys()) + [None],
                        help='动作标签（如果指定video_dir且auto_label=True则忽略）')
    parser.add_argument('--auto_label', action='store_true',
                        help='自动从文件名识别动作标签（仅当使用video_dir时）')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的设备（cuda:0或cpu）')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.video_path is None and args.video_dir is None:
        print("错误: 必须指定 --video_path 或 --video_dir")
        parser.print_help()
        sys.exit(1)
    
    if args.video_path and args.video_dir:
        print("错误: 不能同时指定 --video_path 和 --video_dir")
        parser.print_help()
        sys.exit(1)
    
    # 单个视频处理
    if args.video_path:
        if args.action_label is None:
            # 尝试自动识别
            action_label = auto_detect_action_label(args.video_path)
            if action_label:
                print(f"自动识别动作标签: {action_label}")
            else:
                print("错误: 无法自动识别动作标签，请使用 --action_label 指定")
                sys.exit(1)
        else:
            action_label = args.action_label
        
        if MMPOSE_AVAILABLE:
            pose_detector = PoseDetector(debug=False)
            extract_keypoints_from_video(
                args.video_path,
                pose_detector,
                action_label,
                args.output_dir,
                device=args.device
            )
        else:
            print("错误: MMPose不可用，无法提取关键点")
            sys.exit(1)
    
    # 批量处理
    elif args.video_dir:
        batch_process_videos(
            args.video_dir,
            args.output_dir,
            auto_label=args.auto_label,
            device=args.device
        )

