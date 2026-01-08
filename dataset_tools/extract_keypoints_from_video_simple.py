#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从裁剪后的双杠子动作视频中提取关键点（简化版本）

基于demo_mmpose.py，专注于关键点提取和保存为ST-GCN格式

使用方法：
    # 单个视频
    python extract_keypoints_from_video_simple.py --video_path <视频路径> --output_dir <输出目录> --action_label <动作标签>
    
    # 批量处理目录中的所有视频（自动识别动作标签）
    python extract_keypoints_from_video_simple.py --video_dir <视频目录> --output_dir <输出目录>
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, List

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
brain_path = os.path.abspath(os.path.join(current_dir, '../../brain'))
if brain_path not in sys.path:
    sys.path.insert(0, brain_path)

# 导入必要的模块
try:
    from algorithm.demo_mmpose import PoseDetector
    from mmpose.apis import init_model
    MMPOSE_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入必要的模块: {e}")
    print("请确保brain/algorithm/demo_mmpose.py可用")
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
    """从视频文件名或路径自动识别动作标签"""
    video_name = Path(video_path).stem.lower()
    video_dir = Path(video_path).parent.name.lower()
    
    for action_label, keywords in ACTION_NAME_MAPPING.items():
        for keyword in keywords:
            if keyword.lower() in video_name or keyword.lower() in video_dir:
                return action_label
    
    return None


def extract_keypoints_from_video(
    video_path: str,
    output_dir: str,
    action_label: str,
    device: str = 'cuda:0',
    model_config: Optional[str] = None,
    model_checkpoint: Optional[str] = None
) -> bool:
    """
    从视频中提取关键点并保存为ST-GCN格式
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        action_label: 动作标签
        device: 使用的设备
        model_config: 模型配置文件路径（可选）
        model_checkpoint: 模型权重文件路径（可选）
    """
    if not MMPOSE_AVAILABLE:
        print("错误: MMPose不可用")
        return False
    
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
    
    # 初始化姿态检测器
    # 使用与demo_mmpose.py相同的方式初始化
    try:
        if model_config is None:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            model_config = os.path.join(
                brain_path, 
                'algorithm/config/dekr_hrnet-w32_parallel_ablation_combined.py'
            )
        if model_checkpoint is None:
            model_checkpoint = os.path.join(
                brain_path,
                'algorithm/checkpoints/best_coco_AP_epoch_110.pth'
            )
        
        # 检查文件是否存在
        if not os.path.exists(model_config):
            print(f"警告: 模型配置文件不存在: {model_config}")
            print("使用默认的PoseDetector初始化方式...")
            pose_detector = PoseDetector(debug=False)
        else:
            # 使用init_model直接初始化（如果可用）
            model = init_model(
                model_config,
                model_checkpoint,
                device=device,
                cfg_options=None
            )
            meta_info = model.dataset_meta
            pose_detector = None  # 使用model而不是PoseDetector
    except Exception as e:
        print(f"使用PoseDetector初始化: {e}")
        pose_detector = PoseDetector(debug=False)
        model = None
        meta_info = None
    
    # 如果使用PoseDetector，可以通过它获取meta_info
    if pose_detector and hasattr(pose_detector, 'model'):
        try:
            meta_info = pose_detector.model.dataset_meta if hasattr(pose_detector.model, 'dataset_meta') else None
        except:
            pass
    
    # 存储关键点序列
    sequence_info = []
    frame_count = 0
    valid_frames = 0
    
    # 确定实际使用的图像宽度
    if width > 2000:
        width_n = width // 2
    else:
        width_n = width
    
    # 处理视频帧
    print("开始提取关键点...")
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        # 裁剪帧（如果宽度太大）
        if width > 2000:
            frame = raw_frame[:, :width_n, :]
        else:
            frame = raw_frame
        
        # 提取关键点
        try:
            if model is not None:
                # 使用model直接处理
                from mmpose.apis import inference_topdown
                from mmengine.structures import InstanceData
                
                # 首先进行人体检测
                from mmdet.apis import init_detector, inference_detector
                
                # 简化：假设使用PoseDetector的方式
                # 这里需要根据实际的接口调整
                keypoint_data = None
            else:
                # 使用PoseDetector
                # 由于PoseDetector的接口可能不直接支持单帧处理
                # 我们需要调用内部方法
                # 注意：这里需要根据实际的demo_mmpose.py进行调整
                keypoint_data = None
            
            # 临时方案：使用PoseDetector的process_video方法的逻辑
            # 但我们需要提取关键点而不保存视频
            # 这里提供一个简化的实现
            
            # 由于demo_mmpose.py的接口限制，我们创建一个简化的处理函数
            frame_keypoints = extract_frame_keypoints_simple(
                frame, model if model else pose_detector, meta_info, width_n, height
            )
            
            if frame_keypoints:
                frame_data = {
                    'frame_index': frame_count,
                    'skeleton': [frame_keypoints]
                }
                sequence_info.append(frame_data)
                valid_frames += 1
            else:
                # 添加空骨架
                frame_data = {
                    'frame_index': frame_count,
                    'skeleton': []
                }
                sequence_info.append(frame_data)
        
        except Exception as e:
            print(f"\n警告: 处理第 {frame_count} 帧时出错: {e}")
            frame_data = {
                'frame_index': frame_count,
                'skeleton': []
            }
            sequence_info.append(frame_data)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count}/{total_frames} 帧 (有效: {valid_frames})...", end='\r')
    
    cap.release()
    print(f"\n  处理完成，共提取 {frame_count} 帧 (有效: {valid_frames})")
    
    if valid_frames == 0:
        print(f"警告: 视频 {video_path} 中没有检测到有效的关键点数据")
        return False
    
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


def extract_frame_keypoints_simple(frame, model_or_detector, meta_info, width, height):
    """
    从单帧提取关键点的简化实现
    
    这是一个占位函数，需要根据实际的模型接口实现
    建议直接使用demo_mmpose.py中的逻辑，但提取关键点而不是保存视频
    """
    # TODO: 根据实际的模型接口实现
    # 这里应该调用模型进行关键点检测，然后转换为ST-GCN格式
    return None


def batch_process_videos(
    video_dir: str,
    output_dir: str,
    device: str = 'cuda:0'
):
    """批量处理目录中的所有视频"""
    if not os.path.exists(video_dir):
        print(f"错误: 视频目录不存在: {video_dir}")
        return
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f'*{ext}'))
        video_files.extend(Path(video_dir).glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"警告: 在目录 {video_dir} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    success_count = 0
    failed_count = 0
    
    for i, video_path in enumerate(sorted(video_files), 1):
        print(f"\n[{i}/{len(video_files)}] 处理: {Path(video_path).name}")
        
        # 自动识别动作标签
        action_label = auto_detect_action_label(str(video_path))
        if not action_label:
            print(f"  无法自动识别动作标签，跳过")
            failed_count += 1
            continue
        
        # 提取关键点
        success = extract_keypoints_from_video(
            str(video_path),
            output_dir,
            action_label,
            device=device
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
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
                        help='动作标签（仅用于单个视频，批量处理时自动识别）')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的设备（cuda:0或cpu）')
    
    args = parser.parse_args()
    
    if args.video_path is None and args.video_dir is None:
        print("错误: 必须指定 --video_path 或 --video_dir")
        parser.print_help()
        sys.exit(1)
    
    if args.video_path and args.video_dir:
        print("错误: 不能同时指定 --video_path 和 --video_dir")
        parser.print_help()
        sys.exit(1)
    
    # 单个视频
    if args.video_path:
        if args.action_label is None:
            action_label = auto_detect_action_label(args.video_path)
            if action_label:
                print(f"自动识别动作标签: {action_label}")
            else:
                print("错误: 无法自动识别动作标签，请使用 --action_label 指定")
                sys.exit(1)
        else:
            action_label = args.action_label
        
        extract_keypoints_from_video(
            args.video_path,
            args.output_dir,
            action_label,
            device=args.device
        )
    
    # 批量处理
    elif args.video_dir:
        batch_process_videos(args.video_dir, args.output_dir, device=args.device)

