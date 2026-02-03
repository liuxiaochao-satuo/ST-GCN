#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从裁剪后的双杠子动作视频中提取关键点

这个工具直接基于demo_mmpose.py的逻辑，但专注于：
1. 从视频中提取21点关键点
2. 保存为ST-GCN所需的JSON格式
3. 不进行动作分析和视频标注

使用方法：
    # 单个视频
    python extract_keypoints_from_clipped_videos.py \
        --video_path <视频路径> \
        --output_dir <输出目录> \
        --action_label <动作标签> \
        [--device cuda:0]
    
    # 批量处理目录中的所有视频（自动识别动作标签）
    python extract_keypoints_from_clipped_videos.py \
        --video_dir <视频目录> \
        --output_dir <输出目录> \
        [--device cuda:0]
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, List

from mmpose.apis import init_model, inference_bottomup
from mmpose.structures import split_instances

# 路径设置：使用 st-gcn 仓库内的 mmpose 模型（不要依赖 brain/algorithm/demo_mmpose.py）
current_dir = os.path.dirname(os.path.abspath(__file__))
mmpose_root = os.path.abspath(os.path.join(current_dir, '../mmpose'))

# 默认模型配置和权重（用户已放在 /home/satuo/code/st-gcn/mmpose）
DEFAULT_CONFIG = os.path.join(
    mmpose_root, 'config/dekr_hrnet-w32_parallel_ablation_combined.py'
)
DEFAULT_CHECKPOINT = os.path.join(
    mmpose_root, 'checkpoints/best_coco_AP_epoch_110.pth'
)

# 动作类别映射
ACTION_CLASSES = {
    'jump_to_leg_sit': 0,
    'front_swing': 1,
    'back_swing': 2,
    'front_swing_down': 3,
    'back_swing_down': 4,
}

# 动作标签的中文映射（用于从文件名自动识别）
ACTION_NAME_MAPPING = {
    'jump_to_leg_sit': ['jump_to_leg_sit', '跳上成分腿坐', 'jump_to', 'leg_sit', '支撑'],
    'front_swing': ['front_swing', '前摆', '前摆', 'front'],
    'back_swing': ['back_swing', '后摆', '后摆', 'back'],
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
    model,
    device: str = 'cuda:0',
    score_thr: float = 0.3
) -> bool:
    """
    从视频中提取关键点并保存为ST-GCN格式
    
    使用已训练好的 MMPose 底部点检测模型（21 点）直接进行关键点提取，
    不依赖 demo_mmpose.PoseDetector 接口。
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
    # 获取数据集元信息（主要用于检查关键点数量）
    meta_info = getattr(model, 'dataset_meta', None)
    if meta_info:
        num_keypoints = len(meta_info.get('keypoint_name2id', {}))
        print(f'  模型已加载，关键点数量: {num_keypoints}')
    
    # 确定实际使用的图像宽度
    if width > 2000:
        width_n = width // 2
    else:
        width_n = width
    
    # 存储关键点序列
    sequence_info = []
    frame_count = 0
    valid_frames = 0
    last_nose_pos = None  # 如需多人跟踪，可使用该变量，这里暂未启用
    
    # 诊断统计信息
    no_detection_count = 0  # 未检测到任何人的帧数
    low_confidence_count = 0  # 置信度低于阈值的帧数
    total_detected_persons = 0  # 总共检测到的人数
    max_avg_score = 0.0  # 最大平均置信度
    min_avg_score = 1.0  # 最小平均置信度（仅统计有效帧）
    
    print("开始提取关键点...")
    
    # 处理视频帧
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        # 裁剪帧（如果宽度太大）
        if width > 2000:
            frame = raw_frame[:, :width_n, :]
        else:
            frame = raw_frame
        
        # 提取关键点（直接使用 MMPose 底部点接口）
        try:
            # MMPose bottom-up 推理：返回 DataSample 列表
            batch_results = inference_bottomup(model, frame)
            if not batch_results:
                frame_keypoints = None
                no_detection_count += 1
            else:
                results = batch_results[0]
                pred_instances = results.pred_instances
                # 将预测结果拆分为每个人一个字典，包含 keypoints 和 keypoint_scores
                instances = split_instances(pred_instances)
                
                total_detected_persons += len(instances)

                # 过滤置信度较低的实例
                filtered_instances = []
                max_inst_score = 0.0
                for inst in instances:
                    if 'keypoint_scores' in inst:
                        avg_score = float(np.mean(inst['keypoint_scores']))
                        max_inst_score = max(max_inst_score, avg_score)
                        if avg_score >= score_thr:
                            filtered_instances.append(inst)

                if not filtered_instances:
                    frame_keypoints = None
                    if len(instances) > 0:
                        # 有检测到人但置信度太低
                        low_confidence_count += 1
                else:
                    # 简化：取置信度最高的一个人作为运动员
                    best_inst = max(
                        filtered_instances,
                        key=lambda x: float(np.mean(x['keypoint_scores']))
                    )
                    avg_score = float(np.mean(best_inst['keypoint_scores']))
                    max_avg_score = max(max_avg_score, avg_score)
                    min_avg_score = min(min_avg_score, avg_score)
                    frame_keypoints = {
                        'keypoints': best_inst['keypoints'],
                        'keypoint_scores': best_inst['keypoint_scores'],
                    }
            
            # 如果成功提取关键点，转换为ST-GCN格式
            if frame_keypoints:
                # frame_keypoints应该是一个字典，包含'keypoints'和'keypoint_scores'
                skeleton = convert_to_stgcn_format(
                    frame_keypoints, width_n, height, meta_info
                )
                
                if skeleton:
                    frame_data = {
                        'frame_index': frame_count,
                        'skeleton': [skeleton]
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
        
        # 改进的进度提示：显示百分比和详细信息
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress_pct = (frame_count / total_frames * 100) if total_frames > 0 else 0
            print(f"  进度: {progress_pct:.1f}% ({frame_count}/{total_frames}) | "
                  f"有效帧: {valid_frames} ({valid_frames/frame_count*100:.1f}%)", end='\r')
    
    cap.release()
    print()  # 换行
    
    # 输出详细统计信息
    print(f"  处理完成，共提取 {frame_count} 帧 (有效: {valid_frames})")
    
    if valid_frames == 0:
        print(f"\n⚠️  警告: 视频 {Path(video_path).name} 中没有检测到有效的关键点数据")
        print(f"\n诊断信息:")
        print(f"  - 未检测到人体的帧数: {no_detection_count}/{frame_count} ({no_detection_count/frame_count*100:.1f}%)")
        print(f"  - 置信度低于阈值({score_thr})的帧数: {low_confidence_count}/{frame_count}")
        print(f"  - 总共检测到的人数: {total_detected_persons}")
        if total_detected_persons > 0:
            print(f"  - 平均每帧检测到: {total_detected_persons/frame_count:.2f} 人")
        print(f"\n可能的原因:")
        print(f"  1. 背景复杂或人体被遮挡")
        print(f"  2. 人体姿态不在模型训练范围内")
        print(f"  3. 视频质量较差（模糊、光照不足等）")
        print(f"  4. 置信度阈值({score_thr})设置过高，可尝试降低阈值")
        print(f"  5. 视频中人体过小或过大")
        return False
    else:
        # 输出成功统计
        print(f"  ✓ 有效帧率: {valid_frames/frame_count*100:.1f}%")
        if valid_frames > 0:
            print(f"  ✓ 平均置信度范围: {min_avg_score:.3f} ~ {max_avg_score:.3f}")
    
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


def convert_to_stgcn_format(
    keypoint_result: Dict,
    frame_width: int,
    frame_height: int,
    meta_info: Optional[Dict] = None
) -> Optional[Dict]:
    """
    将关键点检测结果转换为ST-GCN格式
    
    Args:
        keypoint_result: 关键点检测结果，包含'keypoints'和'keypoint_scores'
        frame_width: 图像宽度
        frame_height: 图像高度
        meta_info: 数据集元信息（可选）
    """
    if 'keypoints' not in keypoint_result:
        return None
    
    keypoints = np.array(keypoint_result['keypoints'])
    if len(keypoints.shape) == 1:
        # 如果是扁平化的，重塑为(N, 2)
        keypoints = keypoints.reshape(-1, 2)
    
    scores = np.array(keypoint_result.get('keypoint_scores', np.ones(len(keypoints))))
    
    # 确保有21个关键点
    num_keypoints = 21
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
    model,
    device: str = 'cuda:0',
    score_thr: float = 0.3
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
    print("=" * 60)
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, video_path in enumerate(sorted(video_files), 1):
        progress_pct = (i / len(video_files) * 100)
        print(f"\n[{i}/{len(video_files)}] ({progress_pct:.1f}%) 处理: {Path(video_path).name}")
        print("-" * 60)
        
        # 自动识别动作标签
        action_label = auto_detect_action_label(str(video_path))
        if not action_label:
            print(f"  ⚠️  无法自动识别动作标签，跳过")
            skipped_count += 1
            failed_count += 1
            continue
        
        # 提取关键点
        success = extract_keypoints_from_video(
            str(video_path),
            output_dir,
            action_label,
            model=model,
            device=device,
            score_thr=score_thr,
        )
        
        if success:
            success_count += 1
            print(f"  ✓ 成功")
        else:
            failed_count += 1
            print(f"  ✗ 失败")
    
    print("\n" + "=" * 60)
    print(f"批量处理完成:")
    print(f"  ✓ 成功: {success_count} ({success_count/len(video_files)*100:.1f}%)")
    print(f"  ✗ 失败: {failed_count} ({failed_count/len(video_files)*100:.1f}%)")
    if skipped_count > 0:
        print(f"  ⚠️  跳过（无法识别标签）: {skipped_count}")
    print(f"  总计: {len(video_files)}")
    print(f"\n输出目录: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='从裁剪后的双杠子动作视频中提取关键点',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个视频
  python extract_keypoints_from_clipped_videos.py \\
      --video_path /path/to/video.mp4 \\
      --output_dir /path/to/output \\
      --action_label front_swing

  # 批量处理（自动识别动作标签）
  python extract_keypoints_from_clipped_videos.py \\
      --video_dir /path/to/videos \\
      --output_dir /path/to/output
        """
    )
    
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
    parser.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG,
        help=f'MMPose 模型配置文件路径（默认: {DEFAULT_CONFIG}）')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f'MMPose 模型权重文件路径（默认: {DEFAULT_CHECKPOINT}）')
    
    args = parser.parse_args()
    
    if args.video_path is None and args.video_dir is None:
        print("错误: 必须指定 --video_path 或 --video_dir")
        parser.print_help()
        sys.exit(1)
    
    if args.video_path and args.video_dir:
        print("错误: 不能同时指定 --video_path 和 --video_dir")
        parser.print_help()
        sys.exit(1)
    
    # 初始化 MMPose 模型（无论是单个视频还是批量处理，都只初始化一次）
    if not os.path.exists(args.config):
        print(f"错误: 模型配置文件不存在: {args.config}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型权重文件不存在: {args.checkpoint}")
        sys.exit(1)

    print(f"加载模型配置: {args.config}")
    print(f"加载模型权重: {args.checkpoint}")
    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=None,
    )

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
            model=model,
            device=args.device
        )
    
    # 批量处理
    elif args.video_dir:
        batch_process_videos(
            args.video_dir,
            args.output_dir,
            model=model,
            device=args.device
        )

