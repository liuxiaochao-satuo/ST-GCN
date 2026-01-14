#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JSON格式的关键点数据转换为ST-GCN模型输入格式

功能：
1. 从JSON文件读取关键点序列
2. 转换为ST-GCN所需的numpy数组格式 (C, T, V, M)
3. 应用与训练时相同的数据预处理

使用方法：
    python convert_json_to_stgcn_input.py \
        --json_path <JSON文件路径> \
        --output_path <输出NPY文件路径> [可选]
    
    或者在Python代码中直接使用：
    from convert_json_to_stgcn_input import json_to_stgcn_input
    data_numpy = json_to_stgcn_input('path/to/keypoints.json')
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# 添加ST-GCN路径
current_dir = os.path.dirname(os.path.abspath(__file__))
st_gcn_path = os.path.abspath(os.path.join(current_dir, '..'))
if st_gcn_path not in sys.path:
    sys.path.insert(0, st_gcn_path)


def json_to_stgcn_input(
    json_path: str,
    max_frame: int = 300,
    num_joints: int = 21,
    num_person_in: int = 5,
    num_person_out: int = 1,
    center: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    将JSON格式的关键点数据转换为ST-GCN输入格式
    
    Args:
        json_path: JSON文件路径
        max_frame: 最大帧数（如果序列更长会被截断，更短会被填充）
        num_joints: 关键点数量（默认21）
        num_person_in: 输入时观察的最大人数
        num_person_out: 输出时选择的人数
        center: 是否进行中心化（减去0.5）
        normalize: 是否已经归一化（JSON中的坐标应该已经是[0,1]范围）
    
    Returns:
        numpy数组，形状为 (C, T, V, M)，其中：
        - C = 3 (x坐标, y坐标, score置信度)
        - T = max_frame (时间帧数)
        - V = num_joints (关节数)
        - M = num_person_out (人数)
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        video_info = json.load(f)
    
    # 初始化数据数组
    data_numpy = np.zeros((3, max_frame, num_joints, num_person_in))
    
    # 填充数据
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        if frame_index >= max_frame:
            continue
        
        skeletons = frame_info.get("skeleton", [])
        for m, skeleton_info in enumerate(skeletons):
            if m >= num_person_in:
                break
            
            pose = skeleton_info.get('pose', [])
            score = skeleton_info.get('score', [])
            
            # 确保有正确数量的关键点
            if len(pose) == num_joints * 2:  # 42个值（x,y坐标交替）
                # 提取x和y坐标
                x_coords = pose[0::2]  # x坐标
                y_coords = pose[1::2]  # y坐标
                
                data_numpy[0, frame_index, :, m] = x_coords
                data_numpy[1, frame_index, :, m] = y_coords
                
                if len(score) == num_joints:
                    data_numpy[2, frame_index, :, m] = score
                else:
                    # 如果没有score，使用默认值1.0
                    data_numpy[2, frame_index, :, m] = 1.0
            elif len(pose) > num_joints * 2:
                # 如果有多余的值，只取前num_joints*2个
                x_coords = pose[0::2][:num_joints]
                y_coords = pose[1::2][:num_joints]
                data_numpy[0, frame_index, :, m] = x_coords
                data_numpy[1, frame_index, :, m] = y_coords
                if len(score) >= num_joints:
                    data_numpy[2, frame_index, :, m] = score[:num_joints]
                else:
                    data_numpy[2, frame_index, :, m] = 1.0
    
    # 数据预处理（与训练时相同）
    if normalize:
        # JSON中的坐标应该已经归一化到[0, 1]，这里只需要确认
        # 如果JSON中的坐标已经是归一化的，这一步不需要做
        pass
    else:
        # 如果JSON中的坐标是像素坐标，需要归一化
        # 但这应该在提取关键点时完成，所以这里通常不需要
        pass
    
    if center:
        # 中心化：坐标减去0.5，得到[-0.5, 0.5]范围
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        
        # 置信度为0的关键点坐标设为0
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0
    
    # 按分数排序，选择置信度最高的人
    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2, 0))
    
    # 只保留前num_person_out个人
    data_numpy = data_numpy[:, :, :, 0:num_person_out]
    
    return data_numpy


def convert_json_file(
    json_path: str,
    output_path: Optional[str] = None,
    max_frame: int = 300,
    num_joints: int = 21
) -> np.ndarray:
    """
    转换JSON文件并保存为NPY格式
    
    Args:
        json_path: 输入的JSON文件路径
        output_path: 输出的NPY文件路径（如果为None，则使用JSON文件名）
        max_frame: 最大帧数
        num_joints: 关键点数量
    
    Returns:
        转换后的numpy数组
    """
    # 转换数据
    data_numpy = json_to_stgcn_input(
        json_path,
        max_frame=max_frame,
        num_joints=num_joints
    )
    
    # 保存NPY文件
    if output_path is None:
        json_file = Path(json_path)
        output_path = json_file.with_suffix('.npy')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, data_numpy)
    print(f"已保存: {output_path}")
    print(f"数据形状: {data_numpy.shape} (C, T, V, M)")
    print(f"  C (通道数): {data_numpy.shape[0]} (x, y, score)")
    print(f"  T (时间帧数): {data_numpy.shape[1]}")
    print(f"  V (关节数): {data_numpy.shape[2]}")
    print(f"  M (人数): {data_numpy.shape[3]}")
    
    return data_numpy


def prepare_for_model_inference(
    data_numpy: np.ndarray,
    device: str = 'cuda:0'
) -> 'torch.Tensor':
    """
    准备数据用于模型推理
    
    Args:
        data_numpy: 形状为 (C, T, V, M) 的numpy数组
        device: 使用的设备
    
    Returns:
        torch.Tensor，形状为 (1, C, T, V, M)，已移动到指定设备
    """
    import torch
    
    # 转换为torch tensor
    data = torch.from_numpy(data_numpy)
    
    # 增加batch维度: (C, T, V, M) -> (1, C, T, V, M)
    data = data.unsqueeze(0)
    
    # 转换为float类型并移动到设备
    data = data.float().to(device)
    
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='将JSON格式的关键点数据转换为ST-GCN模型输入格式'
    )
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='输入的JSON文件路径'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='输出的NPY文件路径（可选，默认使用JSON文件名）'
    )
    parser.add_argument(
        '--max_frame',
        type=int,
        default=300,
        help='最大帧数（默认: 300）'
    )
    parser.add_argument(
        '--num_joints',
        type=int,
        default=21,
        help='关键点数量（默认: 21）'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"错误: JSON文件不存在: {args.json_path}")
        sys.exit(1)
    
    # 转换并保存
    data_numpy = convert_json_file(
        args.json_path,
        args.output_path,
        max_frame=args.max_frame,
        num_joints=args.num_joints
    )
    
    print("\n转换完成！")
    print(f"可以使用以下代码加载并用于模型推理：")
    print(f"  import numpy as np")
    print(f"  data = np.load('{args.output_path or Path(args.json_path).with_suffix('.npy')}')")
    print(f"  # 然后在模型推理前添加batch维度并转换为tensor")

