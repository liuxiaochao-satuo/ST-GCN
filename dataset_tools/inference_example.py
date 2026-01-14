#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-GCN模型推理示例

展示如何使用训练好的ST-GCN模型进行动作分类推理

使用方法：
    python inference_example.py \
        --json_path <关键点JSON文件路径> \
        --model_path <训练好的模型权重路径> \
        --config_path <模型配置文件路径>
"""

import os
import sys
import argparse
import numpy as np
import torch

# 添加ST-GCN路径
current_dir = os.path.dirname(os.path.abspath(__file__))
st_gcn_path = os.path.abspath(os.path.join(current_dir, '..'))
if st_gcn_path not in sys.path:
    sys.path.insert(0, st_gcn_path)

from dataset_tools.convert_json_to_stgcn_input import (
    json_to_stgcn_input,
    prepare_for_model_inference
)
from processor.io import IO
from processor.recognition import REC_Processor


# 动作类别映射（与训练时一致）
ACTION_CLASSES = {
    'jumping': 0,
    'jump_to_leg_sit': 1,
    'front_swing': 2,
    'back_swing': 3,
    'front_swing_down': 4,
    'back_swing_down': 5,
}

ACTION_NAMES = {v: k for k, v in ACTION_CLASSES.items()}


def load_model(config_path: str, model_path: str, device: str = 'cuda:0'):
    """
    加载训练好的ST-GCN模型
    
    Args:
        config_path: 配置文件路径（YAML格式）
        model_path: 模型权重文件路径（.pth文件）
        device: 使用的设备
    
    Returns:
        加载好的模型
    """
    import yaml
    
    # 读取配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建IO对象
    io = IO(config['work_dir'], resume=model_path)
    
    # 加载模型
    model = io.load_model(
        config['model'],
        **config['model_args']
    )
    
    model.eval()
    model = model.to(device)
    
    # 加载权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ 已加载模型权重: {model_path}")
    else:
        print(f"⚠️  警告: 模型权重文件不存在: {model_path}")
    
    return model


def predict_action(
    json_path: str,
    model,
    device: str = 'cuda:0',
    max_frame: int = 300,
    num_joints: int = 21
):
    """
    使用ST-GCN模型预测动作类别
    
    Args:
        json_path: 关键点JSON文件路径
        model: 加载好的ST-GCN模型
        device: 使用的设备
        max_frame: 最大帧数
        num_joints: 关键点数量
    
    Returns:
        预测的动作类别名称和置信度
    """
    # 1. 将JSON转换为ST-GCN输入格式
    print(f"正在转换JSON文件: {json_path}")
    data_numpy = json_to_stgcn_input(
        json_path,
        max_frame=max_frame,
        num_joints=num_joints
    )
    print(f"✓ 数据形状: {data_numpy.shape} (C, T, V, M)")
    
    # 2. 准备模型输入
    data = prepare_for_model_inference(data_numpy, device=device)
    print(f"✓ 模型输入形状: {data.shape} (N, C, T, V, M)")
    
    # 3. 模型推理
    print("正在进行模型推理...")
    with torch.no_grad():
        output = model(data)
    
    # 4. 获取预测结果
    probabilities = torch.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()
    
    # 5. 获取所有类别的概率
    all_probs = probabilities[0].cpu().numpy()
    
    return predicted_class, confidence, all_probs


def main():
    parser = argparse.ArgumentParser(description='ST-GCN模型推理示例')
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='关键点JSON文件路径'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='训练好的模型权重路径（.pth文件）'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='config/st_gcn/parallel-bars/train.yaml',
        help='模型配置文件路径（YAML格式）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='使用的设备（cuda:0或cpu）'
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
    
    # 检查文件是否存在
    if not os.path.exists(args.json_path):
        print(f"错误: JSON文件不存在: {args.json_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型权重文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 加载模型
    print("正在加载模型...")
    model = load_model(args.config_path, args.model_path, device=args.device)
    
    # 预测动作
    predicted_class, confidence, all_probs = predict_action(
        args.json_path,
        model,
        device=args.device,
        max_frame=args.max_frame,
        num_joints=args.num_joints
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("预测结果:")
    print("=" * 60)
    print(f"预测动作: {ACTION_NAMES.get(predicted_class, f'未知({predicted_class})')}")
    print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
    print("\n所有动作类别的概率:")
    for class_idx, class_name in sorted(ACTION_NAMES.items()):
        prob = all_probs[class_idx]
        marker = " <-- 预测" if class_idx == predicted_class else ""
        print(f"  {class_name:20s}: {prob:.4f} ({prob*100:.2f}%){marker}")
    print("=" * 60)


if __name__ == '__main__':
    main()

