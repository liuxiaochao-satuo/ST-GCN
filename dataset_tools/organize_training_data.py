#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组织训练数据脚本

功能：
1. 将所有类别的JSON文件合并到一个目录
2. 自动划分训练集和验证集
3. 创建符号链接或复制文件到训练/验证目录

使用方法：
    python organize_training_data.py \
        --keypoints_dir data/keypoints \
        --output_dir data/training \
        --train_ratio 0.8
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List
import random

# 动作类别映射（与extract_keypoints_from_clipped_videos.py保持一致）
ACTION_CLASSES = {
    'jump_to_leg_sit': 0,
    'front_swing': 1,
    'back_swing': 2,
    'front_swing_down': 3,
    'back_swing_down': 4,
}


def collect_json_files(keypoints_dir: str) -> Dict[str, List[str]]:
    """
    收集所有类别的JSON文件
    
    Args:
        keypoints_dir: 关键点数据根目录
        
    Returns:
        按类别组织的文件路径字典
    """
    category_files = {}
    keypoints_path = Path(keypoints_dir)
    
    if not keypoints_path.exists():
        print(f"错误: 目录不存在: {keypoints_dir}")
        return category_files
    
    # 遍历所有子目录
    for category_dir in keypoints_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        if category_name not in ACTION_CLASSES:
            print(f"警告: 未知的类别目录: {category_name}，跳过")
            continue
        
        # 收集该类别下的所有JSON文件
        json_files = sorted(category_dir.glob('*.json'))
        if json_files:
            category_files[category_name] = [str(f) for f in json_files]
            print(f"  类别 {category_name}: {len(json_files)} 个文件")
    
    return category_files


def split_train_val(files: List[str], train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """
    划分训练集和验证集
    
    Args:
        files: 文件路径列表
        train_ratio: 训练集比例
        seed: 随机种子
        
    Returns:
        (train_files, val_files)
    """
    random.seed(seed)
    shuffled_files = files.copy()
    random.shuffle(shuffled_files)
    
    split_idx = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    return train_files, val_files


def organize_data(
    keypoints_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    use_symlink: bool = False,
    seed: int = 42
):
    """
    组织训练数据
    
    Args:
        keypoints_dir: 关键点数据根目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        use_symlink: 是否使用符号链接（True）还是复制文件（False）
        seed: 随机种子
    """
    print("=" * 60)
    print("开始组织训练数据...")
    print("=" * 60)
    
    # 收集所有文件
    print("\n收集JSON文件...")
    category_files = collect_json_files(keypoints_dir)
    
    if not category_files:
        print("错误: 没有找到任何JSON文件")
        return
    
    total_files = sum(len(files) for files in category_files.values())
    print(f"\n总计: {total_files} 个文件，{len(category_files)} 个类别")
    
    # 创建输出目录
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  使用方式: {'符号链接' if use_symlink else '复制文件'}")
    
    # 统计信息
    train_count = 0
    val_count = 0
    category_stats = {}
    
    # 按类别划分数据
    print("\n划分数据集...")
    for category_name, files in category_files.items():
        train_files, val_files = split_train_val(files, train_ratio, seed)
        
        category_stats[category_name] = {
            'total': len(files),
            'train': len(train_files),
            'val': len(val_files)
        }
        
        print(f"  {category_name}: 总计={len(files)}, 训练={len(train_files)}, 验证={len(val_files)}")
        
        # 复制或链接训练集文件
        for file_path in train_files:
            src = Path(file_path)
            dst = train_dir / src.name
            
            if use_symlink:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(os.path.abspath(src))
            else:
                shutil.copy2(src, dst)
            train_count += 1
        
        # 复制或链接验证集文件
        for file_path in val_files:
            src = Path(file_path)
            dst = val_dir / src.name
            
            if use_symlink:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(os.path.abspath(src))
            else:
                shutil.copy2(src, dst)
            val_count += 1
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("数据组织完成!")
    print("=" * 60)
    print(f"\n统计信息:")
    print(f"  总文件数: {total_files}")
    print(f"  训练集: {train_count} ({train_count/total_files*100:.1f}%)")
    print(f"  验证集: {val_count} ({val_count/total_files*100:.1f}%)")
    
    print(f"\n各类别分布:")
    for category, stats in sorted(category_stats.items(), key=lambda x: ACTION_CLASSES[x[0]]):
        label_idx = ACTION_CLASSES[category]
        print(f"  [{label_idx}] {category}:")
        print(f"    总计: {stats['total']}")
        print(f"    训练: {stats['train']} ({stats['train']/stats['total']*100:.1f}%)")
        print(f"    验证: {stats['val']} ({stats['val']/stats['total']*100:.1f}%)")
    
    print(f"\n下一步:")
    print(f"  1. 生成训练数据集:")
    print(f"     python dataset_tools/generate_stgcn_dataset.py \\")
    print(f"         --data_path {train_dir} \\")
    print(f"         --out_folder {output_path} \\")
    print(f"         --split train \\")
    print(f"         --create_label")
    print(f"")
    print(f"  2. 生成验证数据集:")
    print(f"     python dataset_tools/generate_stgcn_dataset.py \\")
    print(f"         --data_path {val_dir} \\")
    print(f"         --out_folder {output_path} \\")
    print(f"         --split val \\")
    print(f"         --create_label")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='组织ST-GCN训练数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置（80%训练，20%验证）
  python organize_training_data.py \\
      --keypoints_dir data/keypoints \\
      --output_dir data/training

  # 自定义训练集比例
  python organize_training_data.py \\
      --keypoints_dir data/keypoints \\
      --output_dir data/training \\
      --train_ratio 0.7

  # 使用符号链接（节省磁盘空间）
  python organize_training_data.py \\
      --keypoints_dir data/keypoints \\
      --output_dir data/training \\
      --use_symlink
        """
    )
    
    parser.add_argument('--keypoints_dir', type=str, required=True,
                        help='关键点数据根目录（包含各类别子目录）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录（将创建train和val子目录）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认: 0.8）')
    parser.add_argument('--use_symlink', action='store_true',
                        help='使用符号链接而不是复制文件（节省磁盘空间）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    
    args = parser.parse_args()
    
    organize_data(
        args.keypoints_dir,
        args.output_dir,
        args.train_ratio,
        args.use_symlink,
        args.seed
    )
