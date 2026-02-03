#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对所有视频和关键点文件进行统一排序重命名

规则：
1. 先排序有关键点的视频和JSON文件（00001.mp4, 00001.json等）
2. 然后排序没有关键点的视频（从有关键点的数量+1开始）
3. 总共915个视频，排到00915
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict


def collect_all_files(video_dir: str, keypoints_dir: str) -> Tuple[List[Path], List[Path], Dict[str, Path]]:
    """
    收集所有视频文件和关键点文件，并建立匹配关系
    
    Returns:
        videos_with_keypoints: 有关键点的视频列表
        videos_without_keypoints: 没有关键点的视频列表
        keypoint_map: 视频文件到关键点文件的映射
    """
    video_path = Path(video_dir)
    keypoints_path = Path(keypoints_dir)
    
    videos_with_keypoints = []
    videos_without_keypoints = []
    keypoint_map = {}
    
    # 遍历所有视频子文件夹
    for action_dir in sorted(video_path.iterdir()):
        if not action_dir.is_dir():
            continue
        
        action_name = action_dir.name
        keypoints_action_dir = keypoints_path / action_name
        
        # 获取该动作的所有视频文件
        video_files = sorted(action_dir.glob("*.mp4"))
        
        # 获取该动作的所有关键点文件
        if keypoints_action_dir.exists():
            keypoint_files = {f.stem: f for f in keypoints_action_dir.glob("*.json")}
        else:
            keypoint_files = {}
        
        # 分类视频文件
        for video_file in video_files:
            video_stem = video_file.stem
            if video_stem in keypoint_files:
                videos_with_keypoints.append(video_file)
                keypoint_map[video_file] = keypoint_files[video_stem]
            else:
                videos_without_keypoints.append(video_file)
    
    return videos_with_keypoints, videos_without_keypoints, keypoint_map


def rename_files(
    videos_with_keypoints: List[Path],
    videos_without_keypoints: List[Path],
    keypoint_map: Dict[Path, Path],
    video_dir: str,
    keypoints_dir: str,
    output_video_dir: str = None,
    output_keypoints_dir: str = None,
    dry_run: bool = False
):
    """
    重命名所有文件
    
    Args:
        videos_with_keypoints: 有关键点的视频列表
        videos_without_keypoints: 没有关键点的视频列表
        keypoint_map: 视频文件到关键点文件的映射
        video_dir: 视频文件夹路径
        keypoints_dir: 关键点文件夹路径
        output_video_dir: 输出视频文件夹路径（如果为None，则在原目录重命名）
        output_keypoints_dir: 输出关键点文件夹路径（如果为None，则在原目录重命名）
        dry_run: 如果为True，只显示将要执行的操作，不实际重命名
    """
    video_path = Path(video_dir)
    keypoints_path = Path(keypoints_dir)
    
    # 确定输出目录
    if output_video_dir is None:
        output_video_dir = video_dir
    if output_keypoints_dir is None:
        output_keypoints_dir = keypoints_dir
    
    output_video_path = Path(output_video_dir)
    output_keypoints_path = Path(output_keypoints_dir)
    
    # 创建输出目录（如果不存在）
    if not dry_run:
        output_video_path.mkdir(parents=True, exist_ok=True)
        output_keypoints_path.mkdir(parents=True, exist_ok=True)
    
    # 创建重命名计划
    rename_plan = []
    counter = 1
    
    # 1. 先处理有关键点的视频和JSON文件
    print("=" * 80)
    print("步骤1: 重命名有关键点的视频和JSON文件")
    print("=" * 80)
    
    for video_file in videos_with_keypoints:
        new_name = f"{counter:05d}"
        new_video_name = f"{new_name}.mp4"
        new_keypoint_name = f"{new_name}.json"
        
        # 确定新的路径（统一输出到指定目录）
        new_video_path = output_video_path / new_video_name
        keypoint_file = keypoint_map[video_file]
        new_keypoint_path = output_keypoints_path / new_keypoint_name
        
        rename_plan.append({
            'type': 'with_keypoint',
            'video_old': video_file,
            'video_new': new_video_path,
            'keypoint_old': keypoint_file,
            'keypoint_new': new_keypoint_path,
            'number': counter
        })
        
        counter += 1
    
    # 2. 处理没有关键点的视频
    print("\n" + "=" * 80)
    print("步骤2: 重命名没有关键点的视频")
    print("=" * 80)
    
    for video_file in videos_without_keypoints:
        new_name = f"{counter:05d}"
        new_video_name = f"{new_name}.mp4"
        
        new_video_path = output_video_path / new_video_name
        
        rename_plan.append({
            'type': 'without_keypoint',
            'video_old': video_file,
            'video_new': new_video_path,
            'number': counter
        })
        
        counter += 1
    
    # 显示重命名计划
    print(f"\n总共需要重命名 {len(rename_plan)} 个文件")
    print(f"  有关键点的视频: {len(videos_with_keypoints)}")
    print(f"  没有关键点的视频: {len(videos_without_keypoints)}")
    
    if dry_run:
        print("\n【预览模式】将要执行的重命名操作：")
        print("-" * 80)
        for i, plan in enumerate(rename_plan[:10], 1):  # 只显示前10个
            if plan['type'] == 'with_keypoint':
                print(f"{plan['number']:05d}. {plan['video_old'].name} -> {plan['video_new'].name}")
                print(f"     {plan['keypoint_old'].name} -> {plan['keypoint_new'].name}")
            else:
                print(f"{plan['number']:05d}. {plan['video_old'].name} -> {plan['video_new'].name}")
        if len(rename_plan) > 10:
            print(f"... 还有 {len(rename_plan) - 10} 个文件")
        print("\n使用 --execute 参数来实际执行重命名操作")
        return
    
    # 执行重命名（分两步：先重命名为临时名称，再重命名为最终名称）
    print("\n开始执行重命名操作...")
    print("-" * 80)
    
    # 第一步：移动到输出目录并重命名为临时名称（避免冲突）
    temp_renames = []
    for plan in rename_plan:
        if plan['type'] == 'with_keypoint':
            # 视频文件：先移动到输出目录的临时名称
            temp_video = output_video_path / f"__temp_{plan['number']:05d}__.mp4"
            if plan['video_old'].exists():
                shutil.move(str(plan['video_old']), str(temp_video))
                temp_renames.append(('video', temp_video, plan['video_new']))
            
            # 关键点文件：先移动到输出目录的临时名称
            temp_keypoint = output_keypoints_path / f"__temp_{plan['number']:05d}__.json"
            if plan['keypoint_old'].exists():
                shutil.move(str(plan['keypoint_old']), str(temp_keypoint))
                temp_renames.append(('keypoint', temp_keypoint, plan['keypoint_new']))
        else:
            # 只有视频文件：先移动到输出目录的临时名称
            temp_video = output_video_path / f"__temp_{plan['number']:05d}__.mp4"
            if plan['video_old'].exists():
                shutil.move(str(plan['video_old']), str(temp_video))
                temp_renames.append(('video', temp_video, plan['video_new']))
    
    # 第二步：从临时名称重命名为最终名称
    success_count = 0
    for file_type, temp_path, final_path in temp_renames:
        try:
            shutil.move(str(temp_path), str(final_path))
            success_count += 1
            if success_count % 50 == 0:
                print(f"  已处理 {success_count}/{len(temp_renames)} 个文件...")
        except Exception as e:
            print(f"  错误: 重命名 {temp_path} -> {final_path} 失败: {e}")
    
    print(f"\n✓ 重命名完成！成功处理 {success_count}/{len(temp_renames)} 个文件")
    
    # 生成重命名日志
    log_file = output_video_path.parent / "rename_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("视频和关键点文件重命名日志\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"总文件数: {len(rename_plan)}\n")
        f.write(f"有关键点的视频: {len(videos_with_keypoints)}\n")
        f.write(f"没有关键点的视频: {len(videos_without_keypoints)}\n\n")
        f.write("详细重命名记录:\n")
        f.write("-" * 80 + "\n")
        
        for plan in rename_plan:
            if plan['type'] == 'with_keypoint':
                f.write(f"{plan['number']:05d}. {plan['video_old']} -> {plan['video_new']}\n")
                f.write(f"     {plan['keypoint_old']} -> {plan['keypoint_new']}\n")
            else:
                f.write(f"{plan['number']:05d}. {plan['video_old']} -> {plan['video_new']}\n")
    
    print(f"重命名日志已保存到: {log_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='对所有视频和关键点文件进行统一排序重命名')
    parser.add_argument(
        '--video_dir',
        type=str,
        default='../data/video',
        help='视频文件夹路径（默认: ../data/video）'
    )
    parser.add_argument(
        '--keypoints_dir',
        type=str,
        default='../data/keypoints',
        help='关键点文件夹路径（默认: ../data/keypoints）'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='实际执行重命名操作（默认只预览）'
    )
    parser.add_argument(
        '--output_video_dir',
        type=str,
        default=None,
        help='输出视频文件夹路径（默认: 与原目录相同，统一到一个目录）'
    )
    parser.add_argument(
        '--output_keypoints_dir',
        type=str,
        default=None,
        help='输出关键点文件夹路径（默认: 与原目录相同，统一到一个目录）'
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = Path(__file__).parent
    video_dir = (script_dir / args.video_dir).resolve()
    keypoints_dir = (script_dir / args.keypoints_dir).resolve()
    
    if not video_dir.exists():
        print(f"错误: 视频文件夹不存在: {video_dir}")
        return
    
    if not keypoints_dir.exists():
        print(f"错误: 关键点文件夹不存在: {keypoints_dir}")
        return
    
    # 收集所有文件
    print("正在收集文件信息...")
    videos_with_keypoints, videos_without_keypoints, keypoint_map = collect_all_files(
        str(video_dir), str(keypoints_dir)
    )
    
    print(f"\n文件统计:")
    print(f"  有关键点的视频: {len(videos_with_keypoints)}")
    print(f"  没有关键点的视频: {len(videos_without_keypoints)}")
    print(f"  总计: {len(videos_with_keypoints) + len(videos_without_keypoints)}")
    
    # 确定输出目录（如果未指定，则使用统一目录）
    if args.output_video_dir is None:
        output_video_dir = video_dir.parent / "video_sorted"
    else:
        output_video_dir = (script_dir / args.output_video_dir).resolve()
    
    if args.output_keypoints_dir is None:
        output_keypoints_dir = keypoints_dir.parent / "keypoints_sorted"
    else:
        output_keypoints_dir = (script_dir / args.output_keypoints_dir).resolve()
    
    print(f"\n输出目录:")
    print(f"  视频: {output_video_dir}")
    print(f"  关键点: {output_keypoints_dir}")
    
    # 执行重命名
    rename_files(
        videos_with_keypoints,
        videos_without_keypoints,
        keypoint_map,
        str(video_dir),
        str(keypoints_dir),
        str(output_video_dir),
        str(output_keypoints_dir),
        dry_run=not args.execute
    )


if __name__ == '__main__':
    main()
