#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
找出没有对应关键点文件的视频

这个脚本会检查video文件夹中的视频文件，找出那些在keypoints文件夹中
没有对应关键点文件的视频（因为背景复杂等原因未能成功提取关键点）。
"""

import os
from pathlib import Path
from collections import defaultdict


def find_missing_keypoints(video_dir: str, keypoints_dir: str):
    """
    找出没有对应关键点文件的视频
    
    Args:
        video_dir: 视频文件夹路径
        keypoints_dir: 关键点文件夹路径
    """
    video_path = Path(video_dir)
    keypoints_path = Path(keypoints_dir)
    
    if not video_path.exists():
        print(f"错误: 视频文件夹不存在: {video_dir}")
        return
    
    if not keypoints_path.exists():
        print(f"错误: 关键点文件夹不存在: {keypoints_dir}")
        return
    
    # 存储缺失关键点的视频
    missing_videos = defaultdict(list)
    total_videos = 0
    total_keypoints = 0
    
    # 遍历所有视频子文件夹
    for action_dir in video_path.iterdir():
        if not action_dir.is_dir():
            continue
        
        action_name = action_dir.name
        keypoints_action_dir = keypoints_path / action_name
        
        # 获取该动作的所有视频文件
        video_files = list(action_dir.glob("*.mp4"))
        total_videos += len(video_files)
        
        # 获取该动作的所有关键点文件（不包含扩展名）
        if keypoints_action_dir.exists():
            keypoint_files = {f.stem for f in keypoints_action_dir.glob("*.json")}
            total_keypoints += len(keypoint_files)
        else:
            keypoint_files = set()
            print(f"警告: 关键点文件夹不存在: {keypoints_action_dir}")
        
        # 找出没有对应关键点的视频
        for video_file in video_files:
            video_stem = video_file.stem
            if video_stem not in keypoint_files:
                missing_videos[action_name].append(video_file)
    
    # 打印结果
    print("=" * 80)
    print("缺失关键点的视频统计")
    print("=" * 80)
    print(f"\n总视频数: {total_videos}")
    print(f"总关键点数: {total_keypoints}")
    print(f"缺失关键点的视频数: {sum(len(videos) for videos in missing_videos.values())}")
    print()
    
    if missing_videos:
        print("按动作分类的缺失视频:")
        print("-" * 80)
        
        for action_name, videos in sorted(missing_videos.items()):
            print(f"\n【{action_name}】")
            print(f"  缺失数量: {len(videos)}")
            print(f"  缺失的视频文件:")
            for video in sorted(videos):
                print(f"    - {video.name}")
        
        # 保存到文件
        output_file = Path(video_dir).parent / "missing_keypoints.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("缺失关键点的视频列表\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"总视频数: {total_videos}\n")
            f.write(f"总关键点数: {total_keypoints}\n")
            f.write(f"缺失关键点的视频数: {sum(len(videos) for videos in missing_videos.values())}\n\n")
            
            for action_name, videos in sorted(missing_videos.items()):
                f.write(f"\n【{action_name}】\n")
                f.write(f"缺失数量: {len(videos)}\n")
                f.write(f"缺失的视频文件:\n")
                for video in sorted(videos):
                    f.write(f"  {video}\n")
        
        print(f"\n结果已保存到: {output_file}")
    else:
        print("✓ 所有视频都有对应的关键点文件！")
    
    return missing_videos


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='找出没有对应关键点文件的视频')
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
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = Path(__file__).parent
    video_dir = (script_dir / args.video_dir).resolve()
    keypoints_dir = (script_dir / args.keypoints_dir).resolve()
    
    find_missing_keypoints(str(video_dir), str(keypoints_dir))
