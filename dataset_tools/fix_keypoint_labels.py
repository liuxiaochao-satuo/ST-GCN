#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正 data/keypoints 下 JSON 文件的 label / label_index 字段

场景：
- 早期生成的 JSON 中，前摆下 / 后摆下 的 label、label_index 可能被写成
  front_swing / back_swing（1/2），而不是 front_swing_down / back_swing_down（3/4）
- 现在希望统一按文件名来确定动作类别，并把 JSON 里的字段也修正过来

使用方法：
    cd /home/satuo/code/st-gcn
    python dataset_tools/fix_keypoint_labels.py \
        --keypoints_root data/keypoints
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict


# 与 extract_keypoints_from_clipped_videos.py / generate_stgcn_dataset.py 保持一致
ACTION_CLASSES: Dict[str, int] = {
    "jump_to_leg_sit": 0,
    "front_swing": 1,
    "back_swing": 2,
    "front_swing_down": 3,
    "back_swing_down": 4,
}


def infer_label_from_filename(json_path: Path) -> (str, int):
    """根据文件名推断动作类别及索引（按名称长度从长到短匹配）"""
    name = json_path.name.lower()
    for action_name, idx in sorted(
        ACTION_CLASSES.items(), key=lambda x: -len(x[0])
    ):
        if action_name in name:
            return action_name, idx
    return "unknown", -1


def fix_single_file(json_path: Path) -> bool:
    """修正单个 JSON 文件，返回是否修改过"""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ⚠️  读取失败: {json_path} => {e}")
        return False

    # 推断应当的标签
    filename_label, filename_idx = infer_label_from_filename(json_path)

    json_label = data.get("label")
    json_idx = data.get("label_index", -1)

    need_fix = False

    # 以文件名推断结果为准（如果识别出来）
    if filename_idx != -1:
        if json_label != filename_label or json_idx != filename_idx:
            need_fix = True
            data["label"] = filename_label
            data["label_index"] = filename_idx
    else:
        # 文件名里识别不出类别，则只在 JSON 本身是合法类别时做规范化
        if json_label in ACTION_CLASSES:
            correct_idx = ACTION_CLASSES[json_label]
            if json_idx != correct_idx:
                need_fix = True
                data["label_index"] = correct_idx

    if need_fix:
        try:
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"  ⚠️  写回失败: {json_path} => {e}")
            return False

    return False


def fix_keypoints_root(root: str):
    root_path = Path(root)
    if not root_path.exists():
        print(f"错误: 目录不存在: {root}")
        return

    print(f"开始修正 JSON 标签: {root_path}")
    total = 0
    fixed = 0

    for json_path in root_path.rglob("*.json"):
        total += 1
        changed = fix_single_file(json_path)
        if changed:
            fixed += 1
            # 只对数量较少的类别打印详细信息，避免输出过多
            name = json_path.name
            if "front_swing_down" in name or "back_swing_down" in name:
                print(f"  ✓ 修正: {json_path.relative_to(root_path)}")

    print("\n修正完成:")
    print(f"  总文件数: {total}")
    print(f"  修正文件: {fixed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="修正 data/keypoints 下 JSON 的 label / label_index 字段"
    )
    parser.add_argument(
        "--keypoints_root",
        type=str,
        default="data/keypoints",
        help="关键点 JSON 根目录（默认: data/keypoints）",
    )

    args = parser.parse_args()
    fix_keypoints_root(args.keypoints_root)

