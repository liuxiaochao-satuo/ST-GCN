#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ST-GCN 训练日志中解析并绘制损失/精度曲线

用法示例：
    cd /home/satuo/code/st-gcn
    python tools/plot_training_curves.py \
        --log_path work_dir/recognition/parallel-bars/ST_GCN_run1/log.txt \
        --out_dir work_dir/recognition/parallel-bars/ST_GCN_run1
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


def parse_log(log_path: str) -> Dict[str, List[float]]:
    """解析 ST-GCN 的 log.txt，提取 train / val 的 mean_loss 和 Top1"""
    train_loss = []
    val_loss = []
    val_top1 = []
    train_epochs = []
    val_epochs = []

    epoch_train = None
    epoch_eval = None

    # 正则模式
    re_train_epoch = re.compile(r"Training epoch:\s+(\d+)")
    re_eval_epoch = re.compile(r"Eval epoch:\s+(\d+)")
    re_mean_loss = re.compile(r"mean_loss:\s+([0-9.+-eE]+)")
    re_top1 = re.compile(r"Top1:\s+([0-9.]+)%")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "Training epoch:" in line:
                m = re_train_epoch.search(line)
                if m:
                    epoch_train = int(m.group(1))
                continue

            if "Eval epoch:" in line:
                m = re_eval_epoch.search(line)
                if m:
                    epoch_eval = int(m.group(1))
                continue

            if "mean_loss:" in line:
                m = re_mean_loss.search(line)
                if not m:
                    continue
                loss = float(m.group(1))
                # 按最近一次出现的是 train 还是 eval 来归类
                if epoch_eval is not None:
                    val_epochs.append(epoch_eval)
                    val_loss.append(loss)
                    epoch_eval = None  # 重置
                elif epoch_train is not None:
                    train_epochs.append(epoch_train)
                    train_loss.append(loss)
                    epoch_train = None  # 重置
                continue

            if "Top1:" in line:
                m = re_top1.search(line)
                if m:
                    acc = float(m.group(1))
                    if epoch_eval is not None:
                        # 有些日志中 Top1 在 mean_loss 后面，此时 epoch_eval 仍为最近一次 Eval epoch
                        val_top1.append((epoch_eval, acc))
                    else:
                        # 兜底：如果 epoch_eval 已被重置，则使用最后一个 val_epochs
                        if val_epochs:
                            val_top1.append((val_epochs[-1], acc))

    return {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "val_epochs": val_epochs,
        "val_loss": val_loss,
        "val_top1": val_top1,
    }


def plot_curves(stats: Dict[str, List[float]], out_dir: str):
    out_path_loss = Path(out_dir) / "loss_curve.png"
    out_path_acc = Path(out_dir) / "val_top1_curve.png"

    # 损失曲线
    plt.figure(figsize=(8, 5))
    if stats["train_epochs"] and stats["train_loss"]:
        plt.plot(stats["train_epochs"], stats["train_loss"], label="Train loss")
    if stats["val_epochs"] and stats["val_loss"]:
        plt.plot(stats["val_epochs"], stats["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_loss)
    plt.close()

    # 验证 Top1 曲线
    if stats["val_top1"]:
        epochs, accs = zip(*stats["val_top1"])
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, accs, marker="o", label="Val Top1 (%)")
        plt.xlabel("Epoch")
        plt.ylabel("Top1 Accuracy (%)")
        plt.title("Validation Top1 Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path_acc)
        plt.close()

    print(f"已保存损失曲线到: {out_path_loss}")
    if stats["val_top1"]:
        print(f"已保存验证 Top1 曲线到: {out_path_acc}")


def main():
    parser = argparse.ArgumentParser(description="绘制 ST-GCN 训练过程曲线")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="训练日志路径，例如 work_dir/.../log.txt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出图片目录（默认与 log.txt 同目录）",
    )

    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"错误: 找不到日志文件: {log_path}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else log_path.parent
    os.makedirs(out_dir, exist_ok=True)

    stats = parse_log(str(log_path))
    if not stats["train_loss"] and not stats["val_loss"]:
        print("警告: 未在日志中解析到损失信息，请检查 log.txt 格式")
        return

    plot_curves(stats, str(out_dir))


if __name__ == "__main__":
    main()

