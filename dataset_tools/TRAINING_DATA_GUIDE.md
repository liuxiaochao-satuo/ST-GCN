# ST-GCN 训练数据组织指南

## 数据组织流程

你已经完成了关键点提取，数据存放在 `data/keypoints/` 下，按类别分目录：
```
data/keypoints/
├── jump_to_leg_sit/     (206个JSON文件)
├── front_swing/         (240个JSON文件)
├── back_swing/          (158个JSON文件)
├── front_swing_down/    (57个JSON文件)
└── back_swing_down/     (64个JSON文件)
```

## 步骤1: 组织训练数据

将所有JSON文件合并并划分训练集和验证集：

```bash
cd /home/satuo/code/st-gcn

# 使用默认设置（80%训练，20%验证）
python dataset_tools/organize_training_data.py \
    --keypoints_dir data/keypoints \
    --output_dir data/training

# 或者使用符号链接（节省磁盘空间）
python dataset_tools/organize_training_data.py \
    --keypoints_dir data/keypoints \
    --output_dir data/training \
    --use_symlink

# 自定义训练集比例（例如70%训练，30%验证）
python dataset_tools/organize_training_data.py \
    --keypoints_dir data/keypoints \
    --output_dir data/training \
    --train_ratio 0.7
```

**输出结构：**
```
data/training/
├── train/          # 训练集JSON文件
└── val/            # 验证集JSON文件
```

## 步骤2: 生成ST-GCN训练数据集

### 2.1 生成训练集

```bash
python dataset_tools/generate_stgcn_dataset.py \
    --data_path data/training/train \
    --out_folder data/training \
    --split train \
    --num_joints 21 \
    --max_frame 300 \
    --create_label
```

### 2.2 生成验证集

```bash
python dataset_tools/generate_stgcn_dataset.py \
    --data_path data/training/val \
    --out_folder data/training \
    --split val \
    --num_joints 21 \
    --max_frame 300 \
    --create_label
```

## 步骤3: 检查生成的数据

生成完成后，`data/training/` 目录下应该有以下文件：

```
data/training/
├── train/
│   └── *.json          # 训练集JSON文件
├── val/
│   └── *.json          # 验证集JSON文件
├── train_data.npy      # 训练数据（N, 3, T, V, M）
├── train_label.pkl     # 训练标签
├── train_label.json     # 训练标签JSON（可选）
├── val_data.npy         # 验证数据
├── val_label.pkl        # 验证标签
└── val_label.json       # 验证标签JSON（可选）
```

## 数据格式说明

### NPY数据格式
- **形状**: `(N, 3, T, V, M)`
  - `N`: 样本数量
  - `3`: 通道数（x坐标, y坐标, score置信度）
  - `T`: 时间帧数（最大300帧）
  - `V`: 关节数（21个关键点）
  - `M`: 人数（通常为1）

### 标签格式
- **PKL文件**: 包含 `(sample_names, labels)` 元组
- **JSON文件**: 包含每个样本的详细信息

### 动作类别映射
```python
{
    'jump_to_leg_sit': 0,
    'front_swing': 1,
    'back_swing': 2,
    'front_swing_down': 3,
    'back_swing_down': 4,
}
```

## 步骤4: 开始训练

数据准备完成后，可以使用ST-GCN的主训练脚本进行训练：

```bash
python main.py \
    --config config/st_gcn/parallel_bars.yaml \
    --work_dir work/parallel_bars \
    --device 0
```

## 注意事项

1. **数据平衡**: 如果各类别样本数量差异较大，建议：
   - 使用数据增强
   - 调整类别权重
   - 使用过采样或欠采样

2. **帧数限制**: 默认最大帧数为300，如果视频更长会被截断，更短会被填充

3. **内存使用**: 
   - 如果使用符号链接（`--use_symlink`），可以节省磁盘空间
   - NPY文件会占用较多内存，确保有足够空间

4. **随机种子**: 默认使用随机种子42，确保结果可复现

## 快速检查脚本

检查数据是否正确组织：

```python
import numpy as np
import pickle

# 检查训练数据
train_data = np.load('data/training/train_data.npy')
train_labels = pickle.load(open('data/training/train_label.pkl', 'rb'))

print(f"训练数据形状: {train_data.shape}")
print(f"训练样本数: {len(train_labels[0])}")
print(f"类别分布: {np.bincount(train_labels[1])}")
```

## 故障排除

1. **找不到文件**: 确保路径正确，使用绝对路径
2. **内存不足**: 减少 `max_frame` 或分批处理
3. **标签不匹配**: 检查JSON文件中的 `label_index` 是否正确
