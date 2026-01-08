# 双杠动作数据集构建完整工作流程

## 工作流程概述

基于您的工作计划，完整的工作流程如下：

```
1. 人工裁剪视频
   └─> 将完整双杠视频裁剪为各个子动作视频片段
   
2. 关键点提取
   └─> 对裁剪后的子动作视频进行21点关键点提取
   
3. 格式转换
   └─> 将关键点数据转换为ST-GCN所需的JSON格式
   
4. 数据集生成
   └─> 生成ST-GCN训练所需的NPY和PKL文件
   
5. 模型训练
   └─> 使用生成的数据集训练ST-GCN模型
```

## 详细步骤

### 步骤1: 人工裁剪视频

将完整的双杠动作视频裁剪为6个子动作视频片段：

- `jumping`: 起跳
- `jump_to_leg_sit`: 跳上成支撑
- `front_swing`: 前摆
- `back_swing`: 后摆
- `front_swing_down`: 前摆下
- `back_swing_down`: 后摆下

**建议的文件命名方式：**
- 在文件名中包含动作标签，例如：`front_swing_001.mp4` 或 `前摆_001.mp4`
- 或者将不同动作的视频放在不同的子目录中

### 步骤2: 关键点提取

使用21点姿态估计模型从裁剪后的视频中提取关键点。

**方法A: 使用demo_mmpose.py（推荐）**

由于`demo_mmpose.py`已经包含了完整的视频处理逻辑，建议：

1. **修改demo_mmpose.py**：添加一个方法，只提取关键点而不进行动作分析

   ```python
   def extract_keypoints_only(self, video_path, output_json_path):
       """只提取关键点，不进行动作分析和视频保存"""
       # 初始化模型（与process_video相同）
       # 处理视频帧，提取关键点
       # 保存为JSON格式
       pass
   ```

2. **或者创建新脚本**：基于`demo_mmpose.py`创建一个简化版本

   ```bash
   python extract_keypoints_from_clipped_videos.py \
       --video_path <视频路径> \
       --output_dir <输出目录> \
       --action_label <动作标签>
   ```

**方法B: 使用批量处理（自动识别动作标签）**

```bash
# 处理整个目录的视频，自动根据文件名识别动作标签
python extract_keypoints_from_clipped_videos.py \
    --video_dir /path/to/clipped_videos \
    --output_dir /path/to/keypoint_output \
    --device cuda:0
```

**输出格式：**
每个视频会生成一个JSON文件，包含：
- 每帧的关键点坐标（归一化到[0, 1]）
- 每帧的关键点置信度分数
- 视频元信息（分辨率、帧率等）
- 动作标签和标签索引

### 步骤3: 格式转换（可选）

如果关键点数据格式不符合要求，使用`keep_keypoints.py`进行转换：

```bash
python keep_keypoints.py \
    --keypoint_data <关键点数据文件> \
    --output_dir <输出目录> \
    --action_label <动作标签> \
    --frame_width 1920 \
    --frame_height 1080
```

### 步骤4: 数据集生成

使用`generate_stgcn_dataset.py`将多个JSON文件组织成训练数据集：

```bash
# 1. 创建标签文件（从JSON文件自动生成）
python generate_stgcn_dataset.py \
    --data_path /path/to/keypoint_json_files \
    --out_folder /path/to/stgcn_dataset \
    --split train \
    --create_label

# 2. 生成训练数据集
python generate_stgcn_dataset.py \
    --data_path /path/to/keypoint_json_files \
    --out_folder /path/to/stgcn_dataset \
    --split train \
    --num_joints 21 \
    --max_frame 300

# 3. 生成验证数据集（如果有验证集）
python generate_stgcn_dataset.py \
    --data_path /path/to/val_keypoint_json_files \
    --out_folder /path/to/stgcn_dataset \
    --split val \
    --num_joints 21 \
    --max_frame 300
```

**输出文件：**
- `train_data.npy`: 训练数据（形状: N, 3, T, V, M）
- `train_label.pkl`: 训练标签
- `val_data.npy`: 验证数据
- `val_label.pkl`: 验证标签

### 步骤5: 模型训练

使用生成的数据集训练ST-GCN模型：

```bash
cd /home/satuo/code/st-gcn
python main.py recognition -c config/st_gcn/parallel-bars/train.yaml
```

## 目录结构建议

```
project_root/
├── clipped_videos/          # 裁剪后的视频片段
│   ├── jumping/
│   │   ├── jumping_001.mp4
│   │   └── jumping_002.mp4
│   ├── front_swing/
│   │   ├── front_swing_001.mp4
│   │   └── front_swing_002.mp4
│   └── ...
│
├── keypoint_json/           # 关键点JSON文件
│   ├── jumping_001.json
│   ├── jumping_002.json
│   ├── front_swing_001.json
│   └── ...
│
├── stgcn_dataset/           # ST-GCN数据集
│   ├── train_data.npy
│   ├── train_label.pkl
│   ├── val_data.npy
│   └── val_label.pkl
│
└── trained_models/          # 训练好的模型
    └── parallel_bars_stgcn.pth
```

## 注意事项

### 1. 关键点提取工具的实现

目前`extract_keypoints_from_clipped_videos.py`中的关键点提取部分需要根据实际的`demo_mmpose.py`接口进行实现。

**建议的实现方式：**

1. **直接修改demo_mmpose.py**（推荐）
   - 添加一个`extract_keypoints_only`方法
   - 只提取关键点，不进行动作分析和视频保存
   - 直接保存为ST-GCN格式的JSON

2. **或者使用process_video并修改输出**
   - 调用`process_video`方法
   - 从`self.key_angles_dicts`中提取关键点信息（如果有存储）
   - 转换为ST-GCN格式

### 2. 关键点提取的关键代码片段

根据`demo_mmpose.py`的结构，关键点提取的核心逻辑应该是：

```python
# 初始化模型
model = init_model(config, checkpoint, device=device)
meta_info = model.dataset_meta

# 处理每帧
for frame in video_frames:
    # 检测人体和关键点
    pred_instances = process_frame_with_model(frame, model)
    
    # 提取关键点和置信度
    keypoints = pred_instances.keypoints  # shape: (21, 2)
    scores = pred_instances.keypoint_scores  # shape: (21,)
    
    # 转换为ST-GCN格式
    skeleton = {
        'pose': [x1, y1, x2, y2, ...],  # 42个值（21个关键点的x,y）
        'score': [s1, s2, ..., s21]      # 21个置信度值
    }
```

### 3. 数据质量检查

在生成数据集之前，建议检查：

- **关键点检测质量**：确保大部分帧都能检测到有效的关键点
- **动作标签准确性**：确保每个视频的标签正确
- **数据分布平衡**：确保各个动作类别的样本数量相对均衡

### 4. 文件名自动识别规则

批量处理时，工具会根据文件名自动识别动作标签：

- `jumping`: 包含 "jumping", "起跳", "jump"
- `jump_to_leg_sit`: 包含 "jump_to_leg_sit", "跳上成支撑", "leg_sit", "支撑"
- `front_swing`: 包含 "front_swing", "前摆", "前摆上", "front"
- `back_swing`: 包含 "back_swing", "后摆", "后摆上", "back"
- `front_swing_down`: 包含 "front_swing_down", "前摆下", "front_down", "前下"
- `back_swing_down`: 包含 "back_swing_down", "后摆下", "back_down", "后下"

如果无法自动识别，请手动指定`--action_label`参数。

## 快速开始示例

```bash
# 1. 人工裁剪视频（使用视频编辑软件）
# 保存到 clipped_videos/ 目录

# 2. 批量提取关键点
python extract_keypoints_from_clipped_videos.py \
    --video_dir ./clipped_videos \
    --output_dir ./keypoint_json \
    --device cuda:0

# 3. 生成训练数据集
python generate_stgcn_dataset.py \
    --data_path ./keypoint_json \
    --out_folder ./stgcn_dataset \
    --split train \
    --create_label \
    --num_joints 21 \
    --max_frame 300

# 4. 训练模型
cd /home/satuo/code/st-gcn
python main.py recognition -c config/st_gcn/parallel-bars/train.yaml
```

## 常见问题

**Q: 如何处理关键点提取失败的情况？**

A: 工具会自动跳过检测失败的帧，并在最终JSON中标记`has_skeleton: false`。建议检查视频质量和模型配置。

**Q: 视频长度不一致怎么办？**

A: 数据集生成工具会自动处理：较短的视频会填充零，较长的视频会截断到`max_frame`（默认300帧）。

**Q: 如何划分训练集和验证集？**

A: 可以将JSON文件分别放在不同的目录中，或者手动将文件分为train和val两个目录，然后分别运行生成工具。

**Q: 需要多少样本才能训练？**

A: 建议每个动作类别至少100-200个样本。如果数据量不足，可以考虑：
- 数据增强（随机平移、旋转、缩放）
- 使用预训练模型进行微调
- 收集更多数据

