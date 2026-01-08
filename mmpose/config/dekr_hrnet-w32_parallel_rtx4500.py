_base_ = ['./default_runtime.py']

# 确保CocoParallelDataset被导入和注册
custom_imports = dict(
    imports=['mmpose.datasets.datasets.body.coco_parallel_dataset'],
    allow_failed_imports=False)

# ============================================================================
# RTX 4060Ti (16GB显存) 优化配置
# ============================================================================

train_cfg = dict(max_epochs=140, val_interval=10)

# 优化器配置 - 使用梯度累积来保持有效batch size
# batch_size=2, accumulative_counts=4 => 有效batch_size=8 (单卡)
# 4卡训练时总有效batch_size=32
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-3),
    accumulative_counts=4  # 梯度累积4次，保持有效batch size
)

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        milestones=[90, 120],
        gamma=0.1,
        by_epoch=True)
]

# 自动学习率缩放 - 基于有效batch size (2*4*4=32)
auto_scale_lr = dict(base_batch_size=32)

default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# Codec配置 - 恢复原始设置以充分利用GPU
codec = dict(
    type='SPR',
    input_size=(512, 512),
    heatmap_size=(128, 128),
    sigma=(4, 2),
    minimal_diagonal_length=32**0.5,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30)  # 恢复为30以支持更多人数检测

model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='DEKRHead',
        in_channels=480,
        num_keypoints=21,
        heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        displacement_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            beta=1 / 9,
            loss_weight=0.002,
        ),
        decoder=codec,
    ),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        nms_dist_thr=0.05,
        shift_heatmap=True,
        align_corners=False))

find_unused_parameters = True

dataset_type = 'CocoParallelDataset'
data_mode = 'bottomup'
# 数据路径 - 使用data盘的绝对路径
# 实际数据集路径：/data/lxc/datasets/coco_paralel
data_root = '/data/lxc/datasets/coco_paralel'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize',
        input_size=codec['input_size'],
        size_factor=32,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

# ============================================================================
# 数据加载器配置 - 针对16GB显存优化（RTX 4060Ti）
# ============================================================================
train_dataloader = dict(
    batch_size=2,  # 16GB显存下进一步减小batch，避免512x512底部模型OOM
    num_workers=4,  # 减少数据加载线程数以降低内存占用
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train_parallel.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val_parallel.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/person_keypoints_val_parallel.json',
    nms_mode='none',
    score_mode='keypoint',
)

test_evaluator = val_evaluator

