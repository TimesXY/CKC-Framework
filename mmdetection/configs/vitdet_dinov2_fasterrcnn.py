# -*- coding: utf-8 -*-

# 1) 让 mmdet 能注册 mmpretrain 的 ViT，并引入 ViTDet 项目里的 SimpleFPN
custom_imports = dict(
    imports=[
        'mmpretrain.models',
        'projects.ViTDet.vitdet',
        'unwrap_neck',  # ✅ 新增
    ],
    allow_failed_imports=False
)

default_scope = 'mmdet'

# ===== 数据集 =====
dataset_type = 'CocoDataset'
data_root = ''

classes = ('foreground',)
metainfo = dict(classes=classes)

train_ann = r'E:\MyDataSet\Breast_USV_P0\annotations\instances_train.json'
val_ann = r'E:\MyDataSet\Breast_USV_P0\annotations\instances_test.json'
train_img = r'E:\MyDataSet\Breast_USV_P0\train'
val_img = r'E:\MyDataSet\Breast_USV_P0\test'

# ===== 训练分辨率（先用 512，跑通后再尝试更大）=====
image_size = (196, 196)

# ===== DINOv2 权重（MMPretrain 官方下载链接）=====
dinov2_ckpt = 'https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth'

# ===== pipeline（尽量保持简单）=====
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_img),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=val_img),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=val_ann, metric=['bbox'])
test_evaluator = val_evaluator

# ===== 模型：Faster R-CNN + DINOv2 ViT + SimpleFPN =====
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=32
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=image_size[0],  # “最常见输入尺寸”，动态输入仍可用
        patch_size=14,
        out_indices=-1,
        out_type='featmap',  # 输出 (B,C,H,W) 供 neck 使用  :contentReference[oaicite:4]{index=4}
        init_cfg=dict(
            type='Pretrained',
            checkpoint=dinov2_ckpt,
            prefix='backbone.'  # 常见：mmpretrain checkpoint key 带 backbone. 前缀 :contentReference[oaicite:5]{index=5}
        )
    ),
    neck=dict(
        type='UnwrapTupleNeck',
        neck=dict(
            type='SimpleFPN',
            backbone_channel=768,
            in_channels=[192, 384, 768, 768],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)
        )
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(type='AnchorGenerator', ratios=[0.5, 1.0, 2.0], scales=[8], strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[1., 1., 1., 1.]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=len(classes),
            bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0., 0., 0., 0.], target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3,
                          match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
                          match_low_quality=False, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)
    )
)

# ===== 训练策略（先用朴素 AdamW 跑通）=====
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=1),
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP_50',
        rule='greater',
        max_keep_ckpts=3,
        save_last=True
    )
)

work_dir = './work_dirs/vitdet_dinov2_fasterrcnn'
