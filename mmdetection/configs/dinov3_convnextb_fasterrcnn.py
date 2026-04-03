# configs/my_dinov3_convnextb_fasterrcnn.py
_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=['mmpretrain.models'],
    allow_failed_imports=False
)
default_scope = 'mmdet'

# 你的类别
metainfo = dict(classes=('foreground',))

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        bgr_to_rgb=True,
        # 用 ImageNet mean/std（timm / DINOv3 常规配置也是这套）
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=32,
    ),
    backbone=dict(
        _delete_=True,
        type='mmpretrain.TIMMBackbone',
        model_name='convnext_base.dinov3_lvd1689m',  # ✅ DINOv3 ConvNeXt-Base
        pretrained=True,  # ✅ 自动下载/缓存权重
        features_only=True,  # ✅ 输出 4 个stage特征图
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[128, 256, 512, 1024],  # ✅ ConvNeXt-Base features_only 输出通道:contentReference[oaicite:2]{index=2}
        out_channels=256,
        num_outs=5,
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
    ),
)

# pipeline：
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# dataloader 里的 dataset 路径你可以在 python 脚本里覆盖
train_dataloader = dict(dataset=dict(metainfo=metainfo, pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(metainfo=metainfo, pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', metric=['bbox'])
test_evaluator = val_evaluator
