"""SegFormer-B5 student configuration for ADE20K."""
import os

default_scope = 'mmseg'
data_root = os.path.join(os.environ['S2RL_DATA_ROOT'], 'ADE20K')
dataset_type = 'ADE20KDataset'
crop_size = (512, 512)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
dataset_options = dict()

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer', in_channels=3, embed_dims=64,
        num_stages=4, num_layers=[3, 6, 40, 3], num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='SegformerHead', in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3], channels=256, dropout_ratio=0.1,
        num_classes=150, norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        # scripts/mmseg/ade20k.sh supplies the selected subset paths.
        data_prefix=dict(img_path='/path/to/selected/images',
                         seg_map_path='/path/to/selected/annotations'),
        pipeline=train_pipeline, **dataset_options))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
        pipeline=test_pipeline, **dataset_options))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys=dict(
        pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0),
        head=dict(lr_mult=10.0))))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=2),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=True, begin=2, end=200),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=25)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=2, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(by_epoch=False)
log_level = 'INFO'
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
load_from = None
resume = False
