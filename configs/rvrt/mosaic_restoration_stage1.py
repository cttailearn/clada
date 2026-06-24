from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *

experiment_name = 'mosaic_restoration_rvrt_stage1'
work_dir = f'./experiments/rvrt/{experiment_name}'
save_dir = './experiments/rvrt'

model = dict(
    type='BasicVSR',
    generator=dict(
        type='RVRTNet',
        mid_channels=64,
        num_blocks=8,
        num_heads=8,
        num_points=9,
        group_size=3,
        group_overlap=1,
        mlp_ratio=2.66),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

data_root = 'datasets/mosaic_removal_vid'

train_dataloader = dict(
    num_workers=4,
    batch_size=2,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=data_root + "/train/crop_unscaled_meta",
        num_frame=16,
        degrade=True,
        use_hflip=True,
        repeatable_random=False,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256),
    collate_fn=dict(type='default_collate'))

val_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MosaicVideoDataset',
        metadata_root_dir=data_root + "/val/crop_unscaled_meta",
        num_frame=30,
        degrade=True,
        use_hflip=False,
        repeatable_random=True,
        random_mosaic_params=True,
        filter_watermark=False,
        filter_nudenet_nsfw=False,
        filter_video_quality=False,
        lq_size=256),
    collate_fn=dict(type='default_collate'))

val_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=100_000, val_interval=4000)
val_cfg = dict(type='MultiValLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)))
