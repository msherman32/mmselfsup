model = dict(
    type='Classification',
    backbone=dict(
        type='UNet',
        in_channels=3),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=128,
        num_classes=1000))
