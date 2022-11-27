# model settings
model = dict(
    type='SimCLR',
    backbone=dict(
        type='UNet',
        in_channels=3,
        norm_cfg=dict(type='BN'),
        ),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=1024,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.1))
