# dataset settings
# data_source = 'Climate'
data_source = 'ImageNet'
dataset_type = 'MultiViewDataset' # TODO: need to modify this?
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip', p=0.2),
    dict(type='RandomVerticalFlip', p=0.2),
    # dict(type='RandomRotation', p=0.2, degrees=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #             dict(type='RandomHorizontalFlip'),
    #             dict(type='RandomVerticalFlip'),
    #             # dict(type='RandomRotation', degrees=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]),
    #     ],
    #     p=0.2
    # ),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=32,  # total 32*8
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))