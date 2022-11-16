import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy


@DATASETS.register_module()
class NewDataset(BaseDataset):

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        # writing your code here
    def __getitem__(self, idx):
        # writing your code here
        return dict(img=img)

    def evaluate(self, results, logger=None):
        return NotImplemented
