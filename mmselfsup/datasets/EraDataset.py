import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose, ToPILImage

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
# from .utils import to_numpy
# from .modules import DataModule
from .module import DataModule
# import DataModule
from ..utils.datetime import Year, Days, Hours
# from PIL import Image
# import numpy as np


@DATASETS.register_module()
class ERA5Dataset(BaseDataset):

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        # writing your code here
        
        # self.data_source = build_datasource(data_source)

        data_module = DataModule(
            dataset = "ERA5",
            task = "forecasting",
            # root_dir = "era5/5.625",
            root_dir = '/content/drive/MyDrive/Climate/.climate_tutorial/data/weatherbench/era5/5.625/',

            in_vars = ["2m_temperature", "total_precipitation"],
            out_vars = ["2m_temperature"],

            train_start_year = Year(2000),
            val_start_year = Year(2015),
            test_start_year = Year(2017),
            end_year = Year(2018),


            pred_range = Days(3),
            subsample = Hours(6),
            specify_range = True,

            min_lat = 26,
            max_lat = 50,
            min_lon = 230,
            max_lon = 310,
            val_lat_start = 35,
            val_lon_start = 0,
            test_lat_start = 35,
            test_lon_start = 0,

            batch_size = 128,
            num_workers = 1
        )
        self.data_source = data_module.train_dataset.inp_data
        # self.CLASSES = self.data_source.CLASSES

        
        # self.pipelines = pipelines
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipelines]
        self.pipeline = Compose(pipeline)
        # self.pipelines = []
        # for pipe in pipelines:
        #     pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
        #     self.pipelines.append(pipeline)
        self.prefetch = prefetch

    def __getitem__(self, idx):
        
        data = self.data_source[idx]
        
        # img = img.astype(np.uint8)
        # img = np.asarray(img)
        # img = Image.fromarray(img)

        data = torch.from_numpy(data)
        transform = ToPILImage()
        img = transform(data)


        img = self.pipeline(img)

        # img = self.data_source.get_img(idx)
        # clustering_label = self.clustering_labels[idx]
        # if self.prefetch:
        #     img = torch.from_numpy(to_numpy(img))
        # return dict(img=img, pseudo_label=clustering_label, idx=idx)
        # return dict(img=data, idx=idx)
        return dict(img=img, idx=idx)

    def evaluate(self, results, logger=None):
        return NotImplemented
