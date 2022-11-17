import mmcv
import numpy as np

from ..builder import DATASOURCES
from .base import BaseDataSource


@DATASOURCES.register_module()
class EraDataSource(BaseDataSource):

    def load_annotations(self):

        data_infos = []
        return data_infos
