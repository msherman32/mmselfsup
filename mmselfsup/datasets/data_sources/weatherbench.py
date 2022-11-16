import mmcv
import numpy as np

from ..builder import DATASOURCES
from .base import BaseDataSource


@DATASOURCES.register_module()
class WeatherbenchDataSource(BaseDataSource):

    def load_annotations(self):

        url = 'https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2F2m_temperature&files=2m_temperature_5.625deg.zip'
        download_and_extract_archive(
          url,
          self.data_prefix,
          f
      
      
        data_infos = []
        # writing your code here.
        return data_infos
