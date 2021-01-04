from dataclasses import dataclass
import os
from typing import Union, List

from torch.utils.data import Dataset, DataLoader


@dataclass
class Experiment:
  expt_id: str = None
  trainset_im_transform: object = None
  trainset_gt_transform: object = None
  testset_im_transform: object = None
  testset_gt_transform: object = None
  category_cnt: int = 0
  trainset: Dataset = None
  validset: Dataset = None
  testset: Dataset = None
  trainset_loader: DataLoader = None
  validset_loader: DataLoader = None
  testset_loader: DataLoader = None


def set_cuda_visible_devices(devices: Union[int, List[int]]):
  if isinstance(devices, int):
    assert devices >= 0
    devices = str(devices)
  elif isinstance(devices, list):
    devices = [str(i) for i in sorted(devices) if i >= 0]
    devices = ",".join(devices)

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = devices
