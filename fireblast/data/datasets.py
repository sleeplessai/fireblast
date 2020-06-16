import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path

import logging

# dev use
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)
#

__all__ = ['get_cub200_anns', 'cub200']


datasets_in_stock = [
  'CUB_200_2011',
  'fgvc-aircraft-2013b',
  'cars196'
]


def get_cub200_anns(root: str = './CUB_200_2011', check: bool = False, **kwargs) -> dict:
  r"""Get CUB200-2011 dataset annotations as dict given root path.
  
  Args:
    root (str): String of root directory path for CUB200-2011 dataset.
    check (bool): If True, checks annotation files existence and fix dict results.

  Returns:
    dict (dict): CUB200-2011 dataset annotation dict object.

  """
  rt_path = Path(root)

  anns_dict = {
    'images': rt_path / 'images.txt',
    'image_folder': rt_path / 'images',
    'classes': rt_path / 'classes.txt',
    'image_class_labels': rt_path / 'image_class_labels.txt',
    'train_test_split': rt_path / 'train_test_split.txt',
    # 'bounding_boxes': rt_path / 'bounding_boxes.txt',
    'root': rt_path
  }

  if check:
    logging.info('Check CUB200 annotation existence')
    for k, v in anns_dict.items():
      if not v.exists():
        anns_dict[k] = None
        logging.warning(f'CUB200.{k} not exists.')

  return anns_dict


def cub200(anns: dict, transform=None, target_transform=None, **kwargs) -> dict:
  r"""Create PyTorch Dataset instances for CUB200_2011 dataset.

  Args:
    anns (dict): Annotation dict returned by get_cuba200_annotation_dict(...).
    transform: Torchvision transform for images.
    target_transform: Torchvision transform for targets.
  
  Returns:
    dict (dict): A dict object contains traintest, train and test PyTorch Dataset instances.

  """
  assert anns['root']
  assert anns['image_folder']
  assert anns['images']
  assert anns['train_test_split']

  traintest = ImageFolder(anns['image_folder'], transform=transform, target_transform=target_transform)

  # sorted(img_list) has same order as traintest.imgs
  img_list = open(anns['images'], 'r').readlines()
  img_list = [anns['image_folder'] / Path(x.split(' ')[1][:-1]) for x in img_list]

  # 1/0 in train_test_split.txt means is_training or not
  is_train = open(anns['train_test_split'], 'r').readlines()
  is_train = [bool(int(x.split(' ')[1][:-1])) for x in is_train]

  img_tt = sorted(list(zip(img_list, is_train)), key=lambda x: x[0])
  img_tt = np.array(img_tt)[:, 1]
  # train/test indices
  train_idx = np.where(img_tt)[0]
  test_idx = np.where(np.logical_not(img_tt))[0]
  
  train = Subset(traintest, indices=train_idx)
  test = Subset(traintest, indices=test_idx)

  return {
    'traintest': traintest,
    'train': train,
    'test': test
  }
