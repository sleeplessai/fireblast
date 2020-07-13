import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from .utils import _check_anns, _ann_to_list


__all__ = [
  'get_cub200_anns', 'cub200',
  'get_fgvc_aircraft_anns', 'fgvc_aircraft'
]

__datasets__ = ['CUB_200_2011', 'fgvc-aircraft-2013b', 'cars196']


def get_cub200_anns(root: str = './CUB_200_2011', check: bool = False, **kwargs) -> dict:
  """Get CUB200-2011 dataset annotations as dict given root path.
  
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
    'bounding_boxes': rt_path / 'bounding_boxes.txt',
    'root': rt_path
  }
  if check: _check_anns(name='CUB200', anns=anns_dict)

  return anns_dict


def cub200(anns: dict, transform=None, target_transform=None, **kwargs) -> dict:
  """Create PyTorch Dataset instances for CUB200_2011 dataset.

  Args:
    anns (dict): Annotation dict returned by get_cub200_anns(...).
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


def get_fgvc_aircraft_anns(root: str = './fgvc-aircraft-2013b', check: bool = False, **kwargs) -> dict:
  """Get FGVC-Aircraft dataset annotations as dict given root path.
  
  Args:
    root (str): String of root directory path for FGVC-Aircraft dataset.
    check (bool): If True, checks annotation files existence and fix dict results.

  Returns:
    dict (dict): FGVC-Aircraft dataset annotation dict object.

  """
  rt_path = Path(root)
  rt_dt_path = Path(root) / 'data'
  anns_dict = {
    'image_folder': rt_dt_path / 'images',
    'images_box': rt_dt_path / 'images_box.txt',
    'images_variant_train': rt_dt_path / 'images_variant_train.txt',
    'images_variant_val': rt_dt_path / 'images_variant_val.txt',
    'images_variant_trainval': rt_dt_path / 'images_variant_trainval.txt',
    'images_variant_test': rt_dt_path / 'images_variant_test.txt',
    'variants': rt_dt_path / 'variants.txt',
    'root': rt_path
  }
  if check: _check_anns(name='FGVC-Aircraft', anns=anns_dict)

  return anns_dict


class FGVCAircraft(Dataset):
  def __init__(self, image_dict: Path, train_ann, transform=None, target_transform=None):
    self.image_dict = image_dict
    self.train_ann = train_ann
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    image = Image.open((self.image_dict / self.train_ann[index][0]).with_suffix('.jpg')).convert('RGB')
    label = self.train_ann[index][1]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)

    return image, label
    
  def __len__(self):
    return len(self.train_ann)


def fgvc_aircraft(anns: dict, transform=None, target_transform=None, **kwargs) -> dict:
  """Create PyTorch Dataset instances for FGVC-Aircraft dataset.

  Args:
    anns (dict): Annotation dict returned by get_fgvc_aircraft_anns(...).
    transform: Torchvision transform for images.
    target_transform: Torchvision transform for targets.
  
  Returns:
    dict (dict): A dict object contains train, val, trainval and test PyTorch Dataset instances.

  """
  assert anns['variants']
  assert anns['image_folder']
  assert anns['images_variant_train']
  assert anns['images_variant_val']
  assert anns['images_variant_trainval']
  assert anns['images_variant_test']

  varts = [l.rstrip() for l in open(anns['variants'], 'r').readlines()]
  varts_idx = {}
  for i, vart in enumerate(varts):
    varts_idx[vart] = i
  
  vart_imgs_train = _ann_to_list(ann_file=anns['images_variant_train'], varts_idx=varts_idx)
  vart_imgs_val = _ann_to_list(ann_file=anns['images_variant_val'], varts_idx=varts_idx)
  vart_imgs_trainval = _ann_to_list(ann_file=anns['images_variant_trainval'], varts_idx=varts_idx)
  vart_imgs_test = _ann_to_list(ann_file=anns['images_variant_test'], varts_idx=varts_idx)

  train = FGVCAircraft(anns['image_folder'], vart_imgs_train, transform=transform, target_transform=target_transform)
  valid = FGVCAircraft(anns['image_folder'], vart_imgs_val, transform=transform, target_transform=target_transform)
  trainval = FGVCAircraft(anns['image_folder'], vart_imgs_trainval, transform=transform, target_transform=target_transform)
  test = FGVCAircraft(anns['image_folder'], vart_imgs_test, transform=transform, target_transform=target_transform)

  return {
    'train': train, 
    'val': valid,
    'trainval': trainval,
    'test': test
  }

