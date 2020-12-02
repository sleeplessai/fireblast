import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.io as sio
import logging


__all__ = [
  'get_cub200_anns', 'cub200',
  'get_fgvc_aircraft_anns', 'fgvc_aircraft',
  'get_cars196_anns', 'cars196'
]

__datasets__ = ['CUB_200_2011', 'fgvc-aircraft-2013b', 'cars196']


class FireblastDataset(Dataset):
  """FireblastDataset is a PyTorch Custom Dataset implementation for inner use.
  For initialization, image dictionary and sample list necessarily specify.
  The sample list augment is a tuple(2-element list) list;
  each tuple contains a pair of image file name string and image category index integer.
  Transform and target_transform remain same with Torchvision.transformers for Dataset.

  """

  def __init__(self, image_dict: Path, samples, transform=None, target_transform=None):
    self.image_dict = image_dict
    self.samples = samples
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    image = Image.open((self.image_dict / self.samples[index][0]).with_suffix('.jpg')).convert('RGB')
    label = self.samples[index][1]
    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)

    return image, label

  def __len__(self):
    return len(self.samples)


def get_cub200_anns(root: str = './data/CUB_200_2011', check: bool = False, **kwargs) -> dict:
  """Get CUB-200-2011 dataset annotations as dict given root path.
    Dataset URL: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

  Args:
    root (str): String of root directory path for CUB-200-2011 dataset.
    check (bool): If True, checks annotation files existence and fix dict results.

  Returns:
    dict (dict): CUB-200-2011 dataset annotation dict object.

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
  if check: _check_anns(name='Caltech-UCSD_Birds-200-2011', anns=anns_dict)

  return anns_dict


def cub200(anns: dict, transform=None, target_transform=None, **kwargs) -> dict:
  """Create PyTorch Dataset instances for CUB-200-2011 dataset.
    Dataset URL: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

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


def get_fgvc_aircraft_anns(root: str = './data/fgvc-aircraft-2013b', check: bool = False, **kwargs) -> dict:
  """Get FGVC-Aircraft dataset annotations as dict given root path.
    Dataset URL: http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

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


def fgvc_aircraft(anns: dict, transform=None, target_transform=None, **kwargs) -> dict:
  """Create PyTorch Dataset instances for FGVC-Aircraft dataset.
    Dataset URL: http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

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

  varts = [l.rstrip('\n') for l in open(anns['variants'], 'r').readlines()]
  varts_idx = {}
  for i, vart in enumerate(varts):
    varts_idx[vart] = i

  vart_imgs_train = _ann_to_list(ann_file=anns['images_variant_train'], varts_idx=varts_idx)
  vart_imgs_val = _ann_to_list(ann_file=anns['images_variant_val'], varts_idx=varts_idx)
  vart_imgs_trainval = _ann_to_list(ann_file=anns['images_variant_trainval'], varts_idx=varts_idx)
  vart_imgs_test = _ann_to_list(ann_file=anns['images_variant_test'], varts_idx=varts_idx)

  train = FireblastDataset(anns['image_folder'], vart_imgs_train, transform=transform, target_transform=target_transform)
  valid = FireblastDataset(anns['image_folder'], vart_imgs_val, transform=transform, target_transform=target_transform)
  trainval = FireblastDataset(anns['image_folder'], vart_imgs_trainval, transform=transform, target_transform=target_transform)
  test = FireblastDataset(anns['image_folder'], vart_imgs_test, transform=transform, target_transform=target_transform)

  return {
    'train': train,
    'val': valid,
    'trainval': trainval,
    'test': test
  }


def get_cars196_anns(root: str = './data/cars196', check: bool = False, **kwargs) -> dict:
  """Get Stanford Cars dataset annotations as dict given root path.
    Dataset URL: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

  Args:
    root (str): String of root directory path for Stanford Cars dataset.
    check (bool): If True, checks annotation files existence and fix dict results.

  Returns:
    dict (dict): Stanford Cars dataset annotation dict object.

  """
  rt_path = Path(root)
  devkit_path = rt_path / 'devkit'
  anns_dict = {
    'train_image_folder': rt_path / 'cars_train',
    'test_image_folder': rt_path / 'cars_test',
    'cars_meta': devkit_path / 'cars_meta.mat',
    'cars_train_annos': devkit_path / 'cars_train_annos.mat',
    'cars_test_annos_withlabels': devkit_path / 'cars_test_annos_withlabels.mat',
    'root': rt_path
  }
  if check: _check_anns(name='Stanford_Cars', anns=anns_dict)

  return anns_dict


def cars196(anns: dict, transform=None, target_transform=None, **kwargs) -> dict:
  """Create PyTorch Dataset instances for Stanford Cars dataset.
    Dataset URL: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

  Args:
    anns (dict): Annotation dict returned by get_cars196_anns(...).
    transform: Torchvision transform for images.
    target_transform: Torchvision transform for targets.

  Returns:
    dict (dict): A dict object contains train and test PyTorch Dataset instances.

  """
  assert anns['cars_meta']
  assert anns['cars_train_annos']
  assert anns['cars_test_annos_withlabels']
  assert anns['train_image_folder']
  assert anns['test_image_folder']

  cars_meta = sio.loadmat(anns['cars_meta'], squeeze_me=True)['class_names'].tolist()
  cars_idx = {}
  for i, car in enumerate(cars_meta):
    cars_idx[car] = i + 1
  # annos: xxyy_bbox, class, fname
  cars_train_li = sio.loadmat(anns['cars_train_annos'], squeeze_me=True)['annotations'].tolist()
  cars_train = [[str(x[-1]), int(x[-2]) - 1] for x in cars_train_li]

  cars_test_li = sio.loadmat(anns['cars_test_annos_withlabels'], squeeze_me=True)['annotations'].tolist()
  cars_test = [[str(x[-1]), int(x[-2]) - 1] for x in cars_test_li]

  train = FireblastDataset(anns['train_image_folder'], cars_train, transform=transform, target_transform=target_transform)
  test = FireblastDataset(anns['test_image_folder'], cars_test, transform=transform, target_transform=target_transform)

  return {
    'train': train,
    'test': test
  }


def _check_anns(name, anns):
  logging.warning(f'Checking {name} annotation existence')
  for k, v in anns.items():
    if not v.exists():
      anns[k] = None
      logging.warning(f'{name}.{k} missed.')


def _ann_to_list(ann_file, varts_idx):
  ann_str_list = [l.rstrip('\n') for l in open(ann_file, 'r').readlines()]
  sample_list = []
  for s in ann_str_list:
    t = s.find(' ')
    vi = [s[:t], varts_idx[s[t + 1:]]]
    sample_list.append(vi)
  return sample_list

