from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from . import Experiment
from .. import get_cub200_anns, cub200
from .. import get_cars196_anns, cars196
from .. import get_fgvc_aircraft_anns, fgvc_aircraft


__all__ = [
  'default_trainset_transform',
  'default_testset_transform',
  'default_cars196',
  'default_aircraft',
  'default_cub200',
  'default_dataloader',
]


def default_trainset_transform(x: Experiment, rand_crop_size=(448, 448), rand_hfilp=True):
  if x.trainset_im_transform:
    return
  _compose = []
  _compose.append(T.Resize(size=(rand_crop_size[0] + 64, rand_crop_size[1] + 64)))
  _compose.append(T.RandomCrop(size=rand_crop_size))
  if rand_hfilp:
    _compose.append(T.RandomHorizontalFlip())
  _compose.append(T.ToTensor())
  _compose.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))  # mean, std on imagenet
  x.trainset_im_transform = T.Compose(_compose)
  x.trainset_gt_transform = None

  return x.trainset_im_transform, x.trainset_gt_transform


def default_testset_transform(x: Experiment, center_crop_size=(448, 448)):
  if x.testset_im_transform:
    return
  _compose = []
  _compose.append(T.Resize(size=(center_crop_size[0] + 64, center_crop_size[1] + 64)))
  _compose.append(T.CenterCrop(size=center_crop_size))
  _compose.append(T.ToTensor())
  _compose.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))  # mean, std on imagenet
  x.testset_im_transform = T.Compose(_compose)
  x.testset_gt_transform = None

  return x.testset_im_transform, x.testset_gt_transform


def default_dataloader(dataset, batch_size=16, shuffle=True, num_workers=10, use_for=None):
  if use_for == 'train':
    return DataLoader(dataset, batch_size=16, shuffle=True, num_workers=num_workers)
  elif use_for in ['test', 'valid']:
    return DataLoader(dataset, batch_size=8, shuffle=False, num_workers=num_workers)
  else:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def default_cars196(x: Experiment, dataset_dir=None, loader=False, batch_size=None):
  x.expt_id = "Car"
  x.category_cnt = 196

  default_trainset_transform(x)
  default_testset_transform(x)

  if dataset_dir:
    _anno = get_cars196_anns(root=dataset_dir, check=True)
  else:
    _anno = get_cars196_anns(check=True)
  cars196_data = cars196(_anno, transform=x.trainset_im_transform, target_transform=x.trainset_gt_transform)
  x.trainset, x.testset = cars196_data['train'], cars196_data['test']
  if not loader:
    return x.trainset, x.testset

  if batch_size and len(batch_size) == 2:
    x.trainset_loader = default_dataloader(x.trainset, batch_size=batch_size[0])
    x.testset_loader = default_dataloader(x.testset, batch_size=batch_size[1], shuffle=False)
  else:
    x.trainset_loader = default_dataloader(x.trainset, use_for='train')
    x.testset_loader = default_dataloader(x.testset, use_for='test')

  return x.trainset_loader, x.testset_loader


def default_cub200(x: Experiment, dataset_dir=None, loader=False, batch_size=None):
  x.expt_id = "CUB"
  x.category_cnt = 200

  default_trainset_transform(x)
  default_testset_transform(x)

  if dataset_dir:
    _anno = get_cub200_anns(root=dataset_dir, check=True)
  else:
    _anno = get_cub200_anns(check=True)
  _data = cub200(_anno, transform=x.trainset_im_transform, target_transform=x.trainset_gt_transform)
  x.trainset, x.testset = _data['train'], _data['test']
  if not loader:
    return x.trainset, x.testset

  if batch_size and len(batch_size) == 2:
    x.trainset_loader = default_dataloader(x.trainset, batch_size=batch_size[0])
    x.testset_loader = default_dataloader(x.testset, batch_size=batch_size[1], shuffle=False)
  else:
    x.trainset_loader = default_dataloader(x.trainset, use_for='train')
    x.testset_loader = default_dataloader(x.testset, use_for='test')
  return x.trainset_loader, x.testset_loader


def default_aircraft(x: Experiment, dataset_dir=None, trainval=True, loader=False, batch_size=None):
  x.expt_id = "Air"
  x.category_cnt = 100

  default_trainset_transform(x)
  default_testset_transform(x)

  if dataset_dir:
    _anno = get_fgvc_aircraft_anns(root=dataset_dir, check=True)
  else:
    _anno = get_fgvc_aircraft_anns(check=True)
  _data = fgvc_aircraft(_anno, transform=x.trainset_im_transform, target_transform=x.trainset_gt_transform)
  x.trainset = _data['trainval'] if trainval else _data['train']
  x.validset, x.testset = _data['val'], _data['test']
  if not loader:
    return x.trainset, x.validset, x.testset

  if batch_size and len(batch_size) == 3:
    x.trainset_loader = default_dataloader(x.trainset, batch_size=batch_size[0])
    x.validset_loader = default_dataloader(x.validset, batch_size=batch_size[1], shuffle=False)
    x.testset_loader = default_dataloader(x.testset, batch_size=batch_size[2], shuffle=False)
  else:
    x.trainset_loader = default_dataloader(x.trainset, use_for='train')
    x.validset_loader = default_dataloader(x.validset, use_for='valid')
    x.testset_loader = default_dataloader(x.testset, use_for='test')

  return x.trainset_loader, x.validset_loader, x.testset_loader

