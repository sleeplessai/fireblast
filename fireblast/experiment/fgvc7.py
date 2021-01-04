import logging

import torchvision.transforms as T

from . import Experiment, default_dataloader
from .. import get_cub200_anns, cub200
from .. import get_cars196_anns, cars196
from .. import get_fgvc_aircraft_anns, fgvc_aircraft


__all__ = [
  'fgvc7_transform_strategy',
  'fgvc7_aug_cub200',
  'fgvc7_aug_cars196',
  'fgvc7_aug_aircraft'
]


def fgvc7_transform_strategy(image_size):
  logging.warning('Using CVPR2020-FGVC7 champion data augmentation')

  training_t = T.Compose([
    T.Resize((image_size[0], image_size[1])),
    T.RandomChoice([T.ColorJitter(brightness=(0.9, 1.1)), T.ColorJitter(contrast=(0.9, 1.1))]),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=0.1)], p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=(-20., 20.), translate=(0., 0.2), scale=(0.8, 1.2)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  ])

  testing_t = T.Compose([
    T.Resize((image_size[0], image_size[1])),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  ])

  return {
    'training': training_t,
    'testing': testing_t
  }


def fgvc7_aug_cub200(x: Experiment, dataset_dir=None, loader=False, batch_size=None):
  x.expt_id = "CUB"
  x.category_cnt = 200

  t_pair = fgvc7_transform_strategy((448, 448))
  x.trainset_im_transform = t_pair['training']
  x.testset_im_transform = t_pair['testing']

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


def fgvc7_aug_cars196(x: Experiment, dataset_dir=None, loader=False, batch_size=None):
  x.expt_id = "Car"
  x.category_cnt = 196

  t_pair = fgvc7_transform_strategy((448, 448))
  x.trainset_im_transform = t_pair['training']
  x.testset_im_transform = t_pair['testing']

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


def fgvc7_aug_aircraft(x: Experiment, dataset_dir=None, trainval=True, loader=False, batch_size=None):
  x.expt_id = "Air"
  x.category_cnt = 100

  t_pair = fgvc7_transform_strategy((448, 448))
  x.trainset_im_transform = t_pair['training']
  x.testset_im_transform = t_pair['testing']

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

