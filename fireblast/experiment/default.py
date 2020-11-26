from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import fireblast.data.datasets as fbdd
import torchvision.transforms as transforms


__all__ = [
  'Experiment',
  'default_trainset_transform',
  'default_testset_transform',
  'default_cars196',
  'default_aircraft',
  'default_cub200',
  'default_dataloader',
]


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
  trainset: Dataset = None
  trainset_loader: DataLoader = None
  validset_loader: DataLoader = None
  testset_loader: DataLoader = None


def default_trainset_transform(x: Experiment, rand_crop_size=(448, 448), rand_hfilp=True):
  if x.trainset_im_transform: return
  _compose = []
  _compose.append(transforms.Resize((int(rand_crop_size[0] * 1.20), int(rand_crop_size[1] * 1.20))))
  _compose.append(transforms.RandomCrop(size=rand_crop_size))
  if rand_hfilp:
    _compose.append(transforms.RandomHorizontalFlip())
  _compose.append(transforms.ToTensor())
  _compose.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))  # mean, std on imagenet
  x.trainset_im_transform = transforms.Compose(_compose)
  x.trainset_gt_transform = None
  return x.trainset_im_transform, x.trainset_gt_transform


def default_testset_transform(x: Experiment, center_crop_size=(448, 448)):
  if x.testset_im_transform: return
  _compose = []
  _compose.append(transforms.Resize((int(center_crop_size[0] * 1.20), int(center_crop_size[1] * 1.20))))
  _compose.append(transforms.CenterCrop(size=center_crop_size))
  _compose.append(transforms.ToTensor())
  _compose.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))  # mean, std on imagenet
  x.testset_im_transform = transforms.Compose(_compose)
  x.testset_gt_transform = None
  return x.testset_im_transform, x.testset_gt_transform


def default_dataloader(dataset, batch_size=16, shuffle=True, num_workers=6, use_for=None):
  if use_for == 'train':
    return DataLoader(dataset, batch_size=12, shuffle=True, num_workers=num_workers)
  elif use_for in ['test', 'valid']:
    return DataLoader(dataset, batch_size=6, shuffle=False, num_workers=num_workers)
  else:
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def default_cars196(x: Experiment, data_loc=None, loader=False):
  x.expt_id = "Cars196"
  x.category_cnt = 196
  default_trainset_transform(x)
  default_testset_transform(x)
  if data_loc:
    _anno = fbdd.get_cars196_anns(root=data_loc, check=True)
  else:
    _anno = fbdd.get_cars196_anns(check=True)
  cars196_data = fbdd.cars196(_anno, transform=x.trainset_im_transform, target_transform=x.trainset_gt_transform)
  x.trainset, x.testset = cars196_data['train'], cars196_data['test']
  if not loader:
    return x.trainset, x.trainset
  x.trainset_loader = default_dataloader(x.trainset, use_for='train')
  x.testset_loader = default_dataloader(x.trainset, use_for='test')
  return x.trainset_loader, x.testset_loader


def default_cub200(x: Experiment, data_loc=None, loader=False):
  x.expt_id = "CUB-200"
  x.category_cnt = 200
  default_trainset_transform(x)
  default_testset_transform(x)
  if data_loc:
    _anno = fbdd.get_cub200_anns(root=data_loc, check=True)
  else:
    _anno = fbdd.get_cub200_anns(check=True)
  _data = fbdd.cub200(_anno, transform=x.trainset_im_transform, target_transform=x.trainset_gt_transform)
  x.trainset, x.trainset = _data['train'], _data['test']
  if not loader:
    return x.trainset, x.trainset
  x.trainset_loader = default_dataloader(x.trainset, use_for='train')
  x.testset_loader = default_dataloader(x.trainset, use_for='test')
  return x.trainset_loader, x.testset_loader


def default_aircraft(x: Experiment, data_loc=None, trainval=True, loader=False):
  x.expt_id = "FGVC-Aircraft"
  x.category_cnt = 102
  default_trainset_transform(x)
  default_testset_transform(x)
  if data_loc:
    _anno = fbdd.get_fgvc_aircraft_anns(root=data_loc, check=True)
  else:
    _anno = fbdd.get_fgvc_aircraft_anns(check=True)
  _data = fbdd.fgvc_aircraft(_anno, transform=x.trainset_im_transform, target_transform=x.trainset_gt_transform)
  x.trainset = _data['trainval'] if trainval else _data['train']
  x.validset, x.testset = _data['val'], _data['test']
  if not loader:
    return x.trainset, x.validset, x.trainset
  x.trainset_loader = default_dataloader(x.trainset, use_for='train')
  x.validset_loader = default_dataloader(x.validset, use_for='valid')
  x.testset_loader = default_dataloader(x.trainset, use_for='test')
  return x.trainset_loader, x.validset_loader, x.testset_loader
