from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import fireblast.data.datasets as fbd
import torchvision.transforms as transforms


__all__ = [
  'FireblastExperiment',
  'default_trainset_transform',
  'default_testset_transform',
  'default_cars196',
  'default_aircraft',
  'default_cub200',
  'default_dataloader',
]


@dataclass
class FireblastExperiment:
  Id: str = None
  TrainsetImageTransform: object = None
  TrainsetTargetTransform: object = None
  TestsetImageTransform: object = None
  TestsetTargetTransform: object = None
  Trainset: Dataset = None
  Testset: Dataset = None
  TrainsetLoader: DataLoader = None
  TestsetLoader: DataLoader = None


def default_trainset_transform(x: FireblastExperiment, rand_crop_size=(384, 384), rand_hfilp=True):
  if x.TrainsetImageTransform: return
  _compose = []
  _compose.append(transforms.Resize((int(rand_crop_size[0] * 1.20), int(rand_crop_size[1] * 1.20))))
  _compose.append(transforms.RandomCrop(size=rand_crop_size))
  if rand_hfilp:
    _compose.append(transforms.RandomHorizontalFlip())
  _compose.append(transforms.ToTensor())
  _compose.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))  # mean, std on imagenet
  x.TrainsetImageTransform = transforms.Compose(_compose)
  x.TrainsetTargetTransform = None
  return x.TrainsetImageTransform, x.TrainsetTargetTransform


def default_testset_transform(x: FireblastExperiment, center_crop_size=(384, 384)):
  if x.TestsetImageTransform: return
  _compose = []
  _compose.append(transforms.Resize((int(center_crop_size[0] * 1.20), int(center_crop_size[1] * 1.20))))
  _compose.append(transforms.CenterCrop(size=center_crop_size))
  _compose.append(transforms.ToTensor())
  _compose.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))  # mean, std on imagenet
  x.TestsetImageTransform = transforms.Compose(_compose)
  x.TestsetTargetTransform = None
  return x.TestsetImageTransform, x.TestsetTargetTransform


def default_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
  return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def default_cars196(x: FireblastExperiment, loader=False):
  x.Id = "Cars196"
  default_trainset_transform(x)
  default_testset_transform(x)
  cars196_anno = fbd.get_cars196_anns(root=r"/home/mlss/data/cars196", check=True)
  cars196_data = fbd.cars196(cars196_anno, transform=x.TrainsetImageTransform, target_transform=x.TrainsetTargetTransform)
  x.Trainset, x.Testset = cars196_data['train'], cars196_data['test']
  if not loader:
    return x.Trainset, x.Testset
  x.TrainsetLoader = default_dataloader(x.Trainset)
  x.TestsetLoader = default_dataloader(x.Testset)
  return x.TrainsetLoader, x.TestsetLoader


def default_cub200(x: FireblastExperiment, loader=False):
  x.Id = "CUB-200"
  default_trainset_transform(x)
  default_testset_transform(x)
  cub200_anno = fbd.get_cub200_anns(root=r"/home/mlss/data/CUB_200_2011", check=True)
  cub200_data = fbd.cub200(cub200_anno, transform=x.TrainsetImageTransform, target_transform=x.TrainsetTargetTransform)
  x.Trainset, x.Testset = cub200_data['train'], cub200_data['test']
  if not loader:
    return x.Trainset, x.Testset
  x.TrainsetLoader = default_dataloader(x.Trainset)
  x.TestsetLoader = default_dataloader(x.Testset)
  return x.TrainsetLoader, x.TestsetLoader


def default_aircraft(x: FireblastExperiment, loader=False):
  x.Id = "FGVC-Aircraft"
  default_trainset_transform(x)
  default_testset_transform(x)
  aircraft_anno = fbd.get_fgvc_aircraft_anns(root=r"/home/mlss/data/fgvc-aircraft-2013b", check=True)
  aircraft_data = fbd.fgvc_aircraft(aircraft_anno, transform=x.TrainsetImageTransform, target_transform=x.TrainsetTargetTransform)
  x.Trainset, x.Testset = aircraft_data['train'], aircraft_data['test']
  if not loader:
    return x.Trainset, x.Testset
  x.TrainsetLoader = default_dataloader(x.Trainset)
  x.TestsetLoader = default_dataloader(x.Testset)
  return x.TrainsetLoader, x.TestsetLoader
