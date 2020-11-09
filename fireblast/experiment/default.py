from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import fireblast.data.datasets as fbd
import torchvision.transforms as transforms


__all__ = [
  'FireblastExperiment',
  'default_transform',
  'default_cars196',
  'default_aircraft',
  'default_cub200',
  'default_dataloader',
]


@dataclass
class FireblastExperiment:
  Id: str = None
  ImageTransform: object = None
  TargetTransform: object = None
  Trainset: Dataset = None
  Testset: Dataset = None
  TrainsetLoader: DataLoader = None
  TestsetLoader: DataLoader = None


def default_transform(x: FireblastExperiment, rand_crop_size=(384, 384), rand_hfilp=True):
  if x.ImageTransform: return
  compose = []
  compose.append(transforms.Resize((int(rand_crop_size[0] * 1.20), int(rand_crop_size[1] * 1.20))))
  compose.append(transforms.RandomCrop(size=rand_crop_size))
  if rand_hfilp:
    compose.append(transforms.RandomHorizontalFlip())
  compose.append(transforms.ToTensor())
  compose.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))  # mean, std on imagenet
  x.ImageTransform = transforms.Compose(compose)
  x.TargetTransform = transforms.ToTensor()
  return x.ImageTransform, x.TargetTransform


def default_dataloader(dataset, batch_size=16, shuffle=True, num_workers=4):
  return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def default_cars196(x: FireblastExperiment, loader=False):
  x.Id = "Cars196"
  default_transform(x)
  cars196_anno = fbd.get_cars196_anns(root=r"/home/mlss/data/cars196", check=True)
  cars196_data = fbd.cars196(cars196_anno, transform=x.ImageTransform, target_transform=x.TargetTransform)
  x.Trainset, x.Testset = cars196_data['train'], cars196_data['test']
  if not loader:
    return x.Trainset, x.Testset
  x.TrainsetLoader = default_dataloader(x.Trainset)
  x.TestsetLoader = default_dataloader(x.Testset)
  return x.TrainsetLoader, x.TestsetLoader


def default_cub200(x: FireblastExperiment, loader=False):
  x.Id = "CUB-200"
  default_transform(x)
  cub200_anno = fbd.get_cub200_anns(root=r"/home/mlss/data/CUB_200_2011", check=True)
  cub200_data = fbd.cub200(cub200_anno, transform=x.ImageTransform, target_transform=x.TargetTransform)
  x.Trainset, x.Testset = cub200_data['train'], cub200_data['test']
  if not loader:
    return x.Trainset, x.Testset
  x.TrainsetLoader = default_dataloader(x.Trainset)
  x.TestsetLoader = default_dataloader(x.Testset)
  return x.TrainsetLoader, x.TestsetLoader


def default_aircraft(x: FireblastExperiment, loader=False):
  x.Id = "FGVC-Aircraft"
  default_transform(x)
  aircraft_anno = fbd.get_fgvc_aircraft_anns(root=r"/home/mlss/data/fgvc-aircraft-2013b", check=True)
  aircraft_data = fbd.fgvc_aircraft(aircraft_anno, transform=x.ImageTransform, target_transform=x.TargetTransform)
  x.Trainset, x.Testset = aircraft_data['train'], aircraft_data['test']
  if not loader:
    return x.Trainset, x.Testset
  x.TrainsetLoader = default_dataloader(x.Trainset)
  x.TestsetLoader = default_dataloader(x.Testset)
  return x.TrainsetLoader, x.TestsetLoader
