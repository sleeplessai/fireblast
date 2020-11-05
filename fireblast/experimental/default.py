from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import fireblast.data.datasets as fbd


__all__ = ['FireblastExperimental', 'set_experimental']


@dataclass
class FireblastExperimental:
  ImageTransform: object = None
  TargetTransform: object = None
  Cars196Trainset: Dataset = None
  Cars196Testset: Dataset = None
  Cars196TrainsetLoader: DataLoader = None
  Cars196TestsetLoader: DataLoader = None
  Cub200Trainset: Dataset = None
  Cub200Testset: Dataset = None
  Cub200TrainsetLoader: DataLoader = None
  Cub200TestsetLoader: DataLoader = None
  AircraftTrainset: Dataset = None
  AircraftTestset: Dataset = None
  AircraftTrainsetLoader: DataLoader = None
  AircraftTestsetLoader: DataLoader = None

  def default_transform(self):
    if self.ImageTransform: return
    import torchvision.transforms as transforms
    self.ImageTransform = transforms.Compose([
      transforms.Resize((550, 550)),
      transforms.RandomCrop((448, 448)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])   # todo: need calc mean and std
    ])
    self.TargetTransform = None
    return self.ImageTransform, self.TargetTransform

  def default_cars196(self):
    self.default_transform()
    cars196_anno = fbd.get_cars196_anns(root=r"/home/mlss/data/cars196", check=True)
    cars196_data = fbd.cars196(cars196_anno, transform=self.ImageTransform, target_transform=self.TargetTransform)
    self.Cars196Trainset, self.Cars196Testset = cars196_data['train'], cars196_data['test']
    return self.Cars196Trainset, self.Cars196Testset

  def default_cub200(self):
    self.default_transform()
    cub200_anno = fbd.get_cub200_anns(root=r"/home/mlss/data/CUB_200_2011", check=True)
    cub200_data = fbd.cub200(cub200_anno, transform=self.ImageTransform, target_transform=self.TargetTransform)
    self.Cub200Trainset, self.Cub200Testset = cub200_data['train'], cub200_data['test']
    return self.Cub200Trainset, self.Cub200Testset

  def default_aircraft(self):
    self.default_transform()
    aircraft_anno = fbd.get_fgvc_aircraft_anns(root=r"/home/mlss/data/fgvc-aircraft-2013b", check=True)
    aircraft_data = fbd.fgvc_aircraft(aircraft_anno, transform=self.ImageTransform, target_transform=self.TargetTransform)
    self.AircraftTrainset, self.AircraftTestset = aircraft_data['train'], aircraft_data['test']
    return self.AircraftTrainset, self.AircraftTestset


def set_experimental():
  fbx_instance = FireblastExperimental()
  fbx_instance.default_transform()
  fbx_instance.default_cars196()
  fbx_instance.default_cub200()
  fbx_instance.default_aircraft()

  return fbx_instance
