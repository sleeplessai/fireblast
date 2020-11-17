import torch
import torch.nn as nn
import torch.nn.functional as F
from .default import FireblastExperiment
from typing import Union
from torch.utils.data import DataLoader


__all__ = ['Loop']


class Loop:

  @staticmethod
  def learn(
    fbxo: Union[FireblastExperiment, DataLoader],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    criterion = F.cross_entropy,
    device: Union[torch.device, str] = torch.device('cuda:0')
  ):
    if isinstance(fbxo, FireblastExperiment):
      assert fbxo.TrainsetLoader
      trainset_loader = fbxo.TrainsetLoader
    elif isinstance(fbxo, DataLoader):
      trainset_loader = fbxo
    else:
      raise RuntimeError

    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.train()
    train_loss = 0.
    correct, total, count = 0, 0, 0
    for index, (image, label) in enumerate(trainset_loader):
      optimizer.zero_grad()
      x, y = image.to(device), label.to(device)
      pred = model(x)
      loss = criterion(pred, y)
      train_loss += loss.item()
      loss.backward()
      optimizer.step()
      print(f'index: {index} | loss: {loss.item()}')
      # accuracy
      _, indices = torch.max(pred, dim=1)
      total += label.size(0)
      correct += indices.eq(y).cpu().sum()
      count = index
    if scheduler: scheduler.step()

    train_loss /= (count + 1)
    train_accuracy = 100. * float(correct) / total
    return train_loss, train_accuracy

  @staticmethod
  def validate(
    fbxo: Union[FireblastExperiment, DataLoader],
    model: nn.Module,
    criterion = F.cross_entropy,
    device: Union[torch.device, str] = torch.device('cuda:0')
  ):
    if isinstance(fbxo, FireblastExperiment):
      assert fbxo.TestsetLoader
      testset_loader = fbxo.TestsetLoader
    elif isinstance(fbxo, DataLoader):
      testset_loader = fbxo
    else:
      raise RuntimeError

    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.eval()
    test_loss = 0.
    correct, total, count = 0, 0, 0
    for index, (image, label) in enumerate(testset_loader):
      x, y = image.to(device), label.to(device)
      pred = model(x)
      loss = criterion(pred, y)
      test_loss += loss.item()
      # accuracy
      _, indices = torch.max(pred, dim=1)
      total += label.size(0)
      correct += indices.eq(y).cpu().sum()
      count = index

    test_loss /= (count + 1)
    test_accuracy = 100. * float(correct) / total
    return test_loss, test_accuracy
