import torch
import torch.nn as nn
import torch.nn.functional as F
from .default import FireblastExperiment
from typing import Union, Tuple
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


__all__ = ['Loop']


class Loop:

  @staticmethod
  def learn(
    fbxo: Union[FireblastExperiment, DataLoader],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    criterion = F.cross_entropy,
    device: Union[torch.device, str] = torch.device('cuda:0'),
    epoch: int = -1,
    summary_writer: torch.utils.tensorboard.SummaryWriter = None,
    stdout_freq: int = 50,
    **kwargs
  ) -> Tuple[float, float]:
    """
    Universal one-epoch leaner loop for training an image classification neural network.

    Args:
      fbxo (FireblastExperiment, Dataloader): A FireblastExperiment or PyTorch DataLoader object for trainset iteration
      model (torch.nn.Module): PyTorch nerual network model(network)
      optimizer (torch.optim.Optimizer): PyTorch optimizer for training the given model
      scheduler (torch.optim.lr_scheduler._LRScheduler): PyTorch lr_scheduler object to update learning rate dynamically
      criterion (Callable): Loss function, cross entropy loss for image classification as default
      device (torch.device, str): The device where the model is trained
      epoch (int): Current training epoch
      summary_writer (torch.utils.tensorboard.SummaryWriter): Tensorboard SummaryWriter records loss and accuracy
      stdout_freq (int): Info stdout frequency, no info will be printed if set std_freq < 0

    Returns:
      train_loss, train_accuracy: Float classification metric pair of training loss and accuracy at this epoch

    """
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
    std_epoch = f'epoch: {epoch} | ' if epoch >= 0 else ''
    for index, (image, label) in enumerate(trainset_loader):
      optimizer.zero_grad()
      x, y = image.to(device), label.to(device)
      pred = model(x)
      loss = criterion(pred, y)
      train_loss += loss.item()
      loss.backward()
      optimizer.step()
      # accuracy
      _, indices = torch.max(pred, dim=1)
      total += label.size(0)
      correct += indices.eq(y).cpu().sum()
      count = index
      # stdout
      if stdout_freq > 0 and index % stdout_freq == 0:
        print(std_epoch + f'index: {index} | loss: {loss.item():.5f}')

    if scheduler: scheduler.step()

    train_loss /= (count + 1)
    train_accuracy = 100. * float(correct) / total

    if stdout_freq > 0:
      print(std_epoch + f'train_loss: {train_loss:.5f} | train_accuracy: {train_accuracy:.4f}%')
    if summary_writer and epoch != -1:
      summary_writer.add_scalar('Training/Loss/epoch', train_loss, epoch)
      summary_writer.add_scalar('Training/Accuracy/epoch', train_accuracy, epoch)

    return train_loss, train_accuracy


  @staticmethod
  def validate(
    fbxo: Union[FireblastExperiment, DataLoader],
    model: nn.Module,
    criterion = F.cross_entropy,
    device: Union[torch.device, str] = torch.device('cuda:0'),
    epoch: int = -1,
    summary_writer: torch.utils.tensorboard.SummaryWriter = None,
    stdout_freq: int = 50,
    **kwargs
  ) -> Tuple[float, float]:
    """
    Universal validation loop for testing an image classification neural network.

    Args:
      fbxo (FireblastExperiment, Dataloader): A FireblastExperiment or PyTorch DataLoader object for testset iteration
      model (torch.nn.Module): PyTorch nerual network model(network)
      criterion (Callable): Loss function, cross entropy loss for image classification as default
      device (torch.device, str): The device for the model validation
      epoch (int): The epoch validation after training
      summary_writer (torch.utils.tensorboard.SummaryWriter): Tensorboard SummaryWriter records loss and accuracy on testset
      stdout_freq (int): Info stdout frequency, no info will be printed if set std_freq < 0

    Returns:
      val_loss, val_accuracy: Float classification metric pair of validation loss and accuracy on testset

    """
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
    val_loss = 0.
    correct, total, count = 0, 0, 0
    std_epoch = f'epoch: {epoch} | ' if epoch >= 0 else ''
    for index, (image, label) in enumerate(testset_loader):
      x, y = image.to(device), label.to(device)
      pred = model(x)
      loss = criterion(pred, y)
      val_loss += loss.item()
      # accuracy
      _, indices = torch.max(pred, dim=1)
      total += label.size(0)
      correct += indices.eq(y).cpu().sum()
      count = index
      # stdout
      if stdout_freq > 0 and index % stdout_freq == 0:
        print(std_epoch + f'index: {index} | loss: {loss.item():.5f}')

    val_loss /= (count + 1)
    val_accuracy = 100. * float(correct) / total

    if stdout_freq > 0:
      print(std_epoch + f'val_loss: {val_loss:.5f} | val_accuracy: {val_accuracy:.4f}%')
    if summary_writer and epoch >= 0:
      summary_writer.add_scalar('Validation/Loss/epoch', val_loss, epoch)
      summary_writer.add_scalar('Validation/Accuracy/epoch', val_accuracy, epoch)

    return val_loss, val_accuracy
