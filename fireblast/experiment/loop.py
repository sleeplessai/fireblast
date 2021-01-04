import time
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from . import Experiment


__all__ = ['Loop']


class Loop:

  @staticmethod
  def learn(
    expt: Union[Experiment, DataLoader],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    criterion = F.cross_entropy,
    multihead: int = 0,
    device: Union[torch.device, str] = torch.device('cuda'),
    epoch: int = -1,
    summary_writer: torch.utils.tensorboard.SummaryWriter = None,
    stdout_freq: int = 50,
    **kwargs
  ) -> Tuple[float, float]:
    """
    Universal one-epoch leaner loop for training an image
    classification neural network.

    Args:
      expt (Experiment, Dataloader):
        Experiment or PyTorch DataLoader object for trainset iteration
      model (torch.nn.Module):
        PyTorch nerual network model
      optimizer (torch.optim.Optimizer):
        PyTorch optimizer for training the given model
      scheduler (torch.optim.lr_scheduler._LRScheduler):
        PyTorch lr_scheduler object to update learning rate dynamically
      criterion (Callable):
        Loss function, cross entropy loss for image classification as default
      multihead (int):
        Multihead classifier count, the model output returns a list of results
      device (torch.device, str):
        The device where the model is trained
      epoch (int):
        Current training epoch
      summary_writer (torch.utils.tensorboard.SummaryWriter):
        Tensorboard SummaryWriter
      stdout_freq (int):
        Info stdout frequency, no info will be printed if set stdout_freq < 0

    Returns:
      train_loss, train_acc tuple(float, float) / list(float), list(float):
        training loss and accuracy at this epoch

    """
    if isinstance(expt, Experiment):
      assert expt.trainset_loader
      trainset_loader = expt.trainset_loader
    elif isinstance(expt, DataLoader):
      trainset_loader = expt
    else:
      raise RuntimeError

    _multihead_model = multihead >= 1
    smry = summary_writer

    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.train()

    if not _multihead_model:
      train_loss, correct, total, count = 0., 0, 0, 0
    else:
      count = 0
      train_loss = [0. for _ in range(multihead)]
      correct = [0 for _ in range(multihead)]
      total = correct[:]

    for index, (image, label) in enumerate(trainset_loader):
      optimizer.zero_grad()
      x, y = image.to(device), label.to(device)
      pred = model(x)
      if not _multihead_model:
        # forwd backwd
        loss = criterion(pred, y)
        train_loss += loss.item()
        loss.backward()

        # accuracy
        _, indices = torch.max(pred, dim=1)
        total += label.size(0)
        correct += indices.eq(y).cpu().sum()
        count = index

        # stdout
        if stdout_freq > 0 and index % stdout_freq == 0:
          _stdout_stm = []
          if epoch >= 0: _stdout_stm.append(('epoch', epoch))
          _stdout_stm.append(('batch', index))
          _stdout_stm.append(('loss', loss.item()))
          _on_air(_stdout_stm, localtime=True)
      else:
        # multihead forwd backwd
        loss = [criterion(p, y) for p in pred]
        train_loss = [a + b.item() for a, b in zip(train_loss, loss)]
        torch.stack(loss).sum().backward()

        # multihead accuracy
        for i in range(multihead):
          _, indices = torch.max(pred[i], dim=1)
          total[i] += label.size(0)
          correct[i] += indices.eq(y).cpu().sum()
          count = index

        # multihead stdout
        if stdout_freq > 0 and index % stdout_freq == 0:
          _stdout_stm = []
          if epoch >= 0: _stdout_stm.append(('epoch', epoch))
          _stdout_stm.append(('batch', index))
          for i in range(multihead):
            _stdout_stm.append((f'loss_{i}', loss[i].item()))
          _stdout_stm.append(('loss_sum', np.sum(loss).item()))
          _on_air(_stdout_stm, localtime=True)
      optimizer.step()
    if scheduler: scheduler.step()

    # summary
    if not _multihead_model:
      # to accuracy percentage
      train_loss /= (count + 1)
      train_acc = 100. * float(correct) / total

      # stdout, tensorboard
      if stdout_freq > 0:
        _stdout_stm = []
        if epoch >= 0:
          _stdout_stm.append(('epoch', epoch))
        _stdout_stm.append(('train_loss', train_loss))
        _stdout_stm.append(('train_acc', train_acc))
        _on_air(_stdout_stm)
        print()
      if smry and epoch != -1:
        smry.add_scalar('Training/Loss/epoch', train_loss, epoch)
        smry.add_scalar('Training/Accuracy/epoch', train_acc, epoch)
    else:
      # to accuracy percentage
      train_loss = [tl / (count + 1) for tl in train_loss]
      train_acc = [100. * float(correct[i]) / total[i] for i in range(multihead)]

      # stdout, tensorboard
      if stdout_freq > 0:
        _stdout_stm = []
        if epoch >= 0:
          _stdout_stm.append(('epoch', epoch))
        for i in range(multihead):
          _stdout_stm.append((f'train_loss_{i}', train_loss[i]))
          _stdout_stm.append((f'train_acc_{i}', train_acc[i]))
        _on_air(_stdout_stm)
        print()
      if smry and epoch != -1:
        loss_scalars, acc_scalars = {}, {}
        for i in range(multihead):
          loss_scalars[f'Loss{i}'] = train_loss[i]
          acc_scalars[f'Acc{i}'] = train_acc[i]
        smry.add_scalars('Training/Loss/epoch', loss_scalars, epoch)
        smry.add_scalars('Training/Accuracy/epoch', acc_scalars, epoch)

    return train_loss, train_acc


  @staticmethod
  def validate(
    expt: Union[Experiment, DataLoader],
    model: nn.Module,
    criterion = F.cross_entropy,
    multihead: int = 0,
    device: Union[torch.device, str] = torch.device('cuda'),
    epoch: int = -1,
    summary_writer: torch.utils.tensorboard.SummaryWriter = None,
    stdout_freq: int = 50,
    **kwargs
  ) -> Tuple[float, float]:
    """
    Universal validation loop for testing an image
    classification neural network.

    Args:
      expt (Experiment, Dataloader):
        Experiment or PyTorch DataLoader object for testset iteration
      model (torch.nn.Module):
        PyTorch nerual network model
      criterion (Callable):
        Loss function, cross entropy loss for image classification as default
      multihead (int):
        Multihead classifier count, the model output returns a list of results
      device (torch.device, str):
        The device for the model validation
      epoch (int):
        The epoch validation after training
      summary_writer (torch.utils.tensorboard.SummaryWriter):
        Tensorboard SummaryWriter
      stdout_freq (int):
        Info stdout frequency, no info will be printed if set stdout_freq < 0

    Returns:
      val_loss, val_acc tuple(float, float) / list(float), list(float):
        validation loss and accuracy at this epoch

    """
    if isinstance(expt, Experiment):
      assert expt.testset_loader
      testset_loader = expt.testset_loader
    elif isinstance(expt, DataLoader):
      testset_loader = expt
    else:
      raise RuntimeError

    _multihead_model = multihead >= 1
    smry = summary_writer

    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    model.eval()

    if not _multihead_model:
      val_loss, correct, total, count = 0., 0, 0, 0
    else:
      count = 0
      val_loss = [0. for _ in range(multihead)]
      correct = [0 for _ in range(multihead)]
      total = correct[:]

    with torch.no_grad():
      for index, (image, label) in enumerate(testset_loader):
        x, y = image.to(device), label.to(device)
        pred = model(x)

        if not _multihead_model:
          # forwd backwd
          loss = criterion(pred, y)
          val_loss += loss.item()

          # accuracy
          _, indices = torch.max(pred, dim=1)
          total += label.size(0)
          correct += indices.eq(y).cpu().sum()
          count = index

          # stdout
          if stdout_freq > 0 and index % stdout_freq == 0:
            _stdout_stm = []
            if epoch >= 0: _stdout_stm.append(('epoch', epoch))
            _stdout_stm.append(('batch', index))
            _stdout_stm.append(('loss', loss.item()))
            _on_air(_stdout_stm, localtime=True)
        else:
          # multihead forwd backwd
          loss = [criterion(p, y) for p in pred]
          val_loss = [a + b.item() for a, b in zip(val_loss, loss)]

          # multihead accuracy
          for i in range(multihead):
            _, indices = torch.max(pred[i], dim=1)
            total[i] += label.size(0)
            correct[i] += indices.eq(y).cpu().sum()
            count = index

          # multihead stdout
          if stdout_freq > 0 and index % stdout_freq == 0:
            _stdout_stm = []
            if epoch >= 0: _stdout_stm.append(('epoch', epoch))
            _stdout_stm.append(('batch', index))
            for i in range(multihead):
              _stdout_stm.append((f'loss_{i}', loss[i].item()))
            _stdout_stm.append(('loss_sum', np.sum(loss).item()))
            _on_air(_stdout_stm, localtime=True)

    # summary
    if not _multihead_model:
      # to accuracy percentage
      val_loss /= (count + 1)
      val_acc = 100. * float(correct) / total

      # stdout, tensorboard
      if stdout_freq > 0:
        _stdout_stm = []
        if epoch >= 0:
          _stdout_stm.append(('epoch', epoch))
        _stdout_stm.append(('val_loss', val_loss))
        _stdout_stm.append(('val_acc', val_acc))
        _on_air(_stdout_stm)
        print()
      if smry and epoch != -1:
        smry.add_scalar('Validation/Loss/epoch', val_loss, epoch)
        smry.add_scalar('Validation/Accuracy/epoch', val_acc, epoch)
    else:
      # to accuracy percentage
      val_loss = [tl / (count + 1) for tl in val_loss]
      val_acc = [100. * float(correct[i]) / total[i] for i in range(multihead)]

      # stdout, tensorboard
      if stdout_freq > 0:
        _stdout_stm = []
        if epoch >= 0:
          _stdout_stm.append(('epoch', epoch))
        for i in range(multihead):
          _stdout_stm.append((f'val_loss_{i}', val_loss[i]))
          _stdout_stm.append((f'val_acc_{i}', val_acc[i]))
        _on_air(_stdout_stm)
        print()
      if smry and epoch != -1:
        loss_scalars, acc_scalars = {}, {}
        for i in range(multihead):
          loss_scalars[f'Loss{i}'] = val_loss[i]
          acc_scalars[f'Acc{i}'] = val_acc[i]
        smry.add_scalars('Validation/Loss/epoch', loss_scalars, epoch)
        smry.add_scalars('Validation/Accuracy/epoch', acc_scalars, epoch)

    return val_loss, val_acc


def _on_air(
  stream: list,
  stdout: bool = True,
  localtime: bool = False,
  eq_sign: str = ':',
  delimiter: str = ', '
) -> str:
  stm_str = ''
  if localtime:
    loc_t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    stm_str += f'time{eq_sign}{loc_t}{delimiter}'
  for k, v in stream:
    if k in ['epoch', 'batch']:
      stm_str += f'{k}{eq_sign}{v}'
    elif 'acc' in k:
      stm_str += f'{k}{eq_sign}{v:.4f}%'
    elif 'loss' in k:
      stm_str += f'{k}{eq_sign}{v:.5f}'
    else:
      continue  # todo more
    stm_str += f'{delimiter}'
  stm_str = stm_str[:-len(delimiter)]
  if stdout: print(stm_str)
  return stm_str
