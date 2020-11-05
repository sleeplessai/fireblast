import logging
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pprint import PrettyPrinter
FBPP = PrettyPrinter(indent=2)


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


def _plot_pil_image(pil_image: Image):
  if isinstance(pil_image, torch.Tensor):
    from torchvision.transforms.functional import to_pil_image
    pil_image = to_pil_image(pil_image)
  if not isinstance(pil_image, Image.Image): return
  plt.imshow(pil_image)
  plt.plot()
  plt.show()
