from fireblast.data.datasets import get_cub200_anns, cub200
from fireblast.data.datasets import get_fgvc_aircraft_anns, fgvc_aircraft
from fireblast.data.datasets import get_cars196_anns, cars196
from fireblast.data.utils import _plot_pil_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# dev use
import logging
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)


def test_fireblast_datasets(name, iteration=False, rand_plot=False):
  im_transfm = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor()
  ])
  if name == 'CUB200':
    anns = get_cub200_anns(r'/home/mlss/data/CUB_200_2011', check=True)
    dtst = cub200(anns=anns, transform=im_transfm)
  elif name == 'Aircraft':
    anns = get_fgvc_aircraft_anns(root=r'/home/mlss/data/fgvc-aircraft-2013b', check=True)
    dtst = fgvc_aircraft(anns=anns, transform=im_transfm)
  elif name == 'Cars196':
    anns = get_cars196_anns(root=r'/home/mlss/data/cars196', check=True)
    dtst = cars196(anns=anns, transform=im_transfm)
  pp.pprint(anns)
  # return # for dev breakpoint
  if not dtst:
    return
  
  for k in dtst.keys():
    img_loader = DataLoader(
      dataset=dtst[k],
      batch_size=128,
      shuffle=True,
      num_workers=6,
      pin_memory=True
    )
    if iteration:
      for idx, item in enumerate(img_loader):
        x, y = item[0].cuda(), item[1].cuda()
        # print(idx, item[0].shape, item[1])
      logging.warning(f'{name}.{k} iteration passed.')
  
  if rand_plot:
    import random
    from torchvision.transforms.functional import to_pil_image
    dt, _ = dtst['test'].__getitem__(random.randint(0, dtst.__len__() - 1))
    _plot_pil_image(dt)


if __name__ == '__main__':
  test_fireblast_datasets(name='CUB200', iteration=True, rand_plot=True)
  test_fireblast_datasets(name='Aircraft', iteration=True, rand_plot=True)
  test_fireblast_datasets(name='Cars196', iteration=True, rand_plot=True)
