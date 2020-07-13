from fireblast.data.datasets import get_cub200_anns, cub200
from fireblast.data.datasets import get_fgvc_aircraft_anns, fgvc_aircraft
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# dev use
import logging
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)
#


def test_fireblast_datasets(name, iteration=False):
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
  elif name == 'Cars':
    # TODO
    pass
  pp.pprint(anns)
  for k in dtst.keys():
    img_loader = DataLoader(
      dataset=dtst[k],
      batch_size=256,
      shuffle=True,
      num_workers=6,
      pin_memory=True
    )
    if iteration:
      for idx, item in enumerate(img_loader):
        x, y = item[0].cuda(), item[1].cuda()
        # print(idx, item[0].shape, item[1])
      logging.warning(f'{name}.{k} iteration test passed.')


if __name__ == '__main__':
  # test_cub200()
  test_fireblast_datasets(name='Aircraft', iteration=True)

