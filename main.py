from fireblast.data.datasets import get_cub200_anns, cub200
from fireblast.data.datasets import get_fgvc_aircraft_anns, fgvc_aircraft
from fireblast.data.datasets import get_cars196_anns, cars196
from fireblast.data.utils import _plot_pil_image, FBPP
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# dev use
import logging


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
  FBPP.pprint(anns)
  # return # for dev breakpoint
  if not dtst:
    return

  if iteration:
    for k in dtst.keys():
      img_loader = DataLoader(
        dataset=dtst[k],
        batch_size=128,
        shuffle=True,
        num_workers=6,
        pin_memory=True
      )
      for idx, item in enumerate(img_loader):
        x, y = item[0].cuda(), item[1].cuda()
        # print(idx, item[0].shape, item[1])
      logging.warning(f'{name}.{k} iteration passed.')

  if rand_plot:
    return
    import random
    from torchvision.transforms.functional import to_pil_image
    dt, _ = dtst['test'].__getitem__(random.randint(0, dtst.__len__() - 1))
    _plot_pil_image(dt)

  return anns, dtst


from fireblast.models.resnet import resnet18, resnet34, resnet50, resnext50_32x4d
from fireblast.models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn


def test_fireblast_models(name="resnet18", pretrained=False, plot_network=False):
  fireblast_models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnext50_32x4d': resnext50_32x4d,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn,
    'vgg16_bn': vgg16_bn
  }

  assert name in fireblast_models.keys() or name == 'all'
  if name == 'all':
    for k, v in fireblast_models.items():
      FBPP.pprint(k)
      if plot_network:
        FBPP.pprint(v(pretrained=False))
  else:
    net = fireblast_models[name](pretrained=pretrained)
    if plot_network: FBPP.pprint(net)
    return net

  return None


import fireblast.experiment.default as fbx


def test_fireblast_experiment():
  fbx_inst = fbx.FireblastExperiment()
  fbx.default_cub200(fbx_inst)
  FBPP.pprint(fbx_inst)
  fbx.default_aircraft(fbx_inst)
  FBPP.pprint(fbx_inst)
  fbx.default_cub200(fbx_inst, loader=True)
  FBPP.pprint(fbx_inst)


if __name__ == '__main__':
  TEST_FBD = False
  TEST_FBM = False
  TEST_FBX = True

  if TEST_FBD:
    test_fireblast_datasets(name='CUB200', iteration=False)
    test_fireblast_datasets(name='Aircraft', iteration=False)
    test_fireblast_datasets(name='Cars196', iteration=False)

  if TEST_FBM:
    test_fireblast_models(name="vgg11", pretrained=False, plot_network=True)
    test_fireblast_models(name="resnet50", pretrained=True, plot_network=True)
    # test_fireblast_models(name="all", pretrained=True, plot_network=True)

  if TEST_FBX:
    test_fireblast_experiment()
