from fireblast.data.datasets import cub200, get_cub200_anns
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


if __name__ == '__main__':

  im_transfm = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
  ])
  tg_transfm = None

  cub2h = get_cub200_anns(r'F:/data/CUB_200_2011', check=True)
  cub200 = cub200(anns=cub2h, transform=im_transfm, target_transform=tg_transfm)
 
  img_loader = DataLoader(
    dataset=cub200['traintest'],
    batch_size=256,
    shuffle=False,
    num_workers=6,
    pin_memory=True
  )
  # for idx, item in enumerate(img_loader):
  #   x, y = item[0].cuda(), item[1].cuda()
  #   print(idx, item[0].shape, item[1])