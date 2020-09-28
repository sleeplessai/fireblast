import torch
import torch.nn as nn
import torch.nn.functional as F

from fe import Backbone
from mst import m_jigsaw_generator, mst_graph_generator

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv


class GConvNet(nn.Module):
  def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation, dropout):
    super(GConvNet, self).__init__()
    self.layers = nn.ModuleList()
    # input layer
    self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
    # hidden layers
    for i in range(n_layers - 1):
      self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
    # output layer
    self.layers.append(GraphConv(n_hidden, out_feats))
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, g, features):
    h = features
    for i, layer in enumerate(self.layers):
      if i != 0:
        h = self.dropout(h)
      h = layer(g, h)
    return h


class WUGNet(nn.Module):
  def __init__(self, in_feats_per_block=2048, out_feats_per_block=256, n_category=196):
    super(WUGNet, self).__init__()
    self.__version__ = "v0"
    self.gpu0 = torch.device('cuda:0')
    self.net1 = Backbone(backbone_1d=in_feats_per_block)
    self.net2 = GConvNet(
      in_feats=in_feats_per_block, n_hidden=512, 
      out_feats=out_feats_per_block, n_layers=8,
      activation=F.elu, dropout=0.5
    )
    self.fc = nn.Linear(256, n_category)

  def forward(self, x, n):
    # batch_size is 1, (1,3,w,h)-like
    assert x.shape[0] == 1 and len(x.shape) == 4 
    assert n in [1, 2, 4, 8, 16]

    x = m_jigsaw_generator(x, n)
    x = torch.cat(x, dim=0)
    x = self.net1(x)
    g = mst_graph_generator(None, n).to(self.gpu0)
    g.ndata['feat'] = x
    x = self.net2(g, g.ndata['feat'])

    with g.local_scope():
      g.ndata['feat'] = x
      # Calculate graph representation by average readout.
      r = dgl.mean_nodes(g, 'feat')
      x = self.fc(r)
      return x


if __name__ == "__main__":
  from PIL import Image
  from torchvision.transforms.functional import resize, to_tensor
  gpu = torch.device('cuda:0')
  img = Image.open('../amanda.jpg')
  img = resize(img, (448, 448))
  img_tensor = to_tensor(img)
  img_tensor = torch.unsqueeze(img_tensor, dim=0)
  x = img_tensor.to(gpu)
  model = WUGNet().to(gpu)
  
  multi_granularity = [1, 2, 4, 8, 16]
  for sz in multi_granularity:
    print(f"Testing granularity size: {sz}")
    R = model(x, sz)
    print(R.shape)
