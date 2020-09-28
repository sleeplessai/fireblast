import random
import numpy as np
import networkx as nx
import dgl


def m_jigsaw_generator(images, n):
  "https://arxiv.org/abs/2003.03836"

  assert images.shape[-1] == images.shape[-2]
  assert n > 0

  l = [[a, b] for a in range(n) for b in range(n)]
  # print(l)
  random.shuffle(l)
  # print(l)
  block_size = images.shape[-1] // n
  rounds = n ** 2
  jigsaw = images.clone()
  blocks = []
  for i in range(rounds):
    x, y = l[i]
    temp = jigsaw[..., 0:block_size, 0:block_size].clone()
    blocks.append(temp)
    jigsaw[..., 0:block_size, 0:block_size] = jigsaw[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size].clone()
    jigsaw[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp
  
  return blocks


def mst_graph_generator(blocks, n):
  g = nx.grid_2d_graph(n, n)
  for e in g.edges():
    g.add_edge(e[0], e[1], weight=random.randint(1, 100))
  g_mst = nx.algorithms.minimum_spanning_tree(g, weight='weight', algorithm='prim')
  g_mst = dgl.from_networkx(g_mst)
  
  return dgl.add_self_loop(g_mst)

