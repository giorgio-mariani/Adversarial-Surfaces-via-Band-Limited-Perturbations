import numpy as np
import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from random import shuffle

def avg_pos(dataset:Dataset):
    x = torch.zeros(dataset[0].pos.shape)
    for data in dataset: x += data.pos
    return x/len(dataset)

def compute_impact(x:torch.Tensor, f:torch.nn.Module, perturbed_pos:torch.Tensor, num_samples:int):
    Z:torch.Tensor = f(x)
    impact = -torch.ones([x.shape[0]])
    vertex_count = x.shape[0]
    vertex_bag = list(range(vertex_count))
    shuffle(vertex_bag)
    for i in tqdm.trange(num_samples):
        vi = vertex_bag[i]
        tmp = x[vi,:].clone()
        x[vi,:] = perturbed_pos[vi,:]
        Z_per = f(x)
        x[vi,:] = tmp
        impact[vi] = torch.sum((Z-Z_per)**2)
    impact = impact/impact.max()
    return impact


