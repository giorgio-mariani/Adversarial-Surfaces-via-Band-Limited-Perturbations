import math
import random

import torch

def transform_rotation_(pos, dims=[0,1,2]):
    phi_n = [random.random()*2*math.pi for _ in dims]
    cos_n = [math.cos(phi) for phi in phi_n]
    sin_n = [math.sin(phi) for phi in phi_n]
    device = pos.device

    R = torch.tensor(
        [[1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]], device=device, dtype=torch.float)

    random.shuffle(dims) # add randomness
    for i in range(len(dims)):
        dim, phi = dims[i], phi_n[i]
        cos, sin = math.cos(phi), math.sin(phi)

        if dim == 0:
            tmp = torch.tensor(
            [[1,   0,   0],
            [ 0, cos,-sin],
            [ 0, sin, cos]], device=device)
        elif dim == 1:
            tmp = torch.tensor(
            [[cos, 0,-sin],
            [ 0,   1,   0],
            [ sin, 0, cos]], device=device)
        elif dim == 2:
            tmp = torch.tensor(
            [[cos,-sin,  0],
            [ sin, cos,  0],
            [   0,   0,  1]], device=device)
        R = R.mm(tmp)
    pos[:,:] = torch.matmul(pos, R.t())


def transform_translation_ (pos):
    n = pos.shape[0]
    comp_device = pos.device
    comp_type = pos.dtype
    mean = torch.tensor([0], device=comp_device, dtype=comp_type)
    std = torch.tensor([0.5], device=comp_device, dtype=comp_type)

    centroid = pos.sum(dim=0, keepdim=True)/n
    offset = torch.normal(mean=mean, std=std)
    pos[:,:] = offset + (pos - centroid)
    return pos
