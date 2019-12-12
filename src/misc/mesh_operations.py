import math
import random

import torch
import tqdm

def transform_rotation_ (pos):
    phi = random.random()*2*math.pi
    cos, sin = math.cos(phi), math.sin(phi)
    R = torch.tensor(
        [[cos, 0,-sin],
        [ 0,   1,   0],
        [ sin, 0, cos]], device=pos.device)
    pos[:,:] = torch.matmul(pos, R.t())
    
def transform_position_(pos, width, offset):
    x = random.random()*width + offset
    y = random.random()*width + offset
    z = random.random()*width + offset

    offset = torch.tensor([x,y,z], device=pos.device)
    pos[:,:] += offset
