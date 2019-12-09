import numpy as np
import tqdm
import torch
import torch.nn.functional as func


def generate_perturbation(
    x:torch.Tensor, 
    target:int, 
    classifier:torch.nn.Module, 
    iteration_number=1000):

    n = x.shape[0]
    r = torch.normal(0, std=0.05, size=[n,3], requires_grad=True)

    C = 1 # metaparameter
    optimizer = torch.optim.Adam([r], lr=1e-3, weight_decay=5e-4)

    #define loss function
    # compute gradient w.r.t. the perturbation
    for i in tqdm.trange(iteration_number):
        optimizer.zero_grad()
        Z = classifier(x+r)
        loss = _adversarial_loss(Z, target) + C*_vector_field_norm(r)
        loss.backward()
        optimizer.step()
    return r

def _vector_field_norm(r:torch.Tensor):
    norm = r.norm(dim=1)
    return norm.sum()

def _adversarial_loss(Z:torch.Tensor, target:int):
    values, index = torch.sort(Z, dim=0)
    argmax = index[-1] if index[-1] != target else index[-2]
    Zt = Z[target]
    Zmax = Z[argmax]
    return Zmax - Zt

