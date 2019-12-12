import numpy as np
import tqdm
import torch
import torch.nn.functional as func


def _optimize_perturbation(
    x:torch.Tensor,
    r:torch.Tensor,
    t:int, 
    C:torch.nn.Module, 
    iter_num=1000,
    c=1.0):

    r = r.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([r], lr=1e-3, weight_decay=5e-4)

    # compute gradient w.r.t. the perturbation
    C.eval()
    for i in tqdm.trange(iter_num):
        optimizer.zero_grad()
        Z = C(x+r)
        loss = _adversarial_loss(Z, t) + c*_vector_field_norm(r)
        loss.backward()
        optimizer.step()
    return r, _adversarial_loss(C(x+r), t)

def _vector_field_norm(r:torch.Tensor):
    norm = r.norm(dim=1)
    return norm.sum()

def _adversarial_loss(Z:torch.Tensor, target:int):
    values, index = torch.sort(Z, dim=0)
    argmax = index[-1] if index[-1] != target else index[-2]
    Zt = Z[target]
    Zmax = Z[argmax]
    return func.relu(Zmax - Zt)

def find_perturbation(
    x:torch.Tensor,
    edge_index:torch.IntTensor,
    target:int,
    classifier:torch.nn.Module,
    max_c=100,
    bsearch_iterations=20,
    optim_iterations=500):
    
    vertex_count = x.shape[0]
    r = torch.normal(0, std=0.05, size=[vertex_count, 3], requires_grad=True, device=x.device)

    range_min, range_max = 0, max_c
    c_optimal,r_optimal = None, None
    for i in range(bsearch_iterations):
        print("binary search step: "+str(i))
        midvalue = (range_min+range_max)/2
        r_optim, f_perturbed = _optimize_perturbation(
            x=x, r=r, t=target,
            C=classifier,
            iter_num=optim_iterations, 
            c=midvalue)
        print("adversarial loss: "+str(f_perturbed.item()))

        if f_perturbed <= 0:
            range_max = midvalue
            c_optimal = midvalue
            r_optimal = r_optim.requires_grad(False)
        else:
            range_min = midvalue
    return c_optimal, r_optimal


