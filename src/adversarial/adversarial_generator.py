import numpy as np
import tqdm
import torch
import torch.nn.functional as func

from  mesh.laplacian import laplacebeltrami_FEM

def find_perturbation(
    x:torch.Tensor,
    edge_index:torch.IntTensor,
    faces:torch.IntTensor,
    target:int,
    classifier:torch.nn.Module,
    max_c=100,
    bsearch_iterations=20,
    optim_iterations=500):
    
    vertex_count = x.shape[0]
    r = torch.normal(0, std=0.001, size=[vertex_count, 3], requires_grad=True, device=x.device)

    range_min, range_max = 0, max_c
    c_optimal,r_optimal = None, None
    for i in range(bsearch_iterations):
        midvalue = (range_min+range_max)/2
        print("binary search step: "+str(i+1))
        print("c value: "+str(midvalue))
        
        r_optim, adv_loss, dist_loss = _optimize_perturbation(
            x=x,f=faces,
            r=r, t=target,
            C=classifier,
            iter_num=optim_iterations, 
            c=midvalue)
        print("adversarial loss: "+str(adv_loss))
        print("Distorsion loss: "+str(dist_loss))

        if adv_loss <= 0:
            range_max = midvalue
            c_optimal = midvalue
            r_optimal = r_optim
        else:
            range_min = midvalue
    return c_optimal, r_optimal

def _optimize_perturbation(
    x:torch.Tensor,
    f:torch.Tensor,
    r:torch.Tensor,
    t:int, 
    C:torch.nn.Module,
    iter_num=1000,
    c=1.0):

    r = r.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([r], lr=1e-3, weight_decay=5e-4)

    # compute gradient w.r.t. the perturbation
    C.eval()
    for i in range(iter_num):
        optimizer.zero_grad()
        Z = C(x+r)
        #loss = c*_adversarial_loss(Z, t, 0) + _vector_field_norm(r)
        loss = c*_adversarial_loss(Z, t) + _area_difference(x, f, r)
        loss.backward()
        optimizer.step()
    Z=C(x+r)
    return r, _adversarial_loss(Z, t).item(), _area_difference(x, f, r).item()

def _adversarial_loss(Z:torch.Tensor, target:int, k=0):
    values, index = torch.sort(Z, dim=0)
    argmax = index[-1] if index[-1] != target else index[-2]
    Zt = Z[target]
    Zmax = Z[argmax]
    return torch.max(Zmax - Zt, torch.tensor([-k], device=Z.device, dtype=Z.dtype))

# disturbance metrics --------------------------------------------------
def _vector_field_norm(r:torch.Tensor):
    norm = r.norm(dim=1)
    return norm.sum()

def _area_difference(pos:torch.Tensor, faces:torch.Tensor, r:torch.Tensor):
    _,_, A1 = laplacebeltrami_FEM(pos,faces)
    _,_, A2 = laplacebeltrami_FEM(pos+r,faces)
    return ((A1-A2)**2).sum()
 
