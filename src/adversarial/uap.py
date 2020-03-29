
import torch
import tqdm

import adversarial.iterative_gradient as ig
from mesh.laplacian import LB_v2
import utils


def _pred(Z):
    _, index = torch.sort(Z, dim=0)
    return index[-1]

def error(data, classifier, v):
    #percentage of misclassified data
    s = 0
    for m in data:
        x, y = m.pos, m.y
        if _pred(classifier(x+v)) != y: s+=1
    error = s/len(data)
    print("mean error: {}".format(error))
    return error


def projection(v, r, eps):
    p = v + r
    return p
    

def UAP_computation(
    data,
    classifier,
    delta:float,
    eps:float,
    K=30):

    device = data[0].pos.device
    typefloat = data[0].pos.dtype
    shape = data[0].pos.shape

    #filter data:
    filter_index = torch.tensor(
        [i for i in range(len(data)) if _pred(classifier(data[i].pos)) == data[i].y],
        dtype=torch.long, device=device)
    data = data[filter_index]

    # start universal adversarial perturbation computation
    v = torch.zeros(shape, dtype=typefloat, device=device)
    while error(data, classifier, v) <= (1-delta):
        for mi in tqdm.tqdm(data):
            xi = mi.pos
            ei = mi.edge_index.t().to(device)
            fi = mi.face.t().to(device)
            yi = mi.y
            
            if _pred(classifier(xi + v)) == yi:
                #Compute the minimal perturbation that sends xi + v 
                # to the decision boundary:
                _,(_,areas) = LB_v2(xi+v, fi)
                eigvals, eigvecs = utils.eigenpairs(xi+v, fi,K=K)
                xi_adversarial =ig.fast_gradient(
                    classifier=classifier,
                    x=xi+v, y=yi,
                    eigvecs=eigvecs,
                    areavec=areas)
                r = (xi - xi_adversarial).detach()
                
                #Update the perturbation:
                v = projection(v, r, eps)
    return v