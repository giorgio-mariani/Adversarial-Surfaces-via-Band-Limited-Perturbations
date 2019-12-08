import numpy as np
import tqdm
import torch 


def generate_perturbation(x:torch.Tensor, classifier:torch.nn.Module, iteration_number=1e3):
    classifier(x)
    #define distortion loss
    loss_distr = 0 # possible distortions are vector field norm or smoothing laplacian (minimize laplacian)
    #define adversarial loss
    loss_adv = 0
    #define loss function
    loss = loss_adv + loss_distr
    # compute gradient w.r.t. the perturbation
    for i in range(iteration_number):
        # TODO compute gradient
        pass
    return None
