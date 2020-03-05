from typing import Tuple

import networkx as nx
import numpy as np
import tqdm
import torch
import torch.nn.functional as func
import torch_sparse as tsparse
import torch_geometric as tgeo
import scipy.sparse

import utils
import mesh.laplacian
import adversarial.generators as adv

########################################################################################
def estimate_perturbation(
  pos:torch.Tensor,
  edges:torch.LongTensor,
  faces:torch.LongTensor,
  target:int,
  classifier:torch.nn.Module,
  search_iterations=20,
  minimization_iterations=1000,
  starting_c:float=1,
  adversarial_generator=adv.SpectralAdversarialGenerator,
  learning_rate:float=1e-4) -> Tuple[adv.AdversarialGenerator, bool]:

  range_min, range_max = 0, starting_c
  optimal_generator = None 
  increasing = True # flag used to detected whether it is the first phase or the second phase 

  for i in range(search_iterations):
    midvalue = (range_min+range_max)/2
    c = range_max if increasing else midvalue

    print("\nbinary search step: "+str(i+1))
    print("range: [{},{}]\nc value: {}".format(range_min, range_max, c))
    print("iterations per step: {}".format(minimization_iterations))
    print("phase: "+ ("incrementing" if increasing else "search"))

    adv_generator = adversarial_generator(
      pos=pos,
      edges=edges,
      faces=faces,
      target=target,
      classifier=classifier,
      adversarial_coeff=c)
    r, adversarial_loss = adv_generator.generate(iter_num=minimization_iterations, lr=learning_rate)

    # update best estimation
    if adversarial_loss <= 0:
      optimal_generator = adv_generator

    # update loop variables
    if increasing and adversarial_loss > 0:
      range_min = range_max
      range_max = range_max*2
    elif increasing and adversarial_loss <= 0:
      increasing = False
    else:
      range_max = range_max if adversarial_loss > 0 else midvalue
      range_min = midvalue  if adversarial_loss > 0 else range_min

  # if unable to find a good c,r pair, return the best found solution
  misclassified = optimal_generator is not None
  if not misclassified:
    optimal_generator = adv_generator
  return optimal_generator, misclassified

##-------------------------------------------------------------------------------
#Measures

  def LB_loss(pos, perturbed_pos, faces, stiff, area, perturbed_stiff, perturbed_area):
    n = pos.shape[0]
    ai, av = area
    ai_r, av_r = perturbed_area
    _,L = tsparse.spspmm(ai, torch.reciprocal(av), *stiff, n, n, n)
    _,perturbed_L = tsparse.spspmm(ai_r, torch.reciprocal(av_r), *perturbed_stiff, n, n, n)
    loss = torch.nn.functional.smooth_l1_loss(L, perturbed_L)
    return loss

  def euclidean_loss(pos, perturbed_pos):
    return torch.norm(perturbed_pos - pos, p=2, dim=-1).sum()

  def MC_loss(pos, perturbed_pos, stiff, area, perturbed_stiff, perturbed_area):
    n = pos.shape[0]
    tmp = tsparse.spmm(*stiff, n, n, pos)
    perturbed_tmp = tsparse.spmm(*perturbed_stiff, n, n, perturbed_pos)
    
    ai, av = area
    ai_r, av_r = perturbed_area

    mcf = tsparse.spmm(ai, torch.reciprocal(av), n, n, tmp)
    perturbed_mcf = tsparse.spmm(ai_r, torch.reciprocal(av_r), n, n, perturbed_tmp)
    diff_norm = torch.norm(mcf - perturbed_mcf,p=2,dim=-1)
    norm_integral = torch.dot(av, diff_norm)
    
    #a_diff = av-av_r
    #area_loss = torch.dot(a_diff,a_diff).sqrt_()
    loss = norm_integral
    return loss