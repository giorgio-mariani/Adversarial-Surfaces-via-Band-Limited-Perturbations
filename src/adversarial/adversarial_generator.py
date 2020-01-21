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

class CarliniAdversarialGenerator(object):
  def __init__(self,
      pos:torch.Tensor,
      edges:torch.LongTensor,
      faces:torch.LongTensor,
      target:int,
      classifier:torch.nn.Module,
      eigs_num:int=150,
      smoothness_coeff:float=1,
      adversarial_coeff:float=1):
    super().__init__()
    utils.check_data(pos, edges, faces, float_type=torch.float)
    float_type = pos.dtype

    self.pos = pos
    self.faces = faces
    self.edges = edges
    self.target = torch.tensor([target], device=pos.device, dtype=torch.long)
    self.classifier = classifier

    #constants
    self.vertex_count = pos.shape[0]
    self.edge_count = edges.shape[0]
    self.face_count = faces.shape[0]

    # compute useful data
    self.L = tgeo.utils.get_laplacian(edges.t(), normalization="rw")
    self.stiff, self.area = mesh.laplacian.LB_v2(self.pos, self.faces)
    
    # compute spectral information
    n = self.vertex_count
    ri,ci = self.stiff[0].cpu().detach().numpy()
    sv = self.stiff[1].cpu().detach().numpy()
    S = scipy.sparse.csr_matrix( (sv, (ri,ci)), shape=(n,n))

    ri,ci = self.area[0].cpu().detach().numpy()
    av = self.area[1].cpu().detach().numpy()
    A = scipy.sparse.csr_matrix( (av, (ri,ci)), shape=(n,n))
    e, phi = scipy.sparse.linalg.eigsh(S, M=A, k=eigs_num, sigma=-1e-6)
    self.eigs_num = eigs_num
    self.eigvals = torch.tensor(e, device=pos.device, dtype=float_type)
    self.eigvecs = torch.tensor(phi, device=pos.device, dtype=float_type)

    # other info
    self.smoothness_coeff = torch.tensor([smoothness_coeff], device=pos.device, dtype=pos.dtype)
    self.adversarial_coeff = torch.tensor([adversarial_coeff], device=pos.device, dtype=pos.dtype)
    self._zero = torch.zeros([1], device=pos.device, dtype=pos.dtype)
    self._smooth_L1_criterion = torch.nn.SmoothL1Loss(reduction='mean')
    self.loss_tracking_mod = 10

    self._r = None
    self._loss_values = None
    self._iteration = None

  def _create_perturbation(self):
    return torch.zeros([self.vertex_count,3], device=self.pos.device, dtype=self.pos.dtype, requires_grad=True)

  @property
  def perturbed_pos(self):
    return self.pos + self._r
    
  def total_loss(self):
    adversarial_loss = self.adversarial_coeff*self.adversarial_loss()
    laplace_beltrami_loss = self.LB_loss()
    least_meshes_loss = self.smoothness_coeff*self.LSM_loss()
    loss = adversarial_loss + laplace_beltrami_loss + least_meshes_loss

    if self._iteration % self.loss_tracking_mod == 0:
      self._loss_values.append(
      {"adversarial":adversarial_loss.item(), 
      "laplace-beltrami":laplace_beltrami_loss.item(),
      "least-square-meshes":least_meshes_loss.item()})
    return loss
  
  def LSM_loss(self):
    n = self.vertex_count
    tmp = tsparse.spmm(*self.L, n, n, self._r) #Least square Meshes problem 
    return (tmp**2).sum()

  def LB_loss(self):
    n = self.vertex_count
    stiff_r, area_r = mesh.laplacian.LB_v2(self.perturbed_pos, self.faces)
    ai, av = self.area
    ai_r, av_r = area_r
    _,L = tsparse.spspmm(ai, torch.reciprocal(av), *self.stiff, n, n, n)
    _,L_r = tsparse.spspmm(ai_r, torch.reciprocal(av_r), *stiff_r, n, n, n)
    loss = self._smooth_L1_criterion(L, L_r)
    return loss

  def adversarial_loss(self) -> torch.Tensor:
    Z = self.classifier(self.perturbed_pos)
    values, index = torch.sort(Z, dim=0)
    argmax = index[-1] if index[-1] != self.target else index[-2] # max{Z(i): i != target}
    Ztarget, Zmax = Z[self.target], Z[argmax]
    return torch.max(Zmax - Ztarget, self._zero)

  def generate(self, iter_num:int=1000, lr=8e-6) -> torch.Tensor:
    # reset variables
    self._r = self._create_perturbation()
    self._iteration = 0
    self._loss_values = []

    # compute gradient w.r.t. the perturbation
    optimizer = torch.optim.Adam([self._r], lr=lr)
    self.classifier.eval()
    for i in range(iter_num):
      self._iteration += 1
      optimizer.zero_grad()
      loss = self.total_loss()
      loss.backward()
      optimizer.step()
    
    # print the losses for the last iteration
    for key,value in self._loss_values[-1].items():
      print(key+": "+str(value))
    return self._r.clone().detach(), self.adversarial_loss().clone().detach()

def estimate_perturbation(
  pos:torch.Tensor,
  edges:torch.LongTensor,
  faces:torch.LongTensor,
  target:int,
  classifier:torch.nn.Module,
  search_iterations=20,
  minimization_iterations=1000,
  starting_c:float=1,
  smoothness_coeff:float=1,
  eigs_num:int=150):

  range_min, range_max = 0, starting_c
  c_optimal = None 
  r_optimal = None
  increasing = True # flag used to detected whether it is the first phase or the second phase 

  for i in range(bsearch_iterations):
    midvalue = (range_min+range_max)/2
    c = range_max if increasing else midvalue

    print("\nbinary search step: "+str(i+1))
    print("range: [{},{}]\nc value: {}".format(range_min, range_max, c))
    print("iterations per step: {}".format(optim_iterations))
    print("phase: "+ ("incrementing" if increasing else "search"))

    adv_generator = CarliniAdversarialGenerator(
      pos=pos,
      edges=edges,
      faces=faces,
      target=target,
      classifier=classifier,
      smoothness_coeff=smoothness_coeff,
      adversarial_coeff=c)
    r, adversarial_loss = adv_generator.generate(iter_num=optim_iterations)

    # update best estimation
    if adversarial_loss <= 0:
      c_optimal, r_optimal = c, r

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
  if r_optimal is None:
    c_optimal, r_optimal = c, r
  return c_optimal, r_optimal
