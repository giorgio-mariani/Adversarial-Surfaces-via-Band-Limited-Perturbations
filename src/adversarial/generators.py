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

def metric(func):
  func.is_metric_decorator = True
  return func

class AdversarialGenerator(object):
  def __init__(self,
      pos:torch.Tensor,
      edges:torch.LongTensor,
      faces:torch.LongTensor,
      target:int,
      classifier:torch.nn.Module,
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

    # other info
    self.adversarial_coeff = torch.tensor([adversarial_coeff], device=pos.device, dtype=pos.dtype)
    self._zero = torch.zeros([1], device=pos.device, dtype=pos.dtype)
    self._smooth_L1_criterion = torch.nn.SmoothL1Loss(reduction='mean')
    self.loss_tracking_mod = 10

    self._r = None
    self._metrics = None
    self._iteration = None
    self._metrics_to_track = None

  @property
  def _metrics_functions(self):
    attributes = ((name, getattr(type(self), name, None)) for name in dir(self))
    return {name: attr for name, attr in attributes if getattr(attr, 'is_metric_decorator', False)}


  def _create_perturbation(self):
    return torch.zeros([self.vertex_count, 3], device=self.pos.device, dtype=self.pos.dtype, requires_grad=True)

  @property
  def perturbed_pos(self):
    return self.pos + self._r
    
  def total_loss(self):
    #laplace_beltrami_loss = self.LB_loss()
    #least_meshes_loss = self.smoothness_coeff*self.LSM_loss()
    adversarial_loss = self.adversarial_loss()
    distance_loss = self.distance_loss()
    loss = self.adversarial_coeff*adversarial_loss + distance_loss

    # add metrics
    if (self._iteration % self.loss_tracking_mod == 0):
      for n, metric in self._metrics_functions.items():
        if n in self._metrics_to_track:
          if n not in self._metrics:
            self._metrics[n] = []
          v = metric(self=self)
          self._metrics[n].append(v.item())
    return loss
  
  @metric
  def LSM_loss(self):
    n = self.vertex_count
    r = self.perturbed_pos - self.pos
    tmp = tsparse.spmm(*self.L, n, n, r) #Least square Meshes problem 
    return (tmp**2).sum()

  @metric
  def LB_loss(self):
    n = self.vertex_count
    stiff_r, area_r = mesh.laplacian.LB_v2(self.perturbed_pos, self.faces)
    ai, av = self.area
    ai_r, av_r = area_r
    _,L = tsparse.spspmm(ai, torch.reciprocal(av), *self.stiff, n, n, n)
    _,L_r = tsparse.spspmm(ai_r, torch.reciprocal(av_r), *stiff_r, n, n, n)
    loss = self._smooth_L1_criterion(L, L_r)
    return loss

  @metric 
  def euclidean_loss(self):
    return torch.norm(self.perturbed_pos - self.pos, p=2, dim=-1).sum()

  @metric
  def MCF_loss(self):
    n = self.vertex_count
    perturbed_pos = self.perturbed_pos
    stiff_r, area_r = mesh.laplacian.LB_v2(perturbed_pos, self.faces)
    
    tmp = tsparse.spmm(*self.stiff, n, n, self.pos)
    perturbed_tmp = tsparse.spmm(*stiff_r, n, n, perturbed_pos)
    
    ai, av = self.area
    ai_r, av_r = area_r

    mcf = tsparse.spmm(ai, torch.reciprocal(av), n, n, tmp)
    perturbed_mcf = tsparse.spmm(ai_r, torch.reciprocal(av_r), n, n, perturbed_tmp)
    diff_norm = torch.norm(mcf - perturbed_mcf,p=2,dim=-1)
    norm_integral = torch.dot(av, diff_norm)
    
    #a_diff = av-av_r
    #area_loss = torch.dot(a_diff,a_diff).sqrt_()
    loss = norm_integral
    return loss

  def distance_loss(self):
    return self.LB_loss()

  @metric
  def adversarial_loss(self) -> torch.Tensor:
    Z = self.classifier(self.perturbed_pos)
    values, index = torch.sort(Z, dim=0)
    argmax = index[-1] if index[-1] != self.target else index[-2] # max{Z(i): i != target}
    Ztarget, Zmax = Z[self.target], Z[argmax]
    return torch.max(Zmax - Ztarget, self._zero)

  @property
  def metrics(self, metric_name:str=None):
    if metric_name is None:
      return self._metrics
    else:
      return self._metrics[metric_name]

  def generate(self, 
    iter_num:int=1000, 
    lr:float=8e-6,
    usetqdm:str=None,
    metrics_to_track="all") -> torch.Tensor:
    # reset variables
    self._r = self._create_perturbation()
    self._iteration = 0
    self._metrics = dict()

    if metrics_to_track is None:
      self._metrics_to_track = []
    elif metrics_to_track=="all":
      self._metrics_to_track = self._metrics_functions.keys()
    elif hasattr(metrics_to_track, "__contains__"):
      self._metrics_to_track = metrics_to_track
    else:
      raise ValueError("Invalid input for 'metrics_to_track', valid values are: None, 'all' or an iterable of strings")


    # compute gradient w.r.t. the perturbation
    optimizer = torch.optim.Adam([self._r], lr=lr)
    self.classifier.eval()

    if usetqdm is None or usetqdm == False:
      iterations =  range(iter_num)
    elif usetqdm == "standard" or usetqdm == True:
      iterations = tqdm.trange(iter_num)
    elif usetqdm == "notebook":
      iterations = tqdm.tqdm_notebook(range(iter_num))
    else:
      raise ValueError("Invalid input for 'usetqdm', valid values are: None, 'standard' and 'notebook'.")

    for i in iterations:
      self._iteration += 1
      optimizer.zero_grad()
      loss = self.total_loss()
      loss.backward()
      optimizer.step()
    
    # print the losses for the last iteration
    for key,values in self._metrics.items():
      print(key+": "+str(values[-1]))
    r = self._r.clone().detach()
    adversarial_loss = self.adversarial_loss().clone().detach()
    return r, adversarial_loss

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class SpectralAdversarialGenerator(AdversarialGenerator):
  def __init__(self,
      pos:torch.Tensor,
      edges:torch.LongTensor,
      faces:torch.LongTensor,
      target:int,
      classifier:torch.nn.Module,
      adversarial_coeff:float=1,
      eigs_num:int=300):
    super().__init__(
      pos=pos,
      edges=edges,
      faces=faces,
      target=target,
      classifier=classifier,
      adversarial_coeff=adversarial_coeff)
    
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
    self.eigvals = torch.tensor(e, device=pos.device, dtype=pos.dtype)
    self.eigvecs = torch.tensor(phi, device=pos.device, dtype=pos.dtype)

  def _create_perturbation(self):
    return torch.zeros([self.eigs_num, 3], device=self.pos.device, dtype=self.pos.dtype, requires_grad=True)

  @property
  def perturbed_pos(self):
    return self.pos + self.eigvecs.matmul(self._r)

#------------------------------------------------------------------------------
class MCFAdversarialGenerator(SpectralAdversarialGenerator):
  def __init__(self,
      pos:torch.Tensor,
      edges:torch.LongTensor,
      faces:torch.LongTensor,
      target:int,
      classifier:torch.nn.Module,
      eigs_num:int=300,
      adversarial_coeff:float=1,
      smoothness_coeff:float=0):
    super().__init__(
      pos=pos,
      edges=edges,
      faces=faces,
      target=target,
      classifier=classifier,
      adversarial_coeff=adversarial_coeff)
  
  def distance_loss(self):
    return self.MCF_loss()

#------------------------------------------------------------------------------
class EuclideanAdversarialGenerator(SpectralAdversarialGenerator):
  def __init__(self, pos, edges, faces, target, classifier, adversarial_coeff=1):
    super().__init__(pos=pos, edges=edges, faces=faces, target=target,
      classifier=classifier, adversarial_coeff=adversarial_coeff)
  
  def distance_loss(self):
    return self.euclidean_loss()

#------------------------------------------------------------------------------
class DistAdversarialGenerator(SpectralAdversarialGenerator):
  def __init__(self,
      pos:torch.Tensor,
      edges:torch.LongTensor,
      faces:torch.LongTensor,
      target:int,
      classifier:torch.nn.Module,
      eigs_num:int=300,
      adversarial_coeff:float=1,
      smoothness_coeff:float=0):
    super().__init__(
      pos=pos, edges=edges, faces=faces, target=target,
      classifier=classifier, adversarial_coeff=adversarial_coeff)
    self.neighborhood = 30
    self.spirals = utils.get_spirals(pos=pos, edges=edges, neighbors_num=self.neighborhood, cutoff=5)
  
  @metric
  def local_euclidean_loss(self):
    flat_spiral_idx = self.spirals.view(-1)
    X = self.pos[flat_spiral_idx].view(-1, self.neighborhood, 3) # shape [n*20*3]
    Xr = self.perturbed_pos[flat_spiral_idx].view(-1,self.neighborhood,3)
    dist = torch.norm(X-self.pos.view(self.vertex_count,1,3), p=2,dim=-1)
    dist_r = torch.norm(Xr-self.perturbed_pos.view(self.vertex_count,1,3), p=2,dim=-1)
    dist_loss = self._smooth_L1_criterion(dist, dist_r)
    return dist_loss
  
  def distance_loss(self):
    return self.local_euclidean_loss()
