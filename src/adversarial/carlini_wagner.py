from typing import Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm
import torch
import torch.nn.functional as func
import torch_sparse as tsparse
import torch_geometric as tgeo
import scipy.sparse

import utils
from .base import AdversarialExample, Builder


#------------------------------------------------------------------------------
class ValueLogger(object):
  def __init__(self, value_functions:dict=None, log_interval:int=10):
    super().__init__()
    self.logged_values =  dict()
    self.obj2name = dict()
    self.value_functions = value_functions if value_functions is not None else dict()
    self.log_interval=log_interval

    #initialize logging metrics
    for n,f in self.value_functions.items():
      self.add_value_name(n,f)

  def reset(self):
    for func, values in self.logged_values.items():
      values.clear()

  def add_value_name(self, name=None, obj=None):
    if obj == None and name == None:
      raise ValueError("either name or obj must be not None!")

    if obj != None and name == None:
      name = obj.__name__

    if obj == None:
      self.logged_values[name] = []
    else:
      self.logged_values[name] = []
      self.obj2name[obj] = name

  def add_scalar(self, name:str, value:float):
    if name in self.logged_values: 
      self.logged_values[name].append(value)
    elif name in self.obj2name:
      self.logged_values[obj2name[name]].append(value)
    else:
      raise ValueError("invalid input name")

  def log(self, adv_example, iteration):
    if self.log_interval != 0 and iteration % self.log_interval == 0:
      for n,f in self.value_functions.items():
        v = f(adv_example).item()
        self.add_scalar(n,v)

  def show(self):
    plt.figure()
    X = [np.array(v) for v in self.logged_values.values()]
    for i,array in enumerate(X): plt.plot(array/array.max())
    legend = ["{}:{:.4g}".format(k, vs[-1])for k,vs in self.logged_values.items()]
    plt.legend(legend)
    plt.show()

#------------------------------------------------------------------------------
class CWAdversarialExample(AdversarialExample):
  def __init__(self,
    pos:torch.Tensor,
    edges:torch.LongTensor,
    faces:torch.LongTensor,
    classifier:torch.nn.Module,
    target:int=None,
    adversarial_coeff:float=1,
    regularization_coeff:float=1,
    logger:ValueLogger=ValueLogger(),
    minimization_iterations:int=100,
    learning_rate:float=5e-5,
    k:float=0):

    super().__init__(
        pos=pos, edges=edges, faces=faces,
        classifier=classifier, target=target)
      
    self.k = torch.tensor([k], device=self.device, dtype=self.dtype)
    self.adversarial_coeff = torch.tensor([adversarial_coeff], device=self.device, dtype=self.dtype)
    self.regularization_coeff = torch.tensor([regularization_coeff], device=self.device, dtype=self.dtype)
    self.minimization_iterations = minimization_iterations
    self.learning_rate = learning_rate
    self.logger = logger

    # for untargeted-attacks use second most probable class    
    if target is None:
      Z = self.classifier(self.pos)
      values, index = torch.sort(Z, dim=0)
      self.target = index[-2] 

    # class components
    self.perturbation = None
    self.distortion_function = None
    self.regularization_function = None

  @property
  def device(self):  return self.pos.device
  
  @property
  def dtype(self): return self.pos.dtype

  @property
  def perturbed_pos(self):
    return self.perturbation.perturb_positions()

  @property
  def is_successful(self):
    return self.adversarial_loss().item() <= 0
  
  def adversarial_loss(self) -> torch.Tensor:
    ppos = self.perturbed_pos
    Z = self.classifier(ppos)
    values, index = torch.sort(Z, dim=0)
    argmax = index[-1] if index[-1] != self.target else index[-2] # max{Z(i): i != target}
    Ztarget, Zmax = Z[self.target], Z[argmax]
    return torch.max(Zmax - Ztarget, -self.k)

  def total_loss(self):
    loss = self.adversarial_coeff*self.adversarial_loss() + self.distortion_function(self)
    if self.regularization_function is not None:
      return loss + self.regularization_coeff*self.regularization_function()
    else:
      return loss

  def compute(self, usetqdm:str=None):
    # reset variables
    self.perturbation.reset()
    self.logger.reset() #ValueLogger({"adversarial":lambda x:x.adversarial_loss()})

    # compute gradient w.r.t. the perturbation
    optimizer = torch.optim.Adam([self.perturbation.r], lr=self.learning_rate)
    self.classifier.eval()

    if usetqdm is None or usetqdm == False:
      iterations =  range(self.minimization_iterations)
    elif usetqdm == "standard" or usetqdm == True:
      iterations = tqdm.trange(self.minimization_iterations)
    elif usetqdm == "notebook":
      iterations = tqdm.tqdm_notebook(range(self.minimization_iterations))
    else:
      raise ValueError("Invalid input for 'usetqdm', valid values are: None, 'standard' and 'notebook'.")

    for i in iterations:
      # compute loss
      optimizer.zero_grad()
      loss = self.total_loss()
      self.logger.log(self, i) #log results
      
      # backpropagate
      loss.backward()
      optimizer.step()

class CWBuilder(Builder):
  def __init__(self, search_iterations=1):
    super().__init__()
    self.adex_data = dict()
    self.adex_functions = dict()
    self.search_iterations = search_iterations

  def set_adversarial_coeff(self,c):
    self.adex_data["adversarial_coeff"] = c
    return self

  def set_perturbation_type(self, type:str, eigs_num:int=100):
    if type == "vertex":
      self.adex_functions["perturbation"] = Perturbation
    elif type == "spectral":
      self.adex_functions["perturbation"] = lambda x: SpectralPerturbation(x, eigs_num=eigs_num)
    else:
      raise ValueError("accepetd values: 'vertex', 'spectral'")
    return self

  def set_distortion_function(self, dfun):
    self.adex_functions["distortion"] = dfun
    return self

  def set_regularization_function(self, regularizer):
    self.adex_functions["regularizer"] = regularizer
    return self
  
  def set_regularization_coeff(self, regularization_coeff):
    self.adex_data["regularization_coeff"] = regularization_coeff
    return self

  
  def set_logger(self, logger):
    self.adex_data["logger"] = logger
    return self

  def set_minimization_iterations(self, minimization_iterations):
    self.adex_data["minimization_iterations"] = minimization_iterations
    return self

  def set_learning_rate(self, learning_rate):
    self.adex_data["learning_rate"] = learning_rate
    return self

  def set_k(self, k):
    self.adex_data["k"] = k
    return self

  def build(self, usetqdm:str=None) -> AdversarialExample:
    # exponential search variable
    range_min, range_max = 0, self.adex_data["adversarial_coeff"]
    optimal_example = None 
    increasing = True # flag used to detected whether it is the first phase or the second phase 

    # start exponential search
    for i in range(self.search_iterations):
      midvalue = (range_min+range_max)/2
      c = range_max if increasing else midvalue

      print("[{},{}] ; c={}".format(range_min, range_max, c))
      
      # create adversarial example
      self.set_adversarial_coeff(c)
      adex = CWAdversarialExample(**self.adex_data)
      
      adex.distortion_function = self.adex_functions["distortion"]
      adex.perturbation = self.adex_functions["perturbation"](adex)
      adex.regularization_function = self.adex_functions.get("regularizer", None)
      adex.compute(usetqdm=usetqdm)

      # get perturbation
      r = adex.perturbation.r
      adex.adversarial_loss().item()

      # update best estimation
      if adex.is_successful:
        optimal_example = adex

      # update loop variables
      if increasing and not adex.is_successful:
        range_min = range_max
        range_max = range_max*2
      elif increasing and adex.is_successful:
        increasing = False
      else:
        range_max = range_max if not adex.is_successful else midvalue
        range_min = midvalue  if not adex.is_successful else range_min

    # if unable to find a good c,r pair, return the best found solution
    is_successful = optimal_example is not None
    if not is_successful: optimal_example = adex
    return optimal_example

#==============================================================================
# perturbation functions ------------------------------------------------------
class Perturbation(object):
  def __init__(self, adv_example:CWAdversarialExample):
    super().__init__()
    self.r = None
    self.adv_example = adv_example
    self.reset()

  def reset(self):
    self.r = torch.zeros(
      [self.adv_example.vertex_count, 3], 
      device=self.adv_example.device, 
      dtype=self.adv_example.dtype, 
      requires_grad=True)

  def perturb_positions(self):
    pos, r = self.adv_example.pos, self.r
    return pos + r

class SpectralPerturbation(object):
  def __init__(self, adv_example, eigs_num=100):
    super().__init__()
    self.r = None
    self.adv_example = adv_example
    self.eigs_num = eigs_num
    self.eigvals,self.eigvecs = utils.eigenpairs(self.adv_example.pos, self.adv_example.faces, K=eigs_num)
    self.reset()

  def reset(self):
    self.r = torch.zeros(
      [self.eigs_num, 3], 
      device=self.adv_example.device, 
      dtype=self.adv_example.dtype, 
      requires_grad=True)

  def perturb_positions(self):
    pos, r = self.adv_example.pos, self.r
    return pos + self.eigvecs.matmul(r)

class LatentPerturbation(object):
  def __init__(self, adv_example):
    super().__init__()
    self.adv_example = adv_example
    self.reset()

  def reset(self):
    raise NotImplementedError()

  def perturb_positions(self):
    raise NotImplementedError()

# regularizers ----------------------------------------------------------------
def LSM_regularizer(adv_example:CWAdversarialExample):
    laplacian = tgeo.utils.get_laplacian(adv_example.edges.t(), normalization="rw")
    n = adv_example.vertex_count
    r = adv_example.perturbed_pos - adv_example.pos
    tmp = tsparse.spmm(*laplacian, n, n, r) #Least square Meshes problem 
    return (tmp**2).sum()

def centroid_regularizer(adv_example:CWAdversarialExample):
  adv_centroid = torch.mean(adv_example.perturbed_pos, dim=-1)
  centroid = torch.mean(adv_example.pos, dim=-1)
  return torch.nn.functional.mse_loss(adv_centroid, centroid)

def area_regularizer(adv_example:CWAdversarialExample):
  raise NotImplementedError() 


# distortion functions --------------------------------------------------------
def LB_distortion(adv_example:CWAdversarialExample):
    n = adv_example.vertex_count
    ppos = adv_example.perturbed_pos
    faces = adv_example.faces
    area = adv_example.area
    stiff = adv_example.stiff

    stiff_r, area_r = utils.laplacebeltrami_FEM_v2(ppos, faces)
    ai, av = area
    ai_r, av_r = area_r
    _,L = tsparse.spspmm(ai, torch.reciprocal(av), *stiff, n, n, n)
    _,L_r = tsparse.spspmm(ai_r, torch.reciprocal(av_r), *stiff_r, n, n, n)
    return torch.nn.functional.smooth_l1_loss(L, L_r)

def L2_distortion(adv_example:CWAdversarialExample):
    diff = adv_example.perturbed_pos - adv_example.pos
    area_indices, area_values = adv_example.area
    #L2_squared = torch.dot(area_values, (diff**2).sum(dim=-1))
    weight_diff = diff*torch.sqrt(area_values.view(-1,1)) # (sqrt(ai)*(xi-perturbed(xi)) )^2  = ai*(x-perturbed(xi))^2
    L2 = weight_diff.norm(p="fro")
    return L2

def spectral_L2_distortion(adv_example:CWAdversarialExample):
  if not isinstance(adv_example.perturbation,SpectralPerturbation):
    raise ValueError("Type of perturbation for input adversarial example is not spectral!")
  spectral_coefficients = adv_example.perturbation.r
  return spectral_coefficients.norm(p="fro")

def MC_distortion(adv_example:CWAdversarialExample):
    n = adv_example.vertex_count
    perturbed_pos = adv_example.perturbed_pos
    stiff_r, area_r = utils.laplacebeltrami_FEM_v2(perturbed_pos, adv_example.faces)
    
    tmp = tsparse.spmm(*adv_example.stiff, n, n, adv_example.pos)
    perturbed_tmp = tsparse.spmm(*stiff_r, n, n, perturbed_pos)
    
    ai, av = adv_example.area
    ai_r, av_r = area_r

    mcf = tsparse.spmm(ai, torch.reciprocal(av), n, n, tmp)
    perturbed_mcf = tsparse.spmm(ai_r, torch.reciprocal(av_r), n, n, perturbed_tmp)
    diff_norm = torch.norm(mcf - perturbed_mcf,p=2,dim=-1)
    norm_integral = torch.dot(av, diff_norm)
    
    #a_diff = av-av_r
    #area_loss = torch.dot(a_diff,a_diff).sqrt_()
    return norm_integral

class LocallyEuclideanDistortion(object):
  def __init__(self, K=30):
    super().__init__()
    self.adv_example = None
    self.kNN = None
    self.neighborhood = K

  def __call__(self, adv_example):
    if adv_example != self.adv_example:
      self.adv_example = adv_example
      self.kNN = utils.misc.kNN(
        pos=adv_example.pos, 
        edges=adv_example.edges, 
        neighbors_num=self.neighborhood, 
        cutoff=5)

    n = adv_example.vertex_count
    pos = adv_example.pos
    ppos = adv_example.perturbed_pos

    flat_kNN = self.kNN.view(-1)
    X = pos[flat_kNN].view(-1, self.neighborhood, 3) # shape [n*K*3]
    Xr = ppos[flat_kNN].view(-1, self.neighborhood, 3)
    dist = torch.norm(X-pos.view(n,1,3), p=2,dim=-1)
    dist_r = torch.norm(Xr-ppos.view(n,1,3), p=2,dim=-1)
    dist_loss = torch.nn.functional.smooth_l1_loss(dist, dist_r)
    return dist_loss


'''
def measure(func):
  func.is_measure_decorator = True
  return func

def _measures_functions(self):
  attributes = ((name, getattr(type(self), name, None)) for name in dir(self))
  return {name: attr for name, attr in attributes if getattr(attr, 'is_measure_decorator', False)}
'''