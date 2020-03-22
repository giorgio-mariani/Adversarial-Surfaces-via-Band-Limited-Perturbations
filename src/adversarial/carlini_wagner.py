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
import mesh.laplacian


#------------------------------------------------------------------------------
class ValueLogger(object):
  def __init__(self, value_functions:dict=None):
    super().__init__()
    self.logged_values =  dict()
    self.obj2name = dict()
    self.value_functions = value_functions if value_functions is not None else dict()
    for n,f in value_functions.items():
      self.add_value_name(n,f)

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

  def log(self, adv_example):
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
class AdversarialExample(object):
  def __init__(self,
      pos:torch.Tensor,
      edges:torch.LongTensor,
      faces:torch.LongTensor,
      target:int,
      classifier:torch.nn.Module,
      adversarial_coeff:float=1,
      log_interval:int=10):
    super().__init__()
    utils.check_data(pos, edges, faces, float_type=torch.float)
    float_type = pos.dtype

    self.pos = pos
    self.faces = faces
    self.edges = edges
    self.adversarial_coeff = torch.tensor([adversarial_coeff], device=pos.device, dtype=pos.dtype)
    self.target = torch.tensor([target], device=pos.device, dtype=torch.long)
    self.classifier = classifier

    #constants
    self.vertex_count = pos.shape[0]
    self.edge_count = edges.shape[0]
    self.face_count = faces.shape[0]

    # compute useful data
    self.laplacian = tgeo.utils.get_laplacian(edges.t(), normalization="rw")
    self.stiff, self.area = mesh.laplacian.LB_v2(self.pos, self.faces)

    # class components
    self.perturbation = None
    self.distortion_function = None
    self.logger = None

    # other info
    self._iteration = None
    self._zero = torch.zeros([1], device=pos.device, dtype=pos.dtype)
    self.log_interval = log_interval

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
    return torch.max(Zmax - Ztarget, self._zero)

  def total_loss(self):
    loss = self.adversarial_coeff*self.adversarial_loss() + self.distortion_function(self)

    # add measures
    if (self._iteration % self.log_interval == 0):
      self.logger.log(self)
    return loss

  def optimize(self,
    iter_num:int=1000,
    lr:float=8e-6,
    usetqdm:str=None,
    logger:ValueLogger=None) -> torch.Tensor:
    # reset variables
    self.perturbation.reset()
    self._iteration = 0
    self.logger = logger if logger is not None else ValueLogger({"adversarial":lambda x:x.adversarial_loss()})

    # compute gradient w.r.t. the perturbation
    optimizer = torch.optim.Adam([self.perturbation.r], lr=lr)
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

class AdversarialExampleBuilder(object):
  def __init__(self):
    super().__init__()
    self.adv_ex_data = dict()
    self.set_log_interval(10)
    self.set_perturbation_type("spectral", 100)

  def set_mesh(self, pos, edges, faces):
    self.adv_ex_data["pos"] = pos
    self.adv_ex_data["edges"] = edges
    self.adv_ex_data["faces"] = faces
    return self

  def set_adversarial_coeff(self,c):
    self.adv_ex_data["adversarial_coeff"] = c
    return self
    
  def set_target(self,t):
    self.adv_ex_data["target"] = t
    return self

  def set_classifier(self, classifier):
    self.adv_ex_data["classifier"] = classifier
    return self

  def set_perturbation_type(self, type:str, eigs_num:int=100):
    if type == "vertex":
      self.adv_ex_data["perturbation"] = Perturbation
    elif type == "spectral":
      self.adv_ex_data["perturbation"] = lambda x: SpectralPerturbation(x, eigs_num=eigs_num)
    else:
      raise ValueError("accepetd values: 'vertex', 'spectral'")
    return self

  def set_distortion_function(self, dfun):
    self.adv_ex_data["distortion"] = dfun
    return self
    
  def set_log_interval(self, log_interval):
    self.adv_ex_data["log_interval"] = log_interval
    return self

  def build(self,
    iterations_number,
    learning_rate:float=8e-6,
    usetqdm:str=None,
    logger:ValueLogger=None):
    #TODO add checks on required fields

    tmp = self.adv_ex_data
    adex = AdversarialExample(
      pos=tmp["pos"],
      edges=tmp["edges"],
      faces=tmp["faces"],
      target=tmp["target"],
      classifier=tmp["classifier"],
      adversarial_coeff=tmp["adversarial_coeff"],
      log_interval=tmp["log_interval"])
    
    adex.distortion_function = tmp["distortion"]
    adex.perturbation = tmp["perturbation"](adex)
    adversarial_loss = adex.optimize(
      iter_num=iterations_number, 
      lr=learning_rate, 
      usetqdm=usetqdm, 
      logger=logger)
    return adex

  def build_and_tune(self,
    starting_coefficient=1,
    search_iterations=10,
    minimization_iterations=1000,
    learning_rate:float=1e-4,
    logger:ValueLogger=None) -> AdversarialExample:
    #TODO add checks on required fields

    range_min, range_max = 0, starting_coefficient
    optimal_example = None 
    increasing = True # flag used to detected whether it is the first phase or the second phase 

    for i in range(search_iterations):
      midvalue = (range_min+range_max)/2
      c = range_max if increasing else midvalue

      print("\nbinary search step: "+str(i+1))
      print("range: [{},{}]\nc value: {}".format(range_min, range_max, c))
      print("iterations per step: {}".format(minimization_iterations))
      print("phase: "+ ("incrementing" if increasing else "search"))

      adv_example = self.set_adversarial_coeff(c).build(
        iterations_number=minimization_iterations, 
        learning_rate=learning_rate,
        logger=logger)
      r = adv_example.perturbation.r
      adv_example.adversarial_loss().item()

      # update best estimation
      if adv_example.is_successful:
        optimal_example = adv_example

      # update loop variables
      if increasing and not adv_example.is_successful:
        range_min = range_max
        range_max = range_max*2
      elif increasing and adv_example.is_successful:
        increasing = False
      else:
        range_max = range_max if not adv_example.is_successful else midvalue
        range_min = midvalue  if not adv_example.is_successful else range_min

    # if unable to find a good c,r pair, return the best found solution
    is_successful = optimal_example is not None
    if not is_successful:
      optimal_example = adv_example
    return optimal_example

#==============================================================================

class Perturbation(object):
  def __init__(self, adv_example:AdversarialExample):
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

    n = self.adv_example.vertex_count
    stiff_indices, stiff_values = self.adv_example.stiff
    area_indices, area_values = self.adv_example.area

    # compute spectral information
    ri,ci = stiff_indices.cpu().clone().detach().numpy()
    sv = stiff_values.cpu().clone().detach().numpy()
    S = scipy.sparse.csr_matrix( (sv, (ri,ci)), shape=(n,n))

    ri,ci = area_indices.cpu().clone().detach().numpy()
    av = area_values.cpu().clone().detach().numpy()
    A = scipy.sparse.csr_matrix( (av, (ri,ci)), shape=(n,n))
    e, phi = scipy.sparse.linalg.eigsh(S, M=A, k=eigs_num, sigma=-1e-6)

    # set eigenvalues and eigenfunctions
    self.eigs_num = eigs_num
    self.eigvals = torch.tensor(e, device=self.adv_example.device, dtype=self.adv_example.dtype)
    self.eigvecs = torch.tensor(phi, device=self.adv_example.device, dtype=self.adv_example.dtype)
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


# -----------------------------------------------------------------------------
def LSM_regularizer(adv_example:AdversarialExample):
    n = adv_example.vertex_count
    r = adv_example.perturbed_pos - adv_example.pos
    tmp = tsparse.spmm(*adv_example.laplacian, n, n, r) #Least square Meshes problem 
    return (tmp**2).sum()

def LB_distortion(adv_example:AdversarialExample):
    n = adv_example.vertex_count
    ppos = adv_example.perturbed_pos
    faces = adv_example.faces
    area = adv_example.area
    stiff = adv_example.stiff

    stiff_r, area_r = mesh.laplacian.LB_v2(ppos, faces)
    ai, av = area
    ai_r, av_r = area_r
    _,L = tsparse.spspmm(ai, torch.reciprocal(av), *stiff, n, n, n)
    _,L_r = tsparse.spspmm(ai_r, torch.reciprocal(av_r), *stiff_r, n, n, n)
    return torch.nn.functional.smooth_l1_loss(L, L_r)

def L2_distortion(adv_example:AdversarialExample):
    diff = adv_example.perturbed_pos - adv_example.pos
    area_indices, area_values = adv_example.area
    L2_squared = torch.dot(area_values, (diff**2).sum(dim=-1))
    return L2_squared #torch.sqrt(L2_squared) (problem with non differentiability of sqrt at zero)

def MC_distortion(adv_example:AdversarialExample):
    n = adv_example.vertex_count
    perturbed_pos = adv_example.perturbed_pos
    stiff_r, area_r = mesh.laplacian.LB_v2(perturbed_pos, adv_example.faces)
    
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
      self.kNN = utils.kNN(
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