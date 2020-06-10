from typing import Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as func
import torch_sparse as tsparse
from torch_geometric.data.data import Data
import scipy.sparse

import utils
from adversarial.base import AdversarialExample, Builder, LossFunction

#------------------------------------------------------------------------------
class ValueLogger(object):
  def __init__(self, value_functions:dict=None, log_interval:int=10):
    super().__init__()
    self.logged_values =  dict()
    self.value_functions = value_functions if value_functions is not None else dict()
    self.log_interval = log_interval

    #initialize logging metrics
    for n, f in self.value_functions.items():
        self.logged_values[n] = []

    for n,f in self.value_functions.items():
      self.add_value_name(n,f)

  def reset(self):
    for func, values in self.logged_values.items():
      values.clear()

  def log(self, adv_example, iteration):
    if self.log_interval != 0 and iteration % self.log_interval == 0:
      for n,f in self.value_functions.items():
        v = f(adv_example).item()
        self.logged_values[n].append(v)

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
    target:int, #NOTE can be None
    adversarial_coeff:float,
    regularization_coeff:float,
    minimization_iterations:int,
    learning_rate:float,    
    additional_model_args:dict,
    logger:ValueLogger=ValueLogger()):

    super().__init__(
        pos=pos, edges=edges, faces=faces,
        classifier=classifier, target=target,
        classifier_args=additional_model_args)
    # coefficients
    self.adversarial_coeff = torch.tensor([adversarial_coeff], device=self.device, dtype=self.dtype_float)
    self.regularization_coeff = torch.tensor([regularization_coeff], device=self.device, dtype=self.dtype_float)
    
    # other parameters
    self.minimization_iterations = minimization_iterations
    self.learning_rate = learning_rate
    self.logger = logger
    self.model_args = additional_model_args

    # for untargeted-attacks use second most probable class    
    if target is None:
      values, index = torch.sort(self.logits, dim=0)
      self._target = index[-2]

    # class components
    self.perturbation = None
    self.adversarial_loss = None
    self.similarity_loss = None
    self.regularization_loss = None

  @property
  def perturbed_pos(self):
    return self.perturbation.perturb_positions()

  def compute(self, usetqdm:str=None, patience=3):
    # reset variables
    self.perturbation.reset()
    self.logger.reset() #ValueLogger({"adversarial":lambda x:x.adversarial_loss()})
    
    # compute gradient w.r.t. the perturbation
    optimizer = torch.optim.Adam([self.perturbation.r], lr=self.learning_rate,betas=(0.5,0.75))

    if usetqdm is None or usetqdm == False:
      iterations =  range(self.minimization_iterations)
    elif usetqdm == "standard" or usetqdm == True:
      iterations = tqdm.trange(self.minimization_iterations)
    elif usetqdm == "notebook":
      iterations = tqdm.tqdm_notebook(range(self.minimization_iterations))
    else:
      raise ValueError("Invalid input for 'usetqdm', valid values are: None, 'standard' and 'notebook'.")
    
    flag, counter = False, patience
    last_r = self.perturbation.r.data.clone();

    for i in iterations:
      # compute loss
      optimizer.zero_grad()
      
      # compute total loss
      similarity_loss =  self.similarity_loss()
      adversarial_loss = self.adversarial_coeff*self.adversarial_loss()
      regularization_loss = 0 if self.regularization_loss is  None else self.regularization_coeff*self.regularization_loss()
      loss = adversarial_loss + similarity_loss + regularization_loss
      self.logger.log(self, i) #log results

      # backpropagate
      loss.backward()
      optimizer.step()

      # cutoff procedure to improve performance
      is_successful = adversarial_loss <= 0
      if is_successful: # NOTE problem here (this operation costs too much)
        counter -= 1
        if counter<=0:
            last_r.data = self.perturbation.r.data.clone()
            flag= True
      else: 
        counter = patience

      if flag and not is_successful:
        self.perturbation.r.data = last_r
        break;     # cutoff policy used to speed-up the tests



class CWBuilder(Builder):
  USETQDM = "usetqdm"
  ADV_COEFF = "adversarial_coeff"
  REG_COEFF = "regularization_coeff"
  MIN_IT = "minimization_iterations"
  LEARN_RATE = "learning_rate"
  MODEL_ARGS = "additional_model_args"

  def __init__(self, search_iterations=1):
    super().__init__()
    self.search_iterations = search_iterations
    self.logger = None

    self._perturbation_factory = LowbandPerturbation
    self._adversarial_loss_factory = AdversarialLoss
    self._similarity_loss_factory = L2Similarity
    self._regularizer_factory = lambda x: None

  def set_perturbation(self, perturbation_factory):
    self._perturbation_factory = perturbation_factory
    return self

  def set_adversarial_loss(self, adv_loss_factory):
    self._adversarial_loss_factory = adv_loss_factory
    return self

  def set_similarity_loss(self, sim_loss_factory):
    self._similarity_loss_factory = sim_loss_factory
    return self

  def set_regularization_loss(self, regularizer_factory):
    self._regularizer_factory = regularizer_factory
    return self
  
  def set_logger(self, logger:ValueLogger):
    self.logger = logger
    return self

  def build(self, **args:dict) -> AdversarialExample:
    usetqdm = args.get(CWBuilder.USETQDM, False)
    self.adex_data[CWBuilder.MIN_IT] = args.get(CWBuilder.MIN_IT, 500)
    self.adex_data[CWBuilder.ADV_COEFF] = args.get(CWBuilder.ADV_COEFF, 1)
    self.adex_data[CWBuilder.REG_COEFF] = args.get(CWBuilder.REG_COEFF, 1)
    self.adex_data[CWBuilder.LEARN_RATE] = args.get(CWBuilder.LEARN_RATE, 1e-3)
    self.adex_data[CWBuilder.MODEL_ARGS] = args.get(CWBuilder.MODEL_ARGS, dict())
    start_adv_coeff = self.adex_data[CWBuilder.ADV_COEFF]
    
    # exponential search variable
    range_min, range_max = 0, start_adv_coeff
    optimal_example = None 
    exp_search = True # flag used to detected whether it is the 
                      # first exponential search phase, or the binary search phase

    # start search
    for i in range(self.search_iterations):
      midvalue = (range_min+range_max)/2
      c = range_max if exp_search else midvalue 

      print("[{},{}] ; c={}".format(range_min, range_max, c))
      
      # create adversarial example
      self.adex_data[CWBuilder.ADV_COEFF] = c #NOTE non-consistent state during execution (problematic during concurrent programming)
      adex = CWAdversarialExample(**self.adex_data)
      
      adex.adversarial_loss = self._adversarial_loss_factory(adex)
      adex.perturbation = self._perturbation_factory(adex)
      adex.similarity_loss = self._similarity_loss_factory(adex)
      adex.regularization_loss = self._regularizer_factory(adex)
      adex.compute(usetqdm=usetqdm)

      # get perturbation
      r = adex.perturbation.r
      adex.adversarial_loss().item()

      # update best estimation
      if adex.is_successful:
        optimal_example = adex

      # update loop variables
      if exp_search and not adex.is_successful:
        range_min = range_max
        range_max = range_max*2
      elif exp_search and adex.is_successful:
        exp_search = False
      else:
        range_max = range_max if not adex.is_successful else midvalue
        range_min = midvalue  if not adex.is_successful else range_min

    # reset the adversarial example to the original state 
    self.adex_data[CWBuilder.ADV_COEFF] = start_adv_coeff 

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
      dtype=self.adv_example.dtype_float, 
      requires_grad=True)

  def perturb_positions(self):
    pos, r = self.adv_example.pos, self.r
    return pos + r

class LowbandPerturbation(object):
  def __init__(self, adv_example, eigs_num=50):
    super().__init__()
    self.r = None
    self.adv_example = adv_example
    self.eigs_num = eigs_num
    self.eigvals,self.eigvecs = utils.eigenpairs(
      self.adv_example.pos, self.adv_example.faces, K=eigs_num)
    self.reset()

  def reset(self):
    self.r = torch.zeros(
      [self.eigs_num, 3], 
      device=self.adv_example.device, 
      dtype=self.adv_example.dtype_float, 
      requires_grad=True)

  def perturb_positions(self):
    pos, r = self.adv_example.pos, self.r
    return pos + self.eigvecs.matmul(r)

#===============================================================================
# adversarial losses ----------------------------------------------------------
class AdversarialLoss(LossFunction):
    def __init__(self, adv_example:AdversarialExample, k:float=0):
        super().__init__(adv_example)
        self.k = torch.tensor([k], device=adv_example.device, dtype=adv_example.dtype_float)

    def __call__(self) -> torch.Tensor:
        ppos = self.adv_example.perturbed_pos
        Z = self.adv_example.classifier(ppos)
        values, index = torch.sort(Z, dim=0)
        argmax = index[-1] if index[-1] != self.adv_example.target else index[-2] # max{Z(i): i != target}
        Ztarget, Zmax = Z[self.adv_example.target], Z[argmax]
        return torch.max(Zmax - Ztarget, -self.k)

class ExponentialAdversarialLoss(LossFunction):
    def __init__(self, adv_example:AdversarialExample):
        super().__init__(adv_example)
    def __call__(self) -> torch.Tensor:
      raise NotImplementedError()

# regularizers ----------------------------------------------------------------
class CentroidRegularizer(LossFunction):
    def __init__(self, adv_example:AdversarialExample):
        super().__init__(adv_example)
    def __call__(self):
        adv_centroid = torch.mean(self.adv_example.perturbed_pos, dim=0)
        centroid = torch.mean(self.adv_example.pos, dim=0)
        return torch.nn.functional.mse_loss(adv_centroid, centroid)

# similarity functions --------------------------------------------------------
class L2Similarity(LossFunction):
    def __init__(self, adv_example:AdversarialExample):
        super().__init__(adv_example)
    
    def __call__(self) -> torch.Tensor:
        diff = self.adv_example.perturbed_pos - self.adv_example.pos
        area_indices, area_values = self.adv_example.area
        weight_diff = diff*torch.sqrt(area_values.view(-1,1)) # (sqrt(ai)*(xi-perturbed(xi)) )^2  = ai*(x-perturbed(xi))^2
        L2 = weight_diff.norm(p="fro") # this reformulation uses the sub-gradient (hance ensuring a valid behaviour at zero)
        return L2

class LocalEuclideanSimilarity(LossFunction):
    def __init__(self, adv_example:AdversarialExample, K:int=30):
        super().__init__(adv_example)
        self.neighborhood = K
        self.kNN = kNN(
            pos=self.adv_example.pos, 
            edges=self.adv_example.edges, 
            neighbors_num=self.neighborhood, 
            cutoff=5) # TODO try to find a way to automatically compute cut-off

    def __call__(self) -> torch.Tensor:
        n = self.adv_example.vertex_count
        pos = self.adv_example.pos
        ppos = self.adv_example.perturbed_pos

        flat_kNN = self.kNN.view(-1)
        X = pos[flat_kNN].view(-1, self.neighborhood, 3) # shape [n*K*3]
        Xr = ppos[flat_kNN].view(-1, self.neighborhood, 3)
        dist = torch.norm(X-pos.view(n,1,3), p=2,dim=-1)
        dist_r = torch.norm(Xr-ppos.view(n,1,3), p=2,dim=-1)
        dist_loss = torch.nn.functional.mse_loss(dist, dist_r, reduction="sum")
        return dist_loss

try:
  from knn_cuda import KNN
  class ChamferSimilarity(LossFunction):
      def __init__(self, adv_example:AdversarialExample):
          super().__init__(adv_example)
          self.knn = KNN(1, transpose_mode=True)

      def __call__(self):
        ppos = self.adv_example.perturbed_pos.view(1,-1,3)
        pos =self.adv_example.pos.view(1,-1,3)
        _, indx = knn(ref=pos, query=ppos)
        diff = ppos - pos[indx.view(-1)]
        term1 = diff.dot(diff).mean()

        _, indx = knn(ref=ppos, query=pos)
        diff = pos - ppos[indx.view(-1)]
        term2 = diff.dot(diff).mean()
        return term1 + term2

  class HausdorffSimilarity(LossFunction):
      def __init__(self, adv_example:AdversarialExample):
          super().__init__(adv_example)
          self.knn = KNN(1, transpose_mode=True)

      def __call__(self):
        ppos = self.adv_example.perturbed_pos.view(1,-1,3)
        pos =self.adv_example.pos.view(1,-1,3)
        _, indx = knn(ref=pos, query=ppos)
        diff = ppos - pos[indx.view(-1)]
        term1 = diff.dot(diff).max()

        _, indx = knn(ref=ppos, query=pos)
        diff = pos - ppos[indx.view(-1)]
        term2 = diff.dot(diff).max()
        return term1 + term2
except ImportError as e:
    pass

#==============================================================================
#------------------------------------------------------------------------------
def generate_adversarial_example(
    mesh:Data, classifier:Module, target:int,
    lowband_perturbation=True, 
    similarity_loss="local_euclidean", 
    regularization="none", **args) -> CWAdversarialExample:
    
    builder = CWBuilder().set_mesh(mesh.pos,mesh.edge_index.t(), mesh.face.t())
    builder.set_classifier(classifier).set_target(target)

    # set type of perturbation
    if lowband_perturbation:
      eigs_num = args["eigs_num"]
      builder.set_perturbation(perturbation_factory=lambda x:LowbandPerturbation(x,eigs_num=eigs_num))
    else:
      builder.set_perturbation(perturbation_factory=Perturbation)
    
    # set type of adversarial loss
    builder.set_adversarial_loss(adv_loss_factory=AdversarialLoss)

    # set type of similarity loss
    if similarity_loss == "local_euclidean":
      builder.set_similarity_loss(sim_loss_factory=LocalEuclideanSimilarity)
    elif similarity_loss == "l2":
      builder.set_similarity_loss(sim_loss_factory=L2Similarity)
    else: raise ValueError()

    # set type of regularizer
    if regularization != "none":
      if regularization == "centroid":
        builder.set_regularization_loss(regularizer_factory=CentroidRegularizer)
      else: raise ValueError()

    return builder.build(**args)
 