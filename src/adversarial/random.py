import random
from collections import namedtuple

import tqdm
from torch_geometric.data import  Dataset

import utils
import mesh.transforms

def find_minimal_random(y, pos, perturb_func, model, iterations=20):
  def _perturbed_pos(alpha):
    return perturb_func(pos, alpha)

  def _perturbed(alpha):
    z = model(_perturbed_pos(alpha))
    _, argmax = torch.max(z, 0)
    return argmax != y

  def _recursive_expansion(alpha, it):
    perturbed = _perturbed(alpha)

    if perturbed :
      return True, alpha, it
    elif it == 0:
      return False, alpha, it
    else: 
      return _recursive_expansion(alpha*2, it-1)

  def _recursive_search(alpha, m, M, it):
    perturbed = _perturbed(alpha)
    if it == 0 : return perturbed, alpha 

    if not perturbed: 
      return _recursive_search( (alpha+M)/2, alpha, M, it-1)
    else:
      perturbed, result_alpha = _recursive_search( (m+alpha)/2, m, alpha, it-1)
      alpha =  result_alpha if perturbed else alpha
      return True, alpha
    
  perturbed, alpha, it = _recursive_expansion(1, iterations)
  if not perturbed:
    return False, alpha, _perturbed_pos(alpha)
  else:
    perturbed, result_alpha  = _recursive_search(alpha/2, 0, alpha, it-1)
    alpha =  result_alpha if perturbed else alpha
    return True, alpha, _perturbed_pos(alpha)


def mass_random_generations(
    num_samples:int, 
    model:torch.nn.Module,
    data:Dataset,
    device="cpu",K=300):
  N = len(data)
  num_samples = min(N, num_samples)
  perm = list(range(N))
  random.shuffle(perm)
  num_misclassified = 0
  num_incorrect = 0

  # start perturbations
  bar = tqdm.tqdm_notebook(total=num_samples,)
  for i in range(N):
    sample = perm[i]
    x = mesh.transforms.transform_translation_(data[sample].pos.to(device))
    f = data[sample].face.t().to(device)
    e = data[sample].edge_index.t().to(device)
    y = data[sample].y.to(device)
    Z = model(x).clone().cpu().detach().numpy().flatten()
    ypred = Z.argmax(axis=0)
    
    if ypred != y: 
      num_incorrect += 1
    else:    
      print("-----------------------------------------------------------------------")
      print("processing sample: {}".format(sample))
          
      # create perturbation
      eigvals, eigvecs = utils.eigenpairs(x.to(torch.float64), faces=f, K=K)
      eigvals = eigvals.to(x.dtype)
      eigvecs = eigvecs.to(x.dtype)

      r = torch.normal(0,1, size=[K, 3], device=x.device, dtype=x.dtype)
      r = torch.nn.functional.normalize(r, p=2, dim=0)
      _, (_,a) = mesh.laplacian.LB_v2(x,f)
      rcoeff = eigvecs.t().mm(torch.diag(a).mm(x))

      def pert(pos, alpha):
        return pos + eigvecs.matmul(rcoeff*r)*alpha

      misclassified, alpha, px = find_minimal_random(y, x, pert, model, iterations=100)
      num_misclassified += 1 if misclassified else 0
  bar.close()
  return num_incorrect, num_misclassified