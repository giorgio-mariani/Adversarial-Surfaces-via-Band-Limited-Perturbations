import torch
import tqdm
import networkx as nx
import numpy as np

from . import eigenpairs

def check_data(pos:torch.Tensor, edges:torch.Tensor, faces:torch.Tensor, float_type:type=torch.double):
    # check input consistency 
    if len(pos.shape)!= 2 or pos.shape[1] != 3 or pos.dtype != float_type:
      raise ValueError("The vertices matrix must have shape [n,3] and type float!")
    if len(edges.shape) != 2 or edges.shape[1] != 2 or edges.dtype != torch.long:
      raise ValueError("The edge index matrix must have shape [m,2] and type long!")
    if len(faces.shape) != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
      raise ValueError("The edge index matrix must have shape [#faces,3] and type long!")

def prediction(classifier:torch.nn.Module, x:torch.Tensor):
  Z = classifier(x)
  prediction = Z.argmax()
  return prediction

def kNN(
  pos:torch.Tensor, 
  edges:torch.LongTensor,
  neighbors_num:int=256,
  cutoff:int=3):
  device = pos.device

  if len(pos.shape)!= 2 or pos.shape[1] != 3:
      raise ValueError("The vertices matrix must have shape [n,3] and type float!")
  if len(edges.shape) != 2 or edges.shape[1] != 2 or edges.dtype != torch.long:
      raise ValueError("The edge index matrix must have shape [m,2] and type long!")

  n = pos.shape[0]
  m = edges.shape[0]
  k = neighbors_num
  edge_index = edges.cpu().clone().detach().numpy() # they are all necessary unfortunately

  graph = nx.Graph()
  graph.add_nodes_from(range(n))
  graph.add_edges_from(edge_index)

  N = np.zeros([n,k], dtype=float)
  spiral = nx.all_pairs_shortest_path(graph, cutoff=cutoff)
  for node_idx, neighborhood in spiral:

    if len(neighborhood) < k:
      raise RuntimeError("Node {} has only {} neighbours, increase cutoff value!".format(node_idx, len(neighborhood)))

    for i, neighbour_idx in enumerate(neighborhood.keys()):
      if i >= k: break
      else: N[node_idx, i] = neighbour_idx
    
  node_neighbours_matrix = torch.tensor(N, device=device, dtype=torch.long)
  return node_neighbours_matrix


#-------------------------------------------------------------------------------------------------


def heat_kernel(eigvals:torch.Tensor, eigvecs:torch.Tensor, t:float) -> torch.Tensor:
    #hk = eigvecs.matmul(torch.diag(torch.exp(-t*eigvals)).matmul(eigvecs.t()))
    tmp = torch.exp(-t*eigvals).view(1,-1)
    hk = (tmp*eigvecs).matmul(eigvecs.t())
    return hk

def diffusion_distance(eigvals:torch.Tensor, eigvecs:torch.Tensor, t:float):
    n, k = eigvecs.shape
    device = eigvals.device
    dtype = eigvals.dtype
    
    hk = heat_kernel(eigvals, eigvecs,2*t)
    tmp = torch.diag(hk).repeat(n, 1)
    return tmp + tmp.t() -2*hk


def compute_distance_mse(pos, perturbed_pos, faces, K, t):
    eigvals1, eigvecs1 = eigenpairs(pos, faces, K)
    eigvals2, eigvecs2 = eigenpairs(perturbed_pos, faces, K)
    d1 = diffusion_distance(eigvals1,eigvecs1,t)
    d2 = diffusion_distance(eigvals2,eigvecs2,t)
    return torch.nn.functional.mse_loss(d1, d2)


#-------------------------------
import torch_sparse as tsparse

def LB_distortion(pos, perturbed_pos, faces, stiff, area, perturbed_stiff, perturbed_area):
  n = pos.shape[0]
  ai, av = area
  ai_r, av_r = perturbed_area
  _,L = tsparse.spspmm(ai, torch.reciprocal(av), *stiff, n, n, n)
  _,perturbed_L = tsparse.spspmm(ai_r, torch.reciprocal(av_r), *perturbed_stiff, n, n, n)
  return torch.nn.functional.smooth_l1_loss(L, perturbed_L)


def MC_distortion(pos, perturbed_pos, stiff, area, perturbed_stiff, perturbed_area):
  n = pos.shape[0]
  tmp = tsparse.spmm(*stiff, n, n, pos)
  perturbed_tmp = tsparse.spmm(*perturbed_stiff, n, n, perturbed_pos)
  
  ai, av = area
  ai_r, av_r = perturbed_area

  mcf = tsparse.spmm(ai, torch.reciprocal(av), n, n, tmp)
  perturbed_mcf = tsparse.spmm(ai_r, torch.reciprocal(av_r), n, n, perturbed_tmp)
  diff_norm = torch.norm(mcf - perturbed_mcf,p=2,dim=-1)
  norm_integral = torch.dot(av, diff_norm)
  return norm_integral

def L2_distortion(pos, perturbed_pos, area, perturbed_area):
  return torch.norm(perturbed_pos - pos, p=2, dim=-1).sum()



def pprint_tree(tensor:torch.Tensor, file=None, _prefix="", _last=True):
    print(_prefix, "`- " if _last else "|- ", str(tensor) , sep="", file=file)
    _prefix += "   " if _last else "|  "
    
    if hasattr(tensor, "grad_fn"):
        child_count = len(tensor.grad_fn.next_functions)
        for i, (child, _) in enumerate(tensor.grad_fn.next_functions):
            _last = i == (child_count - 1)
            if child is not None:
                pprint_tree(child, file, _prefix, _last)
    elif hasattr(tensor, "next_functions"):
        child_count = len(tensor.next_functions)
        for i, (child, _) in enumerate(tensor.next_functions):
            _last = i == (child_count - 1)
            if child is not None:
                pprint_tree(child, file, _prefix, _last)
