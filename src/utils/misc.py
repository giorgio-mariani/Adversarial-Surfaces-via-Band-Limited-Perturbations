import networkx as nx
import numpy as np
import torch
import torch_sparse as tsparse
import torch_scatter as tscatter
import torch_geometric
import tqdm

from . import eigenpairs

def check_data(pos:torch.Tensor=None, edges:torch.Tensor=None, faces:torch.Tensor=None, float_type:type=None):
    # check input consistency 

    if pos is not None:
      if not torch.is_floating_point(pos): 
        raise ValueError("The vertices matrix must have floating point type!")
    
      if float_type is None: float_type = pos.dtype

      if (len(pos.shape)!= 2 or pos.shape[1] != 3) and pos.dtype != float_type:
        raise ValueError("The vertices matrix must have shape [n,3] and type {}!".format(float_type))

    if edges is not None and (len(edges.shape) != 2 or edges.shape[1] != 2 or edges.dtype != torch.long):
      raise ValueError("The edge index matrix must have shape [m,2] and type long!")
    
    if faces is not None and (len(faces.shape) != 2 or faces.shape[1] != 3 or faces.dtype != torch.long):
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

def compute_dd_mse(pos, perturbed_pos, faces, K, t):
    eigvals1, eigvecs1 = eigenpairs(pos, faces, K)
    eigvals2, eigvecs2 = eigenpairs(perturbed_pos, faces, K)
    d1 = diffusion_distance(eigvals1,eigvecs1,t)
    d2 = diffusion_distance(eigvals2,eigvecs2,t)
    return torch.nn.functional.mse_loss(d1, d2)

#----------------------------------------------------------------------------------
def tri_areas(pos, faces):
    check_data(pos=pos, faces=faces)
    v1 = pos[faces[:, 0], :]
    v2 = pos[faces[:, 1], :]
    v3 = pos[faces[:, 2], :]
    v1 = v1 - v3
    v2 = v2 - v3
    return torch.norm(torch.cross(v1, v2, dim=1), dim=1) * .5

def pos_areas(pos, faces): #TODO check correctness
  check_data(pos=pos, faces=faces)
  n = pos.shape[0]
  m = faces.shape[0]
  triareas = tri_areas(pos, faces)/3
  posareas = torch.zeros(size=[n, 1], device=triareas.device, dtype=triareas.dtype)
  for i in range(3):
    posareas += tscatter.scatter_add(triareas, faces[:,i])
  return posareas

def least_square_meshes(pos, edges):
    check_data(pos=pos, edges=edges)
    laplacian = torch_geometric.utils.get_laplacian(edges.t(), normalization="rw")
    n = pos.shape[2]
    tmp = tsparse.spmm(*laplacian, n, n, pos) #Least square Meshes problem 
    return (tmp**2).sum()


#---------------------------------------
def drop_resolution(pos, downsampling_indices) -> torch.Tensor:
  pos = pos[downsampling_indices]
  return 
