import torch
import tqdm
import networkx as nx
import numpy as np
import scipy
import scipy.sparse.linalg  as slinalg

from mesh.laplacian import laplacebeltrami_FEM
from mesh.laplacian import LB_v2


def check_data(pos:torch.Tensor, edges:torch.Tensor, faces:torch.Tensor, float_type:type=torch.double):
    # check input consistency 
    if len(pos.shape)!= 2 or pos.shape[1] != 3 or pos.dtype != float_type:
      raise ValueError("The vertices matrix must have shape [n,3] and type float!")
    if len(edges.shape) != 2 or edges.shape[1] != 2 or edges.dtype != torch.long:
      raise ValueError("The edge index matrix must have shape [m,2] and type long!")
    if len(faces.shape) != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
      raise ValueError("The edge index matrix must have shape [#faces,3] and type long!")


def get_spirals(
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
def eigenpairs(pos:torch.Tensor, faces:torch.Tensor, K:int):
    if pos.shape[-1] != 3:
        raise ValueError("Vertices positions must have shape [n,3]")
    if faces.shape[-1] != 3:
        raise ValueError("Face indices must have shape [m,3]") 
  
    stiff, area, lump = laplacebeltrami_FEM(pos, faces)
    #stiff, area = LB_v2(pos, faces)
    
    n = pos.shape[0]
    device = pos.device
    dtype = pos.dtype

    stiff = stiff.coalesce()
    area = area.coalesce()

    si, sv = stiff.indices().cpu(), stiff.values().cpu()
    ai, av = area.indices().cpu(), area.values().cpu()

    ri,ci = si
    S = scipy.sparse.csr_matrix( (sv, (ri,ci)), shape=(n,n))

    ri,ci = ai
    A = scipy.sparse.csr_matrix( (av, (ri,ci)), shape=(n,n))

    #A_lumped = scipy.sparse.csr_matrix( (lump, (range(n),range(n))), shape=(n,n))

    eigvals, eigvecs = slinalg.eigsh(S, M=A, k=K, sigma=-1e-6)
    eigvals = torch.tensor(eigvals, device=device, dtype=dtype)
    eigvecs = torch.tensor(eigvecs, device=device, dtype=dtype)
    return eigvals, eigvecs

def heat_kernel(eigvals:torch.Tensor, eigvecs:torch.Tensor, t:float) -> torch.Tensor:
    #hk = eigvecs.matmul(torch.diag(torch.exp(-t*eigvals)).matmul(eigvecs.t()))
    tmp = torch.exp(-t*eigvals).view(1,-1)
    hk = (tmp*eigvecs).matmul(eigvecs.t())
    return hk

def diffusion_distance(eigvals:torch.Tensor, eigvecs:torch.Tensor, t:float):
    n, k = eigvecs.shape
    device = eigvals.device
    dtype = eigvals.dtype
    D = torch.zeros([n,n], device=device, dtype=dtype)
    for i in tqdm.trange(k):
        eigvec = eigvecs[:,i].view(-1,1)
        eigval = eigvals[i]
        tmp = eigvec.repeat(1, n)
        tmp = tmp - tmp.t()
        D = D + torch.exp(-2*t*eigval)*(tmp*tmp)
    return D

def compute_distance_mse(pos, perturbed_pos, faces, K, t):
    eigvals1, eigvecs1 = eigenpairs(pos, faces, K)
    eigvals2, eigvecs2 = eigenpairs(perturbed_pos, faces, K)
    d1 = diffusion_distance(eigvals1,eigvecs1,t)
    d2 = diffusion_distance(eigvals2,eigvecs2,t)
    return torch.nn.functional.mse_loss(d1, d2)