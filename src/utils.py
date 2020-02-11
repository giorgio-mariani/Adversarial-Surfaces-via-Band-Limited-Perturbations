import torch
import networkx as nx
import numpy as np

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
  dtype = pos.dtype

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
    
  node_neighbours_matrix = torch.tensor(N, device=device, dtype=dtype)
  return node_neighbours_matrix


 

