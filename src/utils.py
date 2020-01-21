import torch
import networkx as nx

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
  neighbors_num:int=256):

  if len(pos.shape)!= 2 or pos.shape[1] != 3 or pos.dtype != torch.float:
      raise ValueError("The vertices matrix must have shape [n,3] and type float!")
  if len(edges.shape) != 2 or edges.shape[1] != 2 or edges.dtype != torch.long:
      raise ValueError("The edge index matrix must have shape [m,2] and type long!")

  n = pos.shape[0]
  m = edges.shape[0]
  k = neighbors_num

  graph = nx.Graph()
  graph.add_nodes_from(range(n))
  graph.add_edges_from(edges.numpy())

  N = torch.Tensor([n,k])
  for node_index in range(n):
    spiral = nx.single_source_shortest_path_length(G, node, cutoff=K)
    N[node_index, :] = spiral
  return N


 

