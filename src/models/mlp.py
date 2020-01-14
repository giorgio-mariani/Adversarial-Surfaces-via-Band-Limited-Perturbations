import torch.nn
import torch.nn.functional as func
from torch_geometric.data import Data


class MultiLayerPerceptron(torch.nn.Module):
    """
    This class consists of a simple two layers neural network.

    This network is used as a baseline for other more sophisticated approaches, such as spiralnet++ or
    ChebyNet
    """
    def __init__(self, in_features:int, num_classes:int, hidden_layer_features:int=512, dropout:float=0):
        super().__init__()
        
        self.linear1= torch.nn.Linear(in_features, hidden_layer_features)#
        self.linear2= torch.nn.Linear(hidden_layer_features, num_classes)#
        self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, x:torch.Tensor):
        # x has shape [num_nodes, num_dimensions] 
        x = x.view(-1) # flatten
        h = func.relu(self.linear1(x))
        h = self.dropout(h)
        return self.linear2(h)
