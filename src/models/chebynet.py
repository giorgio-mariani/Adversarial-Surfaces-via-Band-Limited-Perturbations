from typing import List

import torch.nn
import torch.nn.functional as func
import torch.sparse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import  ChebConv
from torch_geometric.data import Data

class ChebClassifier(torch.nn.Module):
    def __init__(
        self,
        param_conv_layers:List[int],
        dense_input_nodes:int, 
        num_classes:int,
        K=6):
        """
        arguments:
         * param_conv_layers: number of output features for the all the convolutional
                              layers (the output features for layer i are the input features
                              of layer i+1), the input features of the first conv. layer are
                              assumed to be 3 (position xyz of node)
         * dense_input_nodes: number of nodes after all pooling operations, this value is 
                              used to determine the number of input features for the final
                              dense layer after convolution.
         * num_classes: number of output classes of the classifier.
        """

        super(ChebClassifier, self).__init__()

        # convolutional layers
        param_conv_layers.insert(0,3) # add the first input features
        self.conv = []
        for i in range(len(param_conv_layers)-1):
            self.conv.append(ChebConv(
                param_conv_layers[i],
                param_conv_layers[i+1],
                K = K))

        # dense layer
        self.linear= torch.nn.Linear(
            dense_input_nodes*param_conv_layers[-1], 
            num_classes)

    def forward(
        self,
        data:Data,
        E_t:List[torch.Tensor],
        D_t:List[torch.sparse.FloatTensor]):

        # assert consistency of D_t and E_t with the convolution layers
        if len(E_t) != len(D_t):
            raise ValueError("edge-index list doesn't have the same length as the downscale matrices list.")
        if len(E_t) != len(self.conv) - 1 != len(D_t):
            raise ValueError("the edge-index/downscale-matrix lists must have the same length as the number of convlutional layer minus 1.")
        if data.pos.shape[-1] != self.conv[0].in_channels:
            raise ValueError("input data shape is not compatible with the first convolutional layer.")
        if D_t[-1].shape[0]*self.conv[-1].out_channels != self.linear.in_features:
            raise ValueError("input data shape is not compatible with the final dense layer.")

        # x has shape [num_nodes, num_dimensions], 
        x = data.pos
        # NOTE _indices() is not optimal, but should work
        edge_indices = [data.edge_index] + [E._indices() for E in E_t] # edge_indices is a list of tensor of shape [2, num_edges (at scale i)] 
        pool_number = len(D_t)
 
        # apply chebyshev convolution and pooling layers
        for i in range(pool_number):
            x = func.relu(self.conv[i](x, edge_indices[i]))
            x = pool(x, D_t[i])

        # last convolution and dense layer
        x = self.conv[i+1](x, edge_indices[i+1])
        Z = self.linear(x.view(-1)) #flatten and apply dense layer
        return Z #return the logits


def pool(x:torch.Tensor, downscale_mat:torch.sparse.FloatTensor):
        return torch.sparse.mm(downscale_mat, x)


class StupidNet(torch.nn.Module):
    """
    This class consists of a simple two layers neural network.

    This network is used as a baseline for other more sophisticated approaches, such as spiralnet++ or
    ChebyNet
    """
    def __init__(self, in_features, num_classes:int, hidden_layer_features=512, dropout=0):
        super(StupidNet, self).__init__()
        
        self.linear1= torch.nn.Linear(in_features, hidden_layer_features)#
        self.linear2= torch.nn.Linear(hidden_layer_features, num_classes)#
        self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, 
        data:Data,
        E_t:List[torch.Tensor],
        D_t:List[torch.sparse.FloatTensor]):

        # x has shape [num_nodes, num_dimensions], #edge_index1 [2, num_edges]
        x, edge_index1 = data.pos, data.edge_index
        x = x.view(-1) # flatten
        h = func.relu(self.linear1(x))
        h = self.dropout(h)
        return self.linear2(h)


def pool(x:torch.Tensor, downscale_mat:torch.sparse.FloatTensor):
        return torch.sparse.mm(downscale_mat, x)
