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
        E_t:List[torch.Tensor],
        D_t:List[torch.sparse.FloatTensor],
        num_classes:int, K=6):
        """
        arguments:
         * param_conv_layers: number of output features for the all the convolutional
                              layers (the output features for layer i are the input features
                              of layer i+1), the input features of the first conv. layer are
                              assumed to be 3 (position xyz of node)
         * num_classes: number of output classes of the classifier.
        """

        super(ChebClassifier, self).__init__()
        self.edge_indices = [E_t[i]._indices() for i in range(0,len(E_t))]
    
         # edge_indices is a list of tensor of shape [2, num_edges (at scale i)]
        self.downscale_matrices = [D for D in D_t]

        # convolutional layers
        param_conv_layers.insert(0,3) # add the first input features
        self.conv = []
        for i in range(len(param_conv_layers)-1):
            chebconv = ChebConv(
                param_conv_layers[i],
                param_conv_layers[i+1],
                K = K)
            self.conv.append(chebconv)
            self.add_module("chebconv_"+str(i), chebconv)


        # dense layer
        self.linear = torch.nn.Linear(
            self.downscale_matrices[-1].shape[0]*param_conv_layers[-1],
            num_classes)

    def forward(self, x:torch.Tensor):
        # apply chebyshev convolution and pooling layers
        for i in range(len(self.downscale_matrices)):
            x = func.relu(self.conv[i](x, self.edge_indices[i]))
            x = pool(x, self.downscale_matrices[i])

        # last convolution and dense layer
        x = self.conv[i+1](x, self.edge_indices[i+1])
        Z = self.linear(x.view(-1)) #flatten and apply dense layer
        return Z #return the logits

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        #move the downscaled matrices
        for i in range(len(self.downscale_matrices)):
            self.downscale_matrices[i] = self.downscale_matrices[i].to( *args, **kwargs)
        # move the edge indices
        for i in range(len(self.edge_indices)):
            self.edge_indices[i] = self.edge_indices[i].to( *args, **kwargs)

def pool(x:torch.Tensor, downscale_mat:torch.sparse.FloatTensor):
        return torch.sparse.mm(downscale_mat, x)
