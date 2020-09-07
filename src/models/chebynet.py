import os
from typing import List

import torch.nn
from torch.nn import Parameter
import torch.nn.functional as func
import torch_geometric
import torch_sparse

class ChebnetClassifier(torch.nn.Module):
    def __init__(
        self,
        param_conv_layers:List[int],
        E_t:List[torch.Tensor],
        D_t:List[torch.sparse.FloatTensor],
        num_classes:int, 
        parameters_file=None,
        K=6):
        """
        arguments:
         * param_conv_layers: number of output features for the all the convolutional
                              layers (the output features for layer i are the input features
                              of layer i+1), the input features of the first conv. layer are
                              assumed to be 3 (position xyz of node)
         * num_classes: number of output classes of the classifier.
        """

        super(ChebnetClassifier, self).__init__()
        self.edge_indices = [E_t[i]._indices() for i in range(0,len(E_t))]
    
         # edge_indices is a list of tensor of shape [2, num_edges (at scale i)]
        self.downscale_matrices = [D for D in D_t]

        chebconv = torch_geometric.nn.ChebConv
        linear = torch.nn.Linear

        # convolutional layers
        param_conv_layers.insert(0,3) # add the first input features
        self.conv = []
        for i in range(len(param_conv_layers)-1):
            cheblayer = chebconv(
                param_conv_layers[i],
                param_conv_layers[i+1],
                K = K)
            self.conv.append(cheblayer)
            self.add_module("chebconv_"+str(i), cheblayer)

        # dense layer
        self.linear = linear(
            self.downscale_matrices[-1].shape[0]*param_conv_layers[-1],
            num_classes)

        # load 
        if parameters_file is not None:
            if os.path.exists(parameters_file):
                self.load_state_dict(torch.load(parameters_file))
            else: 
                print("Warning parameters file {} is non-existent".format(parameters_file))

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

#------------------------------------------------------------------------------------------------
class ChebnetClassifier_SHREC14(torch.nn.Module):
    def __init__(
        self,
        nums_conv_units:List[int],
        num_classes:int,
        num_hidden:int=1024,
        parameters_file=None,
        K=6):
        """ Initialize class fields.

        Args:
            nums_conv_units (list of int): number of output features for the all the convolutional
                layers (the output features for layer i are the input features of layer i+1), the 
                input features of the first conv. layer are assumed to be 3 (coordinates xyz of node)
            num_classes (int): number of output classes of the classifier.
            num_dense_units (int): number of input units for last dense layer (output unites are the 
                number of classe)
        """

        super().__init__()
    
        # add random noise to weights (if wanted)
        chebconv = torch_geometric.nn.ChebConv
        linear = torch.nn.Linear

        # convolutional layers
        nums_conv_units.insert(0,3) # add the first input features
        self.conv = []
        for i in range(len(nums_conv_units)-1):
            cheblayer = chebconv(
                nums_conv_units[i],
                nums_conv_units[i+1],
                K = K)
            self.conv.append(cheblayer)
            self.add_module("chebconv_"+str(i), cheblayer)

        # dense layer
        self.linear_hidden = linear(nums_conv_units[-1], num_hidden)
        self.linear_logits = linear(num_hidden, num_classes)

        # load 
        if parameters_file is not None:
            if os.path.exists(parameters_file):
                self.load_state_dict(torch.load(parameters_file))
            else: 
                print("Warning parameters file {} is non-existent".format(parameters_file))

    #TODO convert input so to take only the positions
    def forward(self, pos:torch.Tensor, downscale_matrices, downscaled_edges):
        # apply chebyshev convolution and pooling layers
        h = pos 
        edge_indices = downscaled_edges
        
        for i in range(len(downscale_matrices)):
            # apply convolution
            h = func.relu(self.conv[i](h, edge_indices[i]))
            
            #pooling
            indices, values, size = downscale_matrices[i]
            h = torch_sparse.spmm(indices, values, size[0], size[1], h)

        # last convolution and dense layer
        h = self.conv[i+1](h, edge_indices[i+1])
        h, _ = self.linear_hidden(h).max(dim=0) #flatten and apply dense layer
        Z = self.linear_logits(h)
        return Z #return the logits

    
