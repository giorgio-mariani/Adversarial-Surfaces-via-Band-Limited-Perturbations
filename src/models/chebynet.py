import os
from typing import List

import torch.nn
from torch.nn import Parameter
import torch.nn.functional as func
import torch.sparse
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data

class ChebnetClassifier(torch.nn.Module):
    def __init__(
        self,
        param_conv_layers:List[int],
        E_t:List[torch.Tensor],
        D_t:List[torch.sparse.FloatTensor],
        num_classes:int, 
        parameters_file=None,
        K=6, PNI=False):
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

        # add random noise to weights (if wanted)
        if PNI :
            chebconv = PNIChebConv
            linear = PNILinear
        else:
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



#PNI-------------------------------------------------------------------------------------

class PNIChebConv(torch_geometric.nn.ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization='sym',bias=True, **kwargs):
        self.pni_coefficients = torch.nn.Parameter(torch.zeros(K,in_channels, out_channels))
        super().__init__(in_channels=in_channels, out_channels=out_channels,
            K=K, normalization='sym', bias=True, **kwargs)
        
    def reset_parameters(self):
        super().reset_parametrs()
        self.pni_coefficients.tensor.data.fill_(0)

    def forward(self, x, edge_index, edge_weight=None, batch=None,
                lambda_max=None):
        """"""
        if self.normalization != 'sym' and lambda_max is None:
            raise ValueError('You need to pass `lambda_max` to `forward() in`'
                             'case the normalization is non-symmetric.')
        lambda_max = 2.0 if lambda_max is None else lambda_max

        edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                     self.normalization, lambda_max,
                                     dtype=x.dtype, batch=batch)

        # create white noise for the weights
        with torch.no_grad():
            std = self.weight.std().item()
            white_noise = self.weight.clone().normal_(0,std)
        
        # apply chebyshev convolution
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0] + self.pni_coefficients[0]*white_noise[0])

        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + torch.matmul(Tx_1, self.weight[1] + self.pni_coefficients[1]*white_noise[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k] + self.pni_coefficients[k]*white_noise[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

class PNILinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        self.pni_coefficients = Parameter(torch.Tensor(out_features, in_features))
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        super().reset_parameters()
        self.pni_coefficients.tensor.data.fill_(0)

    def forward(self, input):
        with torch.no_grad():
            std = self.weight.std().item()
            white_noise = self.weight.clone().normal_(0,std)

        return func.linear(input, self.weight+self.pni_coefficients*white_noise, self.bias)
