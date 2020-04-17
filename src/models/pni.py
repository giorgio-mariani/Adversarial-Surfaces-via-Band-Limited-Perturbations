
import torch.nn
from torch.nn import Parameter
import torch.nn.functional as func
import torch_geometric

class PNIChebConv(torch_geometric.nn.ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization='sym',bias=True, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
            K=K, normalization='sym', bias=True, **kwargs)
        self.pni_coefficients = torch.nn.Parameter(torch.zeros(K,in_channels, out_channels))

        
    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "pni_coefficients"):
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
        super().__init__(in_features, out_features, bias)
        self.pni_coefficients = Parameter(torch.Tensor(out_features, in_features))

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "pni_coefficients"):
            self.pni_coefficients.tensor.data.fill_(0)

    def forward(self, input):
        with torch.no_grad():
            std = self.weight.std().item()
            white_noise = self.weight.clone().normal_(0,std)
        return func.linear(input, self.weight+self.pni_coefficients*white_noise, self.bias)

