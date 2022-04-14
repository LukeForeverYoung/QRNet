from torch import nn
import torch
from einops.einops import rearrange, repeat
from .util_models import Initializer

class MuModuleList(nn.ModuleList):
    def forward(self,x,mu):
        for layer in self:
            if type(layer) == DynamicLinear:
                x=layer(x,mu)
            else:
                x=layer(x)
        return x

class ParamDecoder(nn.Module):
    def __init__(self, mu_dim, need_in_dim,need_out_dim,k=30):
        super(ParamDecoder, self).__init__()
        self.need_in_dim=need_in_dim
        self.need_out_dim=need_out_dim
        self.k=k
        self.decoder = nn.Linear(mu_dim, need_in_dim*k) 
        self.V = nn.parameter.Parameter(torch.zeros(k,need_out_dim))
      
    def forward(self, t_feat):
        B=t_feat.shape[0]
        U = self.decoder(t_feat).reshape(B,self.need_in_dim,self.k)  # B x need_in_dim x k
        param=torch.einsum('bik,kj->bij',U,self.V).reshape(B,-1)
        return param

class DynamicLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mu_dim: int, bias=True):
        super(DynamicLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mu_dim = mu_dim
        self.bias=bias
        self.decoder = ParamDecoder(mu_dim, in_dim + 1, out_dim)
    def forward(self, x, mu):

        param=rearrange(self.decoder(mu),'B (dim_A dim_B) -> B dim_A dim_B',dim_A=self.in_dim+1,dim_B=self.out_dim)
        weight=param[:,:-1,:]
        bias=param[:, -1, :]
        x=torch.einsum('b...d,bde->b...e',x,weight)
        if self.bias:
            bias=bias.view(((bias.shape[0],)+(1,)*(len(x.size())-2)+(bias.shape[-1],)))
            x=x+bias
        return x

class MuFCNet(nn.Module):
    """Simple class for non-linear fully connect network"""
    
    def __init__(self, dims, mu_dim, act="ReLU", last_act=False, dropout=0):
        super(MuFCNet, self).__init__()

        layers = []

        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(DynamicLinear(in_dim, out_dim, mu_dim))
            if "" != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(DynamicLinear(dims[-2], dims[-1], mu_dim))
        if "" != act and last_act:
            layers.append(getattr(nn, act)())
        self.layers = MuModuleList(layers)
        # self.main = nn.Sequential(*layers)
        self.apply(Initializer.xavier_normal)

    def forward(self, x, mu):
        return self.layers(x,mu)

if __name__ == "__main__":
    from icecream import ic

    N = 5
    S=10
    B = 200
    in_dim = 256
    out_dim = 768
    mu_dim = 768
    text = torch.randn(B, mu_dim)

    ml = DynamicLinear(in_dim,out_dim,mu_dim)
    visual = torch.randn(B,S, in_dim)
