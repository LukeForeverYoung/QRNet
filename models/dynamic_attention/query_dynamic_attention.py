from torch import nn
import torch
import torch.nn.functional as F
from .Linear import DynamicLinear,MuModuleList

def logsumexp(tensor):
    #tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor, dim=2, keepdim=True)
    outputs = s + (tensor - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate(nn.Module):
    def __init__(self, gate_channels,text_dim, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = MuModuleList([
            DynamicLinear(gate_channels, gate_channels // reduction_ratio,text_dim),
            nn.ReLU(),
            DynamicLinear(gate_channels // reduction_ratio, gate_channels,text_dim)
        ])
        self.pool_types = pool_types
    def forward(self, x ,mu):
        B=x.shape[0] # batchsize
        D=x.shape[-1] # dimension
        channel_att_sum = None
        for pool_type in self.pool_types:
            pre_pool=x.view(B,-1,D)
            if pool_type=='avg':
                avg_pool=torch.mean(pre_pool,dim=1)
                channel_att_raw = self.mlp( avg_pool ,mu)
            elif pool_type=='max':
                max_pool=torch.max(pre_pool,dim=1).values
                channel_att_raw = self.mlp( max_pool ,mu)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum )
        scale = scale.view(((scale.shape[0],)+(1,)*(len(x.size())-2)+(scale.shape[-1],)))

        return x * scale


class SpatialGate(nn.Module):
    def __init__(self,gate_channels,mu_dim):
        super(SpatialGate, self).__init__()
        self.spatial = DynamicLinear(gate_channels,1,mu_dim)
    def forward(self, x, mu):
        assert len(x.size())>2 # B spatial D

        x_out = self.spatial(x,mu)
        scale = torch.sigmoid(x_out) # broadcasting
        res=x*scale
        return res

    
class QueryDynamicAttention(nn.Module):
    def __init__(self,gate_channels,mu_dim, reduction_ratio, pool_types,use_spatial=True,use_channel=True):
        super(QueryDynamicAttention,self).__init__()
        self.ChannelGate = ChannelGate(gate_channels,mu_dim, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(gate_channels,mu_dim)
        self.use_spatial=use_spatial
        self.use_channel=use_channel
    def forward(self, x,mu):
        if self.use_channel:
            x = self.ChannelGate(x,mu)
        if len(x.size())<=2:
            return x
        if self.use_spatial:
            x = self.SpatialGate(x,mu)
        return x

if __name__=='__main__':
    pass