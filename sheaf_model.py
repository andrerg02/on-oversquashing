import math
import inspect

from typing import Optional, Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_scatter import scatter_add
from torch_sparse import SparseTensor

from torch_householder import torch_householder_orgqr

from torch_geometric.nn import (
    MessagePassing,
    SGConv,
    SAGEConv,
    NNConv,
    GCNConv,
    GATConv,
    GPSConv,
    GraphSAGE,
    GINConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool)

from torch_geometric.utils import degree

TensorTriplet = Tuple[Tensor, Tensor, Tensor]
Linear = nn.Linear
Identity = nn.Identity

class SheafDiffusion(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, norm,
                 use_act=False, stalk_dimension=2, left_weights=True, right_weights=True,
                 use_eps=True, dropout=0.0, use_bias=False, sheaf_act='tanh',
                 orth_trans='householder', linear_emb=False, gnn_type='SAGE',
                 gnn_layers=1, gnn_hidden=32, gnn_residual=False, pe_size=0,):
        super(SheafDiffusion, self).__init__()
        assert stalk_dimension > 1

        self.dropout = dropout
        self.use_act = use_act

        self.lin1 = nn.Linear(in_channels, hidden_channels * stalk_dimension)

        self.readout = nn.Linear(hidden_channels * stalk_dimension, out_channels)

        self.layers = nn.ModuleList(
            [FlatBundleConv(in_channels=hidden_channels, out_channels=hidden_channels,
                            stalk_dimension=stalk_dimension, left_weights=left_weights,
                            right_weights=right_weights, use_act=use_act, use_eps=use_eps, dropout=dropout,
                            use_bias=use_bias, sheaf_act=sheaf_act, orth_trans=orth_trans,
                            linear_emb=linear_emb, gnn_type=gnn_type, gnn_layers=gnn_layers,
                            gnn_hidden=gnn_hidden, gnn_residual=gnn_residual, pe_size=pe_size)
             for _ in range(num_layers)]
        )

        if norm == 'batch':
            self.norm = nn.ModuleList(
                [nn.BatchNorm1d(hidden_channels * stalk_dimension) for _ in range(num_layers)]
            )
        elif norm == 'layer':
            self.norm = nn.ModuleList(
                [nn.LayerNorm(hidden_channels * stalk_dimension) for _ in range(num_layers)]
            )
        else:
            self.norm = nn.ModuleList(
                [nn.Identity() for _ in range(num_layers)]
            )

    def forward(self, x, edge_index):
        #x = x.float()
        #edge_index = edge_index.float() if edge_index is not None else None
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)

        if self.use_act:
            x = F.gelu(x)

        for layer, norm in zip(self.layers, self.norm):
            x, _ = layer(x, edge_index)
            x = norm(x)
            
        
        # if self.graph_level:
        #     x = global_mean_pool(x, batch)

        x = self.readout(x)

        return x

class FlatGenSheafConv(MessagePassing):
    r"""The flat bundle convolutional operator. The main model is from 
    the `"Neural Sheaf Diffusion: A Topological Perspective on Heterophily
    and Oversmoothing in GNNs" <https://arxiv.org/pdf/2202.04579>`_ paper,
    with the simplification of the sheaf structure proposed by the
    `"Bundle Neural Networks for message diffusion on graphs"
    <https://arxiv.org/pdf/2405.15540>`_ paper.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stalk_dimension (int, optional): Dimension of the sheaf stalks. (default :obj:`2`)
        left_weights (bool, optional): If True, applies left weights to the features. (default :obj:`True`)
        right_weights (bool, optional): If True, applies right weights to the features. (default :obj:`True`)
        use_eps (bool, optional): If True, uses the adjusted residual connection. (default :obj:`True`)
        dropout (float, optional): Dropout probability of the layer. (default :obj:`0.0`)
        use_bias (bool, optional): Add bias in the weights. (default :obj:`False`)
        sheaf_act (str, optional): Activation function applied on the sheaf maps. (default :obj:`'tanh'`)
        orth_trans (str, optional): Method to learn orthogonal maps. Options are
            :obj:`'householder'`, :obj:`'matrix_exp'`, :obj:`'cayley'`, or
            :obj:`'euler'`. The  :obj:`'euler'` method can only be used if stalk_dimension is 2 or 3. (default :obj:`'householder'`)
        linear_emb (bool, optional): Use a linear+act embedding/readout when learning the sheaf. (default :obj:`True`)
        gnn_type (str, optional): Type of GNN to use for learning the sheaf. Options are
            :obj:`'SAGE'`, :obj:`'GCN'`, :obj:`'GAT'`, :obj:`'NNConv'`, :obj:`'SGC'`, or :obj:`'SumGNN'`. (default :obj:`'SAGE'`)
        gnn_layers (int, optional): Number of GNN layers to use for learning the sheaf. (default :obj:`1`)
        gnn_hidden (int, optional): Number of hidden channels in the GNN layers. (default :obj:`32`)
        gnn_residual (bool, optional): Use residual connections in the GNN layers. (default :obj:`False`)
        pe_size (int, optional): Size of the positional encoding to use in the GNN layers. (default :obj:`0`)
    """

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 stalk_dimension: Optional[int]  = 2,
                 left_weights:    Optional[bool] = True,
                 right_weights:   Optional[bool] = True,
                 use_eps:         Optional[bool] = True,
                 dropout:         Optional[float] = 0.0,
                 use_bias:        Optional[bool] = False,
                 sheaf_act:       Optional[str]  = 'tanh',
                 orth_trans:      Optional[str]  = 'householder',
                 linear_emb:      Optional[bool] = True,
                 gnn_type:        Optional[str]  = 'SAGE',
                 gnn_layers:      Optional[int]  = 1,
                 gnn_hidden:      Optional[int]  = 32,
                 gnn_residual:    Optional[bool] = False,
                 pe_size:         Optional[int]  = 0,
                 ):
        MessagePassing.__init__(self, aggr='add',
                                flow='target_to_source',
                                node_dim=0)

        self.d = stalk_dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.right_weights = right_weights
        self.left_weights = left_weights
        self.use_eps = use_eps
        self.dropout = dropout
        self.orth_trans = orth_trans

        if in_channels != out_channels:
            assert right_weights, \
            f'The right_weights changes from in_channels to out_channels \
            Either set right_weights=True or ensure in_channels == out_channels.'

        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.in_channels,
                                               self.out_channels,
                                               bias=use_bias)
            nn.init.orthogonal_(self.lin_right_weights.weight.data)
        else:
            self.lin_right_weights = nn.Identity()

        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d,
                                              self.d,
                                              bias=use_bias)
            nn.init.eye_(self.lin_left_weights.weight.data)
        else:
            self.lin_left_weights = nn.Identity()
        
        self.sheaf_learner = FlatSheafLearner(
                self.d,
                self.in_channels,
                out_shape = (self.d**2,),
                linear_emb = linear_emb,
                gnn_type = gnn_type,
                gnn_layers = gnn_layers,
                gnn_hidden = gnn_hidden,
                gnn_residual = gnn_residual,
                pe_size = pe_size,
                sheaf_act = sheaf_act)
        
        if use_eps and in_channels == out_channels:
            self.epsilons = nn.Parameter(torch.zeros((self.d, 1)))
    
    def sheaf_effective_resistance(self, data, maps):
        """
        LG_pinv : (B,n,n)
        F_maps  : (B,n,d,d)
        Returns R_tot: (B,)
        """
        R = data.torch_R
        L_G_pinv = data.torch_L_G_pinv
        L_G_pinv = L_G_pinv.view(R.size(0), L_G_pinv.size(1), -1)
        B, n, d = R.size(0), L_G_pinv.size(1), maps.size(-1)
        ones = torch.ones(d, dtype=maps.dtype, device=maps.device)

        # S[b,i,:] solves F[b,i].T @ S[b,i,:] = 1
        S, *_ = torch.linalg.lstsq( maps.transpose(-1,-2), ones.expand(B*n,d) )  # (B,n,d)
        S = S.view(B,n,d)
        
        L_G_pinv = 0.5 * (L_G_pinv + L_G_pinv.transpose(-1, -2))
        evals, U = torch.linalg.eigh(L_G_pinv)                       # (B,n), (B,n,n)
        evals = torch.clamp(evals, min=0.0)                         # kill tiny negatives
        LG_pinv_psd = (U * evals.unsqueeze(-2)) @ U.transpose(-1, -2)

        Cprime = torch.matmul(S, S.transpose(-1,-2))           # (B,n,n)
        K = LG_pinv_psd * Cprime
        #K = 0.5 * (K.transpose(-2,-1) + K)                                # Hadamard

        trK  = K.diagonal(dim1=-2, dim2=-1).sum(-1)
        sumK = K.sum(dim=(-2,-1))
        Rtot = (n * trK - sumK).sum()
        #print(f"Effective Resistance: {Rtot} and shape {Rtot.shape}")
        return Rtot
        
    def batched_sym_matrix_pow(self, matrices: torch.Tensor, p: float) -> torch.Tensor:
        r"""
        Power of a matrix using Eigen Decomposition.
        Args:
            matrices: A batch of matrices.
            p: Power.
            positive_definite: If positive definite
        Returns:
            Power of each matrix in the batch.
        """
        # vals, vecs = torch.linalg.eigh(matrices)
        # SVD is much faster than  vals, vecs = torch.linalg.eigh(matrices) for large batches.
        vecs, vals, _ = torch.linalg.svd(matrices)
        good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
        vals = vals.pow(p).where(good, torch.zeros((), device=matrices.device, dtype=matrices.dtype))
        matrix_power = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
        return matrix_power
    
    def restriction_maps_builder(self, maps : Tensor, edge_index : Tensor):
        row, _ = edge_index

        maps = maps.view(-1, self.d, self.d)

        deg = degree(row, num_nodes=self.graph_size) + 1

        diag_maps = (maps.transpose(-2,-1) @ maps) * deg.view(-1, 1, 1)

        # if self.training:
        #     # During training, we perturb the matrices to ensure they have different singular values.
        #     # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
        #     eps = torch.FloatTensor(self.d).uniform_(-0.001, 0.001).to(device=self.device)
        # else:
        #     eps = torch.zeros(self.d, device=self.device)

        to_be_inv_diag_maps = diag_maps #+ torch.diag(1. + eps).unsqueeze(0) #if self.augmented else diag_maps
        diag_sqrt_inv = self.batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5)

        norm_D = (diag_sqrt_inv @ diag_maps @ diag_sqrt_inv).clamp(min=-1, max=1)

        return norm_D, maps, diag_sqrt_inv
    
    def left_right_linear(self, x: Tensor, left: Linear | Identity,
                          right: Linear | Identity) -> Tensor:
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, self.graph_size * self.d).t()

        x = right(x)
        return x
    
    def forward(self, x: Tensor, edge_index: Tensor, data=None, reff=False):
        self.graph_size = x.size(0)

        assert x.view(self.graph_size, -1).size(1) == self.in_channels * self.d, \
            f'Expected input size {self.in_channels * self.d}, got {x.view(self.graph_size, -1).size(1)}. \
            Are you embedding graph features into sheaf features?'
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_maps = x.reshape(self.graph_size, self.in_channels * self.d)

        maps = self.sheaf_learner(x_maps, edge_index)

        x = x.view(self.graph_size * self.d, -1)
        x0 = x

        D, maps, diag_sqrt_inv = self.restriction_maps_builder(maps, edge_index)

        x = self.left_right_linear(x, self.lin_left_weights, self.lin_right_weights)
        x = x.reshape(self.graph_size, self.d, self.out_channels)

        deg = degree(edge_index[0], num_nodes=self.graph_size) + 1
        Dx = D @ x * deg.pow(-1)[:, None, None]
        Fx = (maps @ diag_sqrt_inv).clamp(min=-1, max=1) @ x
        x = self.propagate(edge_index, x=Fx, diag=Dx, Ft=(diag_sqrt_inv @ maps.transpose(-2,-1)).clamp(min=-1, max=1))
        x = F.dropout(x, p=self.dropout, training=self.training)

        #if self.use_act:
        x = F.gelu(x)

        if self.use_eps and self.in_channels == self.out_channels:
            x = x.view(self.graph_size * self.d, -1)
            x0 = (1 + torch.tanh(self.epsilons).tile(self.graph_size, 1)) * x0 - x
            x = x0

        if reff:
            efec_res = self.sheaf_effective_resistance(data, maps.detach())

        return x.view(self.graph_size, -1), efec_res if reff else 0

    def message(self, x_j, diag_i, Ft_i):
        msg = Ft_i @ x_j

        return diag_i - msg

class FlatBundleConv(MessagePassing):
    r"""The flat bundle convolutional operator. The main model is from 
    the `"Neural Sheaf Diffusion: A Topological Perspective on Heterophily
    and Oversmoothing in GNNs" <https://arxiv.org/pdf/2202.04579>`_ paper,
    with the simplification of the sheaf structure proposed by the
    `"Bundle Neural Networks for message diffusion on graphs"
    <https://arxiv.org/pdf/2405.15540>`_ paper.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stalk_dimension (int, optional): Dimension of the sheaf stalks. (default :obj:`2`)
        left_weights (bool, optional): If True, applies left weights to the features. (default :obj:`True`)
        right_weights (bool, optional): If True, applies right weights to the features. (default :obj:`True`)
        use_eps (bool, optional): If True, uses the adjusted residual connection. (default :obj:`True`)
        dropout (float, optional): Dropout probability of the layer. (default :obj:`0.0`)
        use_bias (bool, optional): Add bias in the weights. (default :obj:`False`)
        sheaf_act (str, optional): Activation function applied on the sheaf maps. (default :obj:`'tanh'`)
        orth_trans (str, optional): Method to learn orthogonal maps. Options are
            :obj:`'householder'`, :obj:`'matrix_exp'`, :obj:`'cayley'`, or
            :obj:`'euler'`. The  :obj:`'euler'` method can only be used if stalk_dimension is 2 or 3. (default :obj:`'householder'`)
        linear_emb (bool, optional): Use a linear+act embedding/readout when learning the sheaf. (default :obj:`True`)
        gnn_type (str, optional): Type of GNN to use for learning the sheaf. Options are
            :obj:`'SAGE'`, :obj:`'GCN'`, :obj:`'GAT'`, :obj:`'NNConv'`, :obj:`'SGC'`, or :obj:`'SumGNN'`. (default :obj:`'SAGE'`)
        gnn_layers (int, optional): Number of GNN layers to use for learning the sheaf. (default :obj:`1`)
        gnn_hidden (int, optional): Number of hidden channels in the GNN layers. (default :obj:`32`)
        gnn_residual (bool, optional): Use residual connections in the GNN layers. (default :obj:`False`)
        pe_size (int, optional): Size of the positional encoding to use in the GNN layers. (default :obj:`0`)
    """

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 stalk_dimension: Optional[int]  = 2,
                 left_weights:    Optional[bool] = True,
                 right_weights:   Optional[bool] = True,
                 use_act:         Optional[bool] = True,
                 use_eps:         Optional[bool] = True,
                 dropout:         Optional[float] = 0.0,
                 use_bias:        Optional[bool] = False,
                 sheaf_act:       Optional[str]  = 'tanh',
                 orth_trans:      Optional[str]  = 'householder',
                 linear_emb:      Optional[bool] = False,
                 gnn_type:        Optional[str]  = 'SAGE',
                 gnn_layers:      Optional[int]  = 1,
                 gnn_hidden:      Optional[int]  = 32,
                 gnn_residual:    Optional[bool] = False,
                 pe_size:         Optional[int]  = 0,
                 ):
        MessagePassing.__init__(self, aggr='add',
                                flow='target_to_source',
                                node_dim=0)

        self.d = stalk_dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.right_weights = right_weights
        self.left_weights = left_weights
        self.use_act = use_act
        self.use_eps = use_eps
        self.dropout = dropout
        self.orth_trans = orth_trans

        if in_channels != out_channels:
            assert right_weights, \
            f'The right_weights changes from in_channels to out_channels \
            Either set right_weights=True or ensure in_channels == out_channels.'

        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.in_channels,
                                               self.out_channels,
                                               bias=use_bias)
            nn.init.orthogonal_(self.lin_right_weights.weight.data)
        else:
            self.lin_right_weights = nn.Identity()

        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d,
                                              self.d,
                                              bias=use_bias)
            nn.init.eye_(self.lin_left_weights.weight.data)
        else:
            self.lin_left_weights = nn.Identity()

        self.orth_transform = Orthogonal(d=self.d,
                                         orthogonal_map=orth_trans)
        
        self.sheaf_learner = FlatSheafLearner(
                self.d,
                self.in_channels,
                out_shape = (self.get_param_size(),),
                linear_emb = linear_emb,
                gnn_type = gnn_type,
                gnn_layers = gnn_layers,
                gnn_hidden = gnn_hidden,
                gnn_residual = gnn_residual,
                pe_size = pe_size,
                sheaf_act = sheaf_act)
        
        if use_eps and in_channels == out_channels:
            self.epsilons = nn.Parameter(torch.zeros((self.d, 1)))

    
    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2
    
    def restriction_maps_builder(self, F: Tensor, edge_index: Tensor):
        row, _ = edge_index

        maps = self.orth_transform(F)

        diag_maps = degree(row, num_nodes=self.graph_size)

        diag_sqrt_inv = (diag_maps + 1).pow(-0.5)

        norm_maps = diag_sqrt_inv.view(-1, 1, 1) * maps

        norm_D = diag_maps * diag_sqrt_inv**2 

        return norm_D, norm_maps, maps
    
    def left_right_linear(self, x: Tensor, left: Linear | Identity,
                          right: Linear | Identity) -> Tensor:
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, self.graph_size * self.d).t()

        x = right(x)
        return x
    
    def torch_total_sheaf_effective_resistance(self, L_G_pinv, R, F_maps):
        ones_d = torch.ones(self.d, dtype=F_maps.dtype, device=F_maps.device)
        S = F_maps @ ones_d
        #S = S.view(L_G_pinv.shape[0], -1, self.d)
        #almost_inner = S * (L_G_pinv @ S)
        #min_node = torch.min(almost_inner.sum(dim=1))
        #print(min_node)

        frobenius_term = torch.sum(S * (L_G_pinv @ S))

        R_F = self.d * R - frobenius_term
        
        return R_F, R, frobenius_term
    
    def torch_batched_effective_resistance(self, data, maps):
        maps = maps.to(torch.float64)

        R = data.R
        L_G_pinv = data.L_G_pinv
        results = 0.
        offset = 0
        for L_pinv, R in zip(L_G_pinv, R):
            L_pinv = torch.tensor(L_pinv, device=maps.device)
            num_nodes = L_pinv.shape[0]
            R_F, _, frobenius_norm = self.torch_total_sheaf_effective_resistance(
                L_pinv, R, maps[offset:offset+num_nodes])
            offset += num_nodes
            results += frobenius_norm

        return results
    
    def forward(self, x: Tensor, edge_index: Tensor, data=None, reff=False):
        self.graph_size = x.size(0)

        assert x.view(self.graph_size, -1).size(1) == self.in_channels * self.d, \
            f'Expected input size {self.in_channels * self.d}, got {x.view(self.graph_size, -1).size(1)}. \
            Are you embedding graph features into sheaf features?'
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_maps = x.reshape(self.graph_size, self.in_channels * self.d)

        maps = self.sheaf_learner(x_maps, edge_index)

        x = x.view(self.graph_size * self.d, -1)
        x0 = x

        D, maps, unnormalized_maps = self.restriction_maps_builder(maps, edge_index)

        x = self.left_right_linear(x, self.lin_left_weights, self.lin_right_weights)
        x = x.reshape(self.graph_size, self.d, self.out_channels)

        deg = degree(edge_index[0], num_nodes=self.graph_size)
        Dx = D[:, None, None] * x * (deg+1e-8).pow(-1)[:, None, None]

        Fx = maps @ x
        x = self.propagate(edge_index, x=Fx, diag=Dx, Ft=maps.transpose(-2,-1))
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_act:
            x = F.gelu(x)

        if self.use_eps and self.in_channels == self.out_channels:
            x = x.view(self.graph_size * self.d, -1)
            x0 = (1 + torch.tanh(self.epsilons).tile(self.graph_size, 1)) * x0 - x
            x = x0

        if reff:
            efec_res = self.torch_batched_effective_resistance(data, unnormalized_maps.detach())

        return x.view(self.graph_size, -1), efec_res if reff else 0

    def message(self, x_j, diag_i, Ft_i):
        msg = Ft_i @ x_j

        return diag_i - msg
    
class Orthogonal(nn.Module):
    """Based on https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#orthogonal"""
    def __init__(self, d, orthogonal_map):
        super().__init__()
        assert orthogonal_map in ["matrix_exp", "cayley", "householder", "euler"]
        self.d = d
        self.orthogonal_map = orthogonal_map

    def get_2d_rotation(self, params, det=1):
        # assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 1
        sin = torch.sin(params * 2 * math.pi)
        cos = torch.cos(params * 2 * math.pi)
        if det == 1:
            return torch.cat([cos, -sin,
                            sin, cos], dim=1).view(-1, 2, 2)
        if det == -1:
            return torch.cat([-cos, sin,
                            sin, cos], dim=1).view(-1, params.size(1), 2, 2)

    def get_3d_rotation(self, params):
        assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 3

        alpha = params[:, 0].view(-1, 1) * 2 * math.pi
        beta = params[:, 1].view(-1, 1) * 2 * math.pi
        gamma = params[:, 2].view(-1, 1) * 2 * math.pi

        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta),  torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)

        return torch.cat(
            [cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g,
             sin_a*cos_b, sin_a*sin_b*sin_g + cos_a*cos_g, sin_a*sin_b*cos_g - cos_a*sin_g,
             -sin_b, cos_b*sin_g, cos_b*cos_g], dim=1).view(-1, 3, 3)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.orthogonal_map != "euler":
            # Construct a lower diagonal matrix where to place the parameters.
            offset = -1 if self.orthogonal_map == 'householder' else 0
            tril_indices = torch.tril_indices(row=self.d, col=self.d, offset=offset, device=params.device)
            new_params = torch.zeros(
                (params.size(0), self.d, self.d), dtype=params.dtype, device=params.device)
            new_params[:, tril_indices[0], tril_indices[1]] = params
            params = new_params

        if self.orthogonal_map == "matrix_exp" or self.orthogonal_map == "cayley":
            # We just need n x k - k(k-1)/2 parameters
            params = params.tril()
            A = params - params.transpose(-2, -1)
            # A is skew-symmetric (or skew-hermitian)
            if self.orthogonal_map == "matrix_exp":
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == "cayley":
                # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
                Id = torch.eye(self.d, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
        elif self.orthogonal_map == 'householder':
            eye = torch.eye(self.d, device=params.device).unsqueeze(0).repeat(params.size(0), 1, 1)
            A = params.tril(diagonal=-1) + eye
            Q = torch_householder_orgqr(A)
        elif self.orthogonal_map == 'euler':
            assert 2 <= self.d <= 3
            if self.d == 2:
                Q = self.get_2d_rotation(params)
            else:
                Q = self.get_3d_rotation(params)
        else:
            raise ValueError(f"Unsupported transformations {self.orthogonal_map}")
        return Q

class SumGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') 
        self.W_s = nn.Linear(in_channels, out_channels, bias=False)
        self.W_n = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out = self.propagate(edge_index, x=x)

        return F.gelu(self.W_s(x) + self.W_n(out))

    def message(self, x_j):
        return x_j

class FlatSheafLearner(nn.Module):
    """Learns a conformal sheaf passing node features through a GNN or MLP + activation."""

    def __init__(self, d:         int,
                 hidden_channels: int,
                 out_shape:       Tuple[int],
                 linear_emb:      bool,
                 gnn_type:        str,
                 gnn_layers:      int,
                 gnn_hidden:      int,
                 gnn_residual:    bool,
                 pe_size:         int,
                 sheaf_act:       str = 'tanh'):
        super(FlatSheafLearner, self).__init__()
        
        assert len(out_shape) in [1, 2]
        assert (gnn_type, gnn_residual) != ('SGC', True), "SGC does not support residual connections."
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.gnn_layers = gnn_layers
        self.residual = gnn_residual
        self.linear_emb = linear_emb
        self.gnn_hidden = gnn_hidden
        self.layer_type = gnn_type
        self.sheaf_act = sheaf_act
        out_channels = int(np.prod(out_shape))

        if sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'relu':
            self.act = F.relu
        elif sheaf_act == 'gelu':
            self.act = F.gelu
        elif sheaf_act == 'sigmoid':
            self.act = torch.sigmoid
        elif sheaf_act == 'elu':
            self.act = F.elu
        elif sheaf_act == 'id':
            self.act = lambda x: x
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

        if gnn_layers > 0:
            self.gnn = self.get_layer_type(gnn_type)
            if linear_emb:
                self.emb1 = nn.Linear((hidden_channels + pe_size) * d, gnn_hidden)
                self.phi = self.gnn_builder(gnn_type, gnn_hidden, gnn_hidden, gnn_layers)
                self.emb2 = nn.Linear(gnn_hidden, out_channels)
            else:
                self.phi = self.gnn_builder(gnn_type, (hidden_channels + pe_size) * d, out_channels, gnn_layers, gnn_hidden)

        else:
            self.phi = torch.nn.Linear(hidden_channels, int(np.prod(out_shape)), bias=False)

    def get_layer_type(self, layer_type):
        if layer_type == 'GCN':
            model_cls = GCNConv
        elif layer_type == 'GAT':
            model_cls = GATConv
        elif layer_type == 'SAGE':
            model_cls = SAGEConv
        elif layer_type == 'SGC':
            model_cls = SGConv
        elif layer_type == 'GPS':
            model_cls = GPSConv
        elif layer_type == 'NNConv':
            model_cls = NNConv
        elif layer_type == 'GIN':
            model_cls = GINConv
        elif layer_type == 'SumGNN':
            model_cls = SumGNN
        else:
            raise ValueError(f"Unsupported GNN layer type: {layer_type}")
        return model_cls
    
    def gnn_builder(self, gnn_type, in_channels, out_channels, num_layers, hidden_channels=None):
        gnn = self.get_layer_type(gnn_type)
        layers = nn.ModuleList()
        if hidden_channels is None or num_layers == 1:
            if gnn_type == 'GPS':
                raise NotImplementedError("Lacking GPSConv setup.")
            elif gnn_type == 'NNConv':
                edge_net = nn.LazyLinear(in_channels*out_channels)
                for i in range(num_layers):
                    layers.append(gnn(in_channels, out_channels, nn=edge_net, aggr='add'))
            elif gnn_type == 'SGC':
                layers = gnn(in_channels, out_channels, K=num_layers)
            elif gnn_type in ['GCN', 'GAT', 'SAGE', 'SumGNN']:
                for i in range(num_layers):
                    layers.append(gnn(in_channels, out_channels))
            elif gnn_type == 'GIN':
                for i in range(num_layers):
                    layers.append(gnn(nn.Sequential(nn.Linear(in_channels, out_channels),nn.BatchNorm1d(out_channels), nn.ReLU(),nn.Linear(out_channels, out_channels))))
            else:
                raise ValueError(f"Unsupported GNN layer type: {gnn_type}")
        else:
            if gnn_type == 'GPS':
                raise NotImplementedError("GPSConv is not implemented.")
            elif gnn_type == 'NNConv':
                edge_net = nn.LazyLinear(in_channels*hidden_channels)
                layers.append(gnn(in_channels, hidden_channels, nn=edge_net, aggr='add'))
                edge_net = nn.LazyLinear(hidden_channels**2)
                for i in range(num_layers-2):
                    layers.append(gnn(hidden_channels, hidden_channels, nn=edge_net, aggr='add'))
                edge_net = nn.LazyLinear(hidden_channels*out_channels)
                layers.append(NNConv(hidden_channels, out_channels, nn=edge_net, aggr='add'))
            elif gnn_type == 'SGC':
                layers.append(SGConv(in_channels, hidden_channels, K=1))
                layers.append(SGConv(hidden_channels, hidden_channels, K=num_layers-2))
                layers.append(SGConv(hidden_channels, out_channels, K=1))
            elif gnn_type in ['GCN', 'GAT', 'SAGE', 'SumGNN']:
                layers.append(gnn(in_channels, hidden_channels))
                for i in range(num_layers-2):
                    layers.append(gnn(hidden_channels, hidden_channels))
                layers.append(gnn(hidden_channels, out_channels))
            elif gnn_type == 'GIN':
                layers.append(gnn(nn.Sequential(nn.Linear(in_channels, hidden_channels),nn.BatchNorm1d(hidden_channels), nn.ReLU(),nn.Linear(hidden_channels, hidden_channels))))
                for i in range(num_layers-2):
                    layers.append(gnn(nn.Sequential(nn.Linear(hidden_channels, hidden_channels),nn.BatchNorm1d(hidden_channels), nn.ReLU(),nn.Linear(hidden_channels, hidden_channels))))
                layers.append(gnn(nn.Sequential(nn.Linear(hidden_channels, out_channels),nn.BatchNorm1d(out_channels), nn.ReLU(),nn.Linear(out_channels, out_channels))))
            else:
                raise ValueError(f"Unsupported GNN layer type: {gnn_type}")
        return layers

    def forward(self, x, edge_index, edge_attr=None, pe=None):
        pe = pe if pe is not None else torch.empty(x.size(0), 0, device=x.device)
        maps = torch.cat([x, pe], -1)
        
        if self.gnn_layers > 0:
            sig = inspect.signature(self.gnn.forward)
            if self.linear_emb:
                maps = self.emb1(maps)
                maps = F.gelu(maps)
                if self.layer_type != 'SGC':
                    for layer in range(self.gnn_layers):
                        prev = maps
                        if edge_attr is not None and 'edge_attr' in sig.parameters:
                            maps = self.phi[layer](maps, edge_index, edge_attr=edge_attr)
                        else:
                            maps = self.phi[layer](maps, edge_index)
                        #maps = F.gelu(maps)
                        if self.residual:
                            maps = maps + prev
                else:
                    maps = self.phi(maps, edge_index)
                    maps = F.gelu(maps)
                maps = self.emb2(maps)
            else:
                if self.layer_type != 'SGC':
                    for layer in range(self.gnn_layers):
                        prev = maps
                        if edge_attr is not None and 'edge_attr' in sig.parameters:
                            maps = self.phi[layer](maps, edge_index, edge_attr=edge_attr)
                        else:
                            maps = self.phi[layer](maps, edge_index)
                        maps = F.gelu(maps) if (self.gnn_layers != 1 and layer != self.gnn_layers - 1) else maps
                        if self.residual and layer not in [self.gnn_layers - 1, 0]:
                            maps = maps + prev
                else:
                    if self.gnn_layers == 1:
                        maps = self.phi(maps, edge_index)
                    else:
                        for layer in self.phi:
                            maps = layer(maps, edge_index)
                            maps = F.gelu(maps) if layer != self.phi[-1] else maps
        else:
            maps = maps.view(-1, self.d, self.hidden_channels).sum(dim=1)
            maps = self.phi(maps)

        return self.act(maps)

class LocalConcatFlatSheafLearnerVariant(nn.Module):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatFlatSheafLearnerVariant, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels, int(np.prod(out_shape)), bias=False)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x):
        x_cat = x.reshape(-1, self.d, self.hidden_channels).sum(dim=1)
        x_cat = self.linear1(x_cat)
        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])