import math
import numpy as np
import torch
import torch.nn.functional  as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import warnings

from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain, _no_grad_uniform_
from torch.cuda.amp import custom_bwd, custom_fwd

from torchmetrics import F1Score

from tqdm import tqdm

def matmul_adj(X, E1, adj_E2, TFs_idx, r2g_dist, modality='atac'):
    if modality=='atac':
        X1 = torch.matmul(X, adj_E2)
        X = torch.matmul(X1[:, TFs_idx], E1)
        D = torch.einsum('ij,ji->i', adj_E2[:, TFs_idx], E1)
        Y_diag = torch.einsum('ij,j->ij', X, D)
        X = X - Y_diag
    elif modality=='rna':
        X1 = torch.matmul(X, E1)
        X = torch.matmul(X1, adj_E2)
        D = torch.einsum('ij,ji->i', E1, adj_E2[:,TFs_idx])
        Y_diag = torch.einsum('ij,j->ij', X[:, TFs_idx], D)
        X[:, TFs_idx] = X[:, TFs_idx] - Y_diag

    return X, X1

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

class LossFunctions:
    def __init__(self, dev):
        super(LossFunctions, self).__init__()
        self.eps = 1e-8
        self.dev = dev

    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type='mse'):
        if rec_type == 'mse':
            if dropout_mask is None:
                loss = torch.mean((real - predicted).pow(2))
            else:
                loss = torch.mean(torch.sum((real - predicted).pow(2) * dropout_mask, dim=1) / torch.sum(dropout_mask, dim=1))
        elif rec_type == 'mae':
            if dropout_mask is None:
                loss = torch.mean((real - predicted).abs())
            else:
                loss = torch.mean(torch.sum((real - predicted).abs() * dropout_mask, dim=1) / torch.sum(dropout_mask, dim=1))
        elif rec_type == 'bce':
            if dropout_mask is None:
                loss = F.binary_cross_entropy_with_logits(predicted, real, reduction='none').mean()
            else:
                dropout_mask = dropout_mask.clone()
                n_regions = dropout_mask.sum()
                class_weight = (dropout_mask.shape[0]*dropout_mask.shape[1] - n_regions)/n_regions

                return torch.nn.BCEWithLogitsLoss(reduce='mean')(predicted, real)
        else:
            raise Exception
        return loss

    def gaussian_loss(self, mu, logvar):
        """KL divergence loss."""
        loss = -0.5 * torch.mean(1. + logvar - mu.pow(2) - logvar.exp(), )
        return loss.mean()

def pos_xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, 0, a)

class posLinear(nn.Module):
    """ Linear layer with positive weights"""
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(posLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        pos_xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.abs(), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = posLinear(in_dim, z_dim, bias=False)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu.squeeze(2), logvar.squeeze(2)

class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, nonLinear, use_cuda=True):
        super(InferenceNet, self).__init__()
        self.use_cuda = use_cuda
        self.inference_qzyx = torch.nn.ModuleList([
            posLinear(x_dim, z_dim, bias=False),
            nonLinear,
            posLinear(z_dim, z_dim, bias=False),
            nonLinear,
            Gaussian(z_dim, 1)
        ])

    def sample_latent(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if self.use_cuda:
            eps = eps.to(torch.device('cuda'))
        eps = eps.mul_(std).add_(mu)
        return eps

    def qzxy(self, x):
        for layer in self.inference_qzyx:
            x = layer(x)
        return x

    def forward(self, x, E1, E2, TFs_idx, r2g_dist, modality=None):
        mu, logvar = self.qzxy(x)
        z_reg = self.sample_latent(mu, logvar)
        z, z1 = matmul_adj(z_reg, E1, E2, TFs_idx, r2g_dist, modality=modality)
        output = {'mean'  : mu, 'logvar': logvar, 'gaussian': z, 'gaussian1': z1, 'z_reg':z_reg}
        return output

class GenerativeNet(nn.Module):
    """Decoder network"""
    def __init__(self, x_dim, z_dim, nonLinear):
        super(GenerativeNet, self).__init__()
        self.generative_pxz = torch.nn.ModuleList([
            posLinear(1, z_dim, bias=False),
            nonLinear,
            posLinear(z_dim, z_dim, bias=False),
            nonLinear,
            posLinear(z_dim, x_dim, bias=False),
        ])

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z):
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        output = {'x_rec': x_rec}
        return output

class GenerativeNetATAC(nn.Module):
    """Decoder network"""
    def __init__(self, x_dim, z_dim, nonLinear):
        super(GenerativeNetATAC, self).__init__()
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(1, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z):
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        output = {'x_rec': x_rec}
        return output

class VAE(nn.Module):
    def __init__(self, TFs_idx, r2g_dist_coo, x_dim, z_dim, dev, E1_dropout=0.2):
        super(VAE, self).__init__()
        self.eps = 1e-8
        self.r2g_dist = torch.sparse_coo_tensor(torch.tensor([r2g_dist_coo.row.tolist(), r2g_dist_coo.col.tolist()]), torch.tensor(1/r2g_dist_coo.data).float(), r2g_dist_coo.shape, requires_grad=False).coalesce().float().to(dev)
        self.adj_E2 = nn.Parameter(torch.zeros(r2g_dist_coo.size, device=dev, requires_grad=True) + self.eps)
        self.TFs_idx = torch.tensor(TFs_idx).to(dev)

        self.n_gene = n_gene = r2g_dist_coo.shape[1]
        self.n_tfs = len(TFs_idx)
        self.n_region = n_region = r2g_dist_coo.shape[0]
        nonLinear = nn.Tanh()
        if dev!='cpu':
            use_cuda=True
        else:
            use_cuda=False
        self.inference_rna = InferenceNet(x_dim, z_dim, nonLinear, use_cuda)
        self.generative_rna = GenerativeNet(x_dim, z_dim, nonLinear)
        self.generative_atac = GenerativeNet(x_dim, z_dim, nonLinear)
        self.E1_dropout = nn.Dropout(p=E1_dropout)
        self.losses = LossFunctions(dev=dev)
        self.device = dev

        # Layers initilization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x_rna, x_atac, dropout_mask_rna, dropout_mask_atac, opt=None, adj_E1=None):
        x_rna_tfs = x_rna[:, self.TFs_idx]
        E1 = self.E1_dropout(adj_E1)
        E2 = torch.sparse_coo_tensor(self.r2g_dist.indices(), dclamp(self.adj_E2, min=0, max=None), self.r2g_dist.shape).to_dense()
        out_inf_rna = self.inference_rna(x_rna_tfs.view(x_rna_tfs.size(0), -1, 1), E1, E2, self.TFs_idx, self.r2g_dist, modality='rna')
        out_gen_rna = self.generative_rna(out_inf_rna['gaussian'])
        out_gen_atac = self.generative_atac(out_inf_rna['gaussian1'])

        if opt.bin_acc==True:
            loss_acc = 'bce'
            f1 = F1Score(num_classes=1).to(opt.device)
            f1_atac = f1(out_gen_atac['x_rec'].ravel(), x_atac.int().ravel())
        else:
            loss_acc = 'mae'
            f1_atac = torch.Tensor([0])

        loss_rec_rna = self.losses.reconstruction_loss(x_rna, out_gen_rna['x_rec'], dropout_mask_rna, rec_type='mae')
        loss_rec_atac = self.losses.reconstruction_loss(x_atac, out_gen_atac['x_rec'], dropout_mask_atac, rec_type=loss_acc) * opt.atac_gamma
        loss_gauss_rna = self.losses.gaussian_loss(out_inf_rna['mean'], out_inf_rna['logvar']) * opt.beta

        loss = loss_rec_rna + loss_gauss_rna + loss_rec_atac

        return loss, loss_rec_rna.detach(), loss_rec_atac, loss_gauss_rna.detach(), out_gen_rna['x_rec'].detach(), out_inf_rna['gaussian'].detach(), out_inf_rna['mean'].detach(),  out_inf_rna['logvar'].detach(), out_inf_rna['gaussian1'].detach(), f1_atac.detach()

