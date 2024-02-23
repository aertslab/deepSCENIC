import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_uniform_
from torchmetrics import F1Score


class LossFunctions:
    def __init__(self, dev):
        super(LossFunctions, self).__init__()
        self.eps = 1e-8
        self.dev = dev

    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type='mse'):
        if rec_type == 'mse':
            if dropout_mask==False:
                loss = torch.mean((real - predicted).pow(2)) #+ (1- torch.mean(F.cosine_similarity(real, predicted, dim=1)))
            else:
                mask = real!=0
                loss = torch.mean(torch.sum((real - predicted).pow(2) * mask, dim=1) / torch.sum(mask, dim=1))
        elif rec_type == 'mae':
            if dropout_mask==False:
                loss = torch.mean((real - predicted).abs())
            else:
                mask = real!=0
                loss = torch.mean(torch.sum((real - predicted).abs() * mask, dim=1) / torch.sum(mask, dim=1))
        elif rec_type == 'bce':
            mask = ~(real == -1).all(dim=1)
            if real[mask].shape[0] !=0:
                loss = F.binary_cross_entropy_with_logits(predicted[mask], real[mask], reduction='none').mean()
            else:
                loss = torch.Tensor([0]).to(self.dev)
        elif rec_type == 'cos':
            mask = ~(real == -1).all(dim=1)
            if real[mask].shape[0] !=0:
                loss = (1- torch.mean(F.cosine_similarity(real[mask], predicted[mask], dim=1)))
            else:
                loss = torch.Tensor([0]).to(self.dev)
        else:
            raise Exception
        return loss

    def gaussian_loss(self, mu, logvar):
        """KL divergence loss."""
        loss = -0.5 * torch.mean(1. + logvar - mu.pow(2) - logvar.exp(), )
        return loss.mean()

def pos_xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    """ Positively constrained Xavier uniform initialization
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, 0, a)

class posLinear(nn.Module):
    """ Linear layer with constrained positive weights
    """
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
        pos_xavier_uniform_(self.weight)
        if self.bias is not None:
            # Initialize bias to zero
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, input: Tensor) -> Tensor:
        # Constrain weights to be positive
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

    def forward(self, x):
        mu, logvar = self.qzxy(x) # Encoder pass
        z_reg = self.sample_latent(mu, logvar)  # Sample latent space
        output = {'mean'  : mu, 'logvar': logvar, 'z_reg':z_reg}
        return output


class GenerativeNet(nn.Module):
    """Decoder network"""
    def __init__(self, x_dim, z_dim, nonLinear, bias=False):
        super(GenerativeNet, self).__init__()
        self.generative_pxz = torch.nn.ModuleList([
            posLinear(1, z_dim, bias=bias),
            nonLinear,
            posLinear(z_dim, z_dim, bias=bias),
            nonLinear,
            posLinear(z_dim, x_dim, bias=bias),
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
    def __init__(self, TFs_idx, r2g_dist_coo, x_dim, z_dim, opt):
        super(VAE, self).__init__()
        self.eps = 1e-4
        self.r2g_dist = torch.sparse_coo_tensor(torch.tensor([r2g_dist_coo.row.tolist(), r2g_dist_coo.col.tolist()]), torch.tensor(r2g_dist_coo.data).float(), r2g_dist_coo.shape, requires_grad=False).coalesce().float().to(opt.device)
        self.adj_E2 = nn.Parameter(torch.zeros(r2g_dist_coo.size, device=opt.device, requires_grad=True) + self.eps)
        # self.adj_E2 = nn.Parameter(torch.randn(r2g_dist_coo.size, device=dev, requires_grad=True))
        self.TFs_idx = torch.tensor(TFs_idx).to(opt.device)

        self.n_gene = r2g_dist_coo.shape[1]
        self.n_tfs = len(TFs_idx)
        self.n_region = r2g_dist_coo.shape[0]
        nonLinear = nn.Tanh()
        if opt.device!='cpu':
            use_cuda=True
        else:
            use_cuda=False
        self.inference_rna = InferenceNet(x_dim, z_dim, nonLinear, use_cuda)
        self.generative_rna = GenerativeNet(x_dim, z_dim, nonLinear)
        self.generative_atac = GenerativeNet(x_dim, z_dim, nonLinear, bias=opt.bin_acc)
        self.losses = LossFunctions(dev=opt.device)
        self.device = opt.device
        self.opt = opt

        # Layers initilization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def predict(self, x_rna, adj_E1=None, adj_E2=None):
        x_rna_tfs = x_rna[:, self.TFs_idx]

        out_inf_rna = self.inference_rna(x_rna_tfs.view(x_rna_tfs.size(0), -1, 1))
        if self.opt.train==True:
            enh_act = torch.matmul(out_inf_rna['z_reg'], adj_E1.T)
        else:
            enh_act = torch.matmul(out_inf_rna['mean'], adj_E1.T)
        out_gen_atac = self.generative_atac(enh_act)
        if adj_E2 is not None:
            if len(adj_E2.shape)<2:
                E2 = torch.sparse_coo_tensor(self.r2g_dist.indices(), adj_E2.abs(), self.r2g_dist.shape).to_dense()
            else:
                E2 = adj_E2
            z_rna = torch.matmul(enh_act, E2)
            # z_rna[z_rna<0] = 0
            out_gen_rna = self.generative_rna(z_rna)
        else:
            z_rna = None
            out_gen_rna = None
        return out_gen_rna, out_gen_atac, out_inf_rna, enh_act, z_rna
    
    def pretrain(self, x_rna_tfs, x_atac, adj_E1=None, idxs=None):

            out_inf_rna = self.inference_rna(x_rna_tfs.view(x_rna_tfs.size(0), -1, 1))
            if self.opt.train==True:
                enh_act = torch.matmul(out_inf_rna['z_reg'], adj_E1.T)
            else:
                enh_act = torch.matmul(out_inf_rna['mean'], adj_E1.T)
            out_gen_atac = self.generative_atac(enh_act)

            if self.opt.bin_acc==True:
                f1 = F1Score(task='binary',num_classes=1).to(self.opt.device)
                mask = ~(x_atac == -1).all(dim=1)
                if x_atac[mask].shape[0] !=0:
                    f1_atac = f1(out_gen_atac['x_rec'][mask].ravel(), x_atac[mask][:, idxs].int().ravel())
            else:
                f1_atac = torch.Tensor([0]).to(self.opt.device)                        

            loss_rec_atac = self.losses.reconstruction_loss(x_atac[:, idxs], out_gen_atac['x_rec'], False, rec_type=self.opt.loss_atac)
            loss_gauss_rna = self.losses.gaussian_loss(out_inf_rna['mean'], out_inf_rna['logvar']) * self.opt.beta

            loss = loss_gauss_rna + loss_rec_atac

            return loss,  loss_rec_atac.detach(), loss_gauss_rna.detach(), f1_atac.detach()


    def forward(self, x_rna, x_atac, dropout_mask_rna=None, dropout_mask_atac=None, adj_E1=None, idxs=None):
        x_rna_tfs = x_rna[:, self.TFs_idx]
        E2 = torch.sparse_coo_tensor(self.r2g_dist.indices(), self.adj_E2.abs(), self.r2g_dist.shape).to_dense()

        out_inf_rna = self.inference_rna(x_rna_tfs.view(x_rna_tfs.size(0), -1, 1))
        if self.opt.train==True:
            enh_act = torch.matmul(out_inf_rna['z_reg'], adj_E1.T)
        else:
            enh_act = torch.matmul(out_inf_rna['mean'], adj_E1.T)
        # enh_act[enh_act<0] = 0
        z_rna = torch.matmul(enh_act, E2)
        # z_rna[z_rna<0] = 0
        out_gen_rna = self.generative_rna(z_rna)
        out_gen_atac = self.generative_atac(enh_act)    

        
        with torch.no_grad():
            if self.opt.bin_acc==True:
                f1 = F1Score(task='binary',num_classes=1).to(self.opt.device)
                mask = ~(x_atac == -1).all(dim=1)
                if x_atac[mask].shape[0] !=0:
                    f1_atac = f1(out_gen_atac['x_rec'][mask].ravel(), x_atac[mask].int().ravel())
                del mask    
            else:
                f1_atac = torch.Tensor([0]).to(self.opt.device)  

        loss_rec_rna = self.losses.reconstruction_loss(x_rna, out_gen_rna['x_rec'], dropout_mask_rna, rec_type=self.opt.loss_rna) * self.opt.rna_tau
        loss_rec_atac = self.losses.reconstruction_loss(x_atac, out_gen_atac['x_rec'], dropout_mask_atac, rec_type=self.opt.loss_atac) * self.opt.atac_tau
        loss_gauss_rna = self.losses.gaussian_loss(out_inf_rna['mean'], out_inf_rna['logvar']) * self.opt.beta
        
        with torch.no_grad():
            rna_pos_loss = torch.nan_to_num(out_gen_rna['x_rec'][out_gen_rna['x_rec']<0].mean().abs(), 0)

        E2_sparse_loss = torch.sparse_coo_tensor(self.r2g_dist.indices(), self.adj_E2.abs() * self.r2g_dist.values(), self.r2g_dist.shape).coalesce().to(self.opt.device)
        if idxs is not None:
            E2_sparse_idxs = E2_sparse_loss.indices()[0] # take row indices for non zero elements
            E2_sparse_idxs = torch.where(torch.isin(E2_sparse_idxs, idxs.to(self.opt.device)))[0]
            E2_sparse_loss = E2_sparse_loss.values()[E2_sparse_idxs].mean() * self.opt.gamma
        else:
            E2_sparse_loss = E2_sparse_loss.values().mean() * self.opt.gamma
        # loss = loss_rec_rna + loss_gauss_rna + loss_rec_atac

        return loss_rec_rna, loss_rec_atac, loss_gauss_rna, E2_sparse_loss, rna_pos_loss, out_gen_rna['x_rec'].detach(), out_gen_atac['x_rec'].detach(), z_rna.detach(), out_inf_rna['mean'].detach(),  out_inf_rna['logvar'].detach(), enh_act.detach(), f1_atac.detach()
