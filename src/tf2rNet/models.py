import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import splev
from torch import nn
from torch.nn import init
from torchmetrics import F1Score

class LossFunctions:
    def bce_loss(self, real, predicted):
        loss = F.binary_cross_entropy_with_logits(predicted, real, reduction='none').mean()
        loss = torch.mean((real - predicted).pow(2))

        return loss

class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self, dev='cuda'):
        super(StochasticReverseComplement, self).__init__()
        self.dev = dev

    def forward(self, seq_1hot):
        if self.training:
            rc_seq_1hot = torch.gather(seq_1hot, index = torch.tile(torch.LongTensor([3, 2, 1, 0]).to(self.dev), ([seq_1hot.size(0),seq_1hot.size(1),1])), dim=-1)
            rc_seq_1hot = torch.flip(rc_seq_1hot, dims=[1])
            reverse_bool = torch.rand(1) > 0.5
            if reverse_bool:
                src_seq_1hot = rc_seq_1hot
            else:
                src_seq_1hot = seq_1hot

            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, torch.tensor(False)

class SwitchReverse(nn.Module):
    """Reverse predictions if the inputs were reverse complemented."""

    def __init__(self):
        super(SwitchReverse, self).__init__()

    def forward(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        xd = len(x.size())
        if xd == 2:
            rev_axes = [0]
        elif xd == 3:
            rev_axes = [0, 1]
        else:
            raise ValueError("Cannot recognize SwitchReverse input dimensions %d." % xd)

        if reverse:
            x= torch.flip(x, dims=rev_axes)

        return x


class StochasticShift(nn.Module):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, pad="uniform"):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
        self.pad = pad

    def forward(self, seq_1hot):
        if self.training:
            shift_i = torch.FloatTensor(1).uniform_(0,len(self.augment_shifts)).type(torch.int64)
            shift = torch.gather(self.augment_shifts, index=shift_i, dim=0)
            if shift:
                sseq_1hot = shift_sequence(seq_1hot, shift)
            else:
                sseq_1hot = seq_1hot

            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({"shift_max": self.shift_max, "pad": self.pad})
        return config


def shift_sequence(seq, shift, pad_value=0.25):
    """Shift a sequence left or right by shift_amount.
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if len(seq.size()) != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.size()

    pad = pad_value * torch.ones_like(seq[:, 0 : torch.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return torch.cat([pad, sliced_seq], dim=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return torch.cat([sliced_seq, pad], dim=1)

    if shift > 0:
        sseq = _shift_right(seq)
    else:
        sseq = _shift_left(seq)

    sseq = sseq.reshape(input_shape)

    return sseq

class conv_block(nn.Module):
    def __init__(self, x_dim, ch_dim, activation, n_filters=None, kernel_size=1, strides=1, dilation_rate=1, dropout=0, residual=False, pool_size=1, batch_norm=True, bn_momentum=0.90, padding='same'):
        """
            n_filters:       Conv1D filter number
            kernel_size:   Conv1D kernel_size
            activation     Activation function
            strides:       Conv1D strides
            dilation_rate: Conv1D dilation rate
            dropout:       Dropout rate probability
            residual:      Residual connection boolean
            pool_size:     Max pool width
            batch_norm:    Apply batch normalization
            bn_momentum:   BatchNorm momentum
        """   
        super(conv_block, self).__init__()
        self.nonLinear = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.residual = residual
        self.pool_size = pool_size
        self.padding = padding
        self.dilation_rate = dilation_rate
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(n_filters, momentum=bn_momentum)
            if residual:
               init.zeros_(self.bn_layer.weight)
            else:
               init.ones_(self.bn_layer.weight)
        if dropout > 0 :
            self.dropout_layer = nn.Dropout(p=dropout)
        self.conv_layer = nn.Conv1d(in_channels = ch_dim,
            out_channels = n_filters,
            kernel_size = kernel_size,
            stride = strides,
            padding = 'same',
            dilation = dilation_rate,
            bias = False
        )
        init.kaiming_normal_(self.conv_layer.weight, mode='fan_out')
        if pool_size > 1:
            if padding == "same":
                P = int(((strides-1)*x_dim-strides+pool_size)/2)
                self.maxpool_layer = nn.MaxPool1d(pool_size, padding=P)
        
    def forward(self, inputs):
        current = self.nonLinear(inputs)
        current = self.conv_layer(current)

        # batch norm
        if self.batch_norm:
            current = self.bn_layer(current)

        # dropout
        if self.dropout > 0:
            current = self.dropout_layer(current)
 
        # residual add
        if self.residual:
            current = inputs + current

        # pool
        if self.pool_size > 1:
            current = self.maxpool_layer(current)

        return current

class conv_tower(nn.Module):
    def __init__(self, x_dim_init, n_filters_init, n_filters_end=None, n_filters_mult=None, divisible_by=1, repeat=1, **kwargs):
        """Construct a reducing convolution block.
        Args:
            n_filters_init:  Initial Conv1D filters
            n_filters_end:   End Conv1D filters
            n_filters_mult:  Multiplier for Conv1D filters
            divisible_by:  Round filters to be divisible by (eg a power of two)
            repeat:        Tower repetitions
        Returns:
            [batch_size, seq_length, features] output sequence
        """
        super(conv_tower, self).__init__()
        self.conv_blocks = nn.ModuleList()

        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)

        # determine multiplier
        if n_filters_mult is None:
            assert n_filters_end is not None
            n_filters_mult = np.exp(np.log(n_filters_end / n_filters_init) / (repeat - 1))

        # initialize filters
        
        rep_filters_in = rep_filters_out = n_filters_init
        x_dim = x_dim_init

        for ri in range(repeat):
            # convolution
            self.conv_blocks.append(conv_block(x_dim=x_dim, ch_dim=_round(rep_filters_in), n_filters=_round(rep_filters_out), activation=nn.GELU(), **kwargs))

            # update filters
            rep_filters_in = rep_filters_out
            rep_filters_out *= n_filters_mult
            x_dim = x_dim // 2

    def forward(self, inputs):
        current = inputs
        for layer in self.conv_blocks:
            current = layer(current)

        return current


class dense_block(nn.Module):
    def __init__(self, x_dim, ch_dim, activation, n_units=None, flatten=False, dropout=0, residual=False, batch_norm=True, bn_momentum=0.90):
        """Construct a single dense block.
           Args:
               n_units:        Numbe of Conv1D filters
               activation:     relu/gelu/etc
               flatten:        Flatten across positional axis
               dropout:        Dropout rate probability
               residual:       Residual connection boolean
               batch_norm:     Apply batch normalization
               bn_momentum:    BatchNorm momentum
           Returns:
               [batch_size, seq_length(?), features] output sequence
        """
        super(dense_block, self).__init__()
        self.nonLinear = activation
        self.flatten = flatten
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.residual = residual
        if n_units is None:
            n_units = inputs.shape[-1]
        self.dense_layer = nn.Linear(x_dim*ch_dim, n_units, bias=(not batch_norm))
        init.kaiming_normal_(self.dense_layer.weight, mode='fan_out')
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(n_units, momentum=bn_momentum)
            if residual:
               init.zeros_(self.bn_layer.weight)
            else:
               init.ones_(self.bn_layer.weight)
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)        

    def forward(self, inputs):
        current = self.nonLinear(inputs)
     
        if self.flatten:
            current = torch.flatten(current, start_dim=1)

        # dense
        current = self.dense_layer(current)

        # batch norm
        if self.batch_norm:
            current = self.bn_layer(current)

        # dropout
        if self.dropout > 0:
            current = self.dropout_layer(current)

        # residual add
        if self.residual:
            current = inputs + current

        return current


class final(nn.Module):
    def __init__(self, ch_dim, n_units=None, flatten=False):
        """Final simple transformation before comparison to targets.
        Args:
            inputs:         [batch_size, seq_length, features] input sequence
            units:          Dense units
            activation:     relu/gelu/etc
            flatten:        Flatten positional axis.
        Returns:
            [batch_size, units] output sequence
        """
        super(final, self).__init__()
        self.dense_layer = nn.Linear(ch_dim, n_units, bias=True)
        init.kaiming_normal_(self.dense_layer.weight, mode='fan_out')
        self.flatten = flatten

    def forward(self, inputs):
        current = inputs
        
        if self.flatten:
            current = torch.flatten(current, start_dim=1)

        current = self.dense_layer(current)

        return current, self.dense_layer.weight


class TF2rNet(nn.Module):
    def __init__(self, bottleneck_size, n_TFs, seq_len=500, dev='cuda'):
        """create keras CNN model.
        Args:
            bottleneck_size:int. size of the bottleneck layer.
            n_TFs:        int. number of cells in the dataset. Defined the number of tasks.
            seq_len:        int. peak size used to train. Default to 1344.
        """
        super(TF2rNet, self).__init__()
        self.conv_block1 = conv_block(x_dim=seq_len, ch_dim=4, n_filters=288, activation=nn.GELU(), kernel_size=17, pool_size=3)
        self.conv_tower1 = conv_tower(x_dim_init=448, n_filters_init=288, n_filters_mult=1.122, repeat=6, kernel_size=5, pool_size=2,)
        self.conv_block2 = conv_block(x_dim=7, ch_dim=512, n_filters=256, activation=nn.GELU(), kernel_size=1)
        self.dense_block1 = dense_block(x_dim=2, ch_dim=256, flatten=True, activation=nn.GELU(), n_units=bottleneck_size, dropout=0.2, batch_norm=True)
        self.nonLinear = nn.GELU()

        self.StochasticReverseComplement_layer = StochasticReverseComplement(dev=dev)
        self.losses = LossFunctions()
        self.dev = dev

    def forward(self, sequence):
        sequence = F.one_hot(sequence.to(torch.int64), num_classes=4).to(torch.float).swapaxes(1,2)
#        (sequence, reverse_bool) = self.StochasticReverseComplement_layer(sequence)  # enable random rv
        current_fwd = sequence
        current_rev = torch.flip(sequence, dims=(1,2))

        current_fwd = self.conv_block1(current_fwd)
        current_fwd = self.conv_tower1(current_fwd)
        current_fwd = self.conv_block2(current_fwd)
        current_fwd = self.dense_block1(current_fwd)
        
        current_rev = self.conv_block1(current_rev)
        current_rev = self.conv_tower1(current_rev)
        current_rev = self.conv_block2(current_rev)
        current_rev = self.dense_block1(current_rev)

        current = torch.stack((current_fwd, current_rev), dim=1)
        return current

class MotifNet(nn.Module):
    def __init__(self, tfs, bottleneck_size, emb_len=5, explain=False, dev='cuda'):
        super(MotifNet, self).__init__()
        self.ctx_head_layer = nn.Conv1d(in_channels = 1,
            out_channels = len(tfs),
            kernel_size = bottleneck_size,
            padding = 'valid',
            bias = False
        )
        self.ctx_lin = nn.Linear(emb_len, 1)

        self.n_TFs = len(tfs)
        self.emb_len = emb_len
        self.bottleneck_size = bottleneck_size
        self.device = dev

    def forward(self, seq=None, emb=None, explain=False):
        emb = emb.reshape(-1, 1, self.bottleneck_size)
        ctx_head = self.ctx_head_layer(emb).reshape(-1, self.emb_len, self.n_TFs).swapaxes(1,2)
        ctx_head = torch.squeeze(self.ctx_lin(ctx_head))
        return ctx_head.to(torch.float)

class LossFunctions:
    def bce_loss(self, real, predicted):
        loss = F.binary_cross_entropy_with_logits(predicted, real, reduction='mean')
        return loss



"""
Sei architecture modified from https://github.com/FunctionLab/sei-framework/
"""

def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.

    """

    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d"
                  % (order - (1 - intercept)))

        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1])

    elif knots is not None:
        inner_knots = knots

    all_knots = np.concatenate(
        ([np.min(x), np.max(x)] * order, inner_knots))

    all_knots.sort()

    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)

    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))

    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n/2.0)
        dist = np.log(np.abs(dist) + 1) * ( 2*(dist>0)-1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist),np.max(dist),n_knots+2)[1:-1]
        return torch.from_numpy(bs(
            dist, knots=knots, intercept=True)).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(
            dist, df=df, intercept=True)).float()



class BSplineTransformation(nn.Module):

    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input):
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input.is_cuda:
                self._spline_tr = self._spline_tr.cuda()
        
        return  torch.matmul(input, self._spline_tr)



class BSplineConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, degrees_of_freedom, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, log=False, scaled = True):
        super(BSplineConv1D, self).__init__()
        self._df = degrees_of_freedom
        self._log = log
        self._scaled = scaled

        self.spline = nn.Conv1d(1, degrees_of_freedom, kernel_size, stride, padding, dilation,
            bias=False)
        self.spline.weight = spline_factory(kernel_size, self._df, log=log).view(self._df, 1, kernel_size)
        if scaled:
            self.spline.weight = self.spline.weight / kernel_size            
        self.spline.weight = nn.Parameter(self.spline.weight)
        self.spline.weight.requires_grad = False
        self.conv1d = nn.Conv1d(in_channels * degrees_of_freedom, out_channels, 1, 
            groups = groups, bias=bias)

    def forward(self, input):
        batch_size, n_channels, length = input.size()
        spline_out = self.spline(input.view(batch_size * n_channels,1,length))
        conv1d_out = self.conv1d(spline_out.view(batch_size, n_channels * self._df,  length))
        return conv1d_out


class Sei(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        sequence_length : int
        """
        super(Sei, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))

        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))

        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9,padding=4),
            nn.ReLU(inplace=True))

        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
            nn.ReLU(inplace=True))

        self._spline_df = int(128/8)        
        self.spline_tr = nn.Sequential(
            nn.Dropout(p=0.5),
            BSplineTransformation(self._spline_df, scaled=False))

        self.losses = LossFunctions()
        self.f1 = F1Score(num_classes=1)


    def forward(self, x, y=None, explain=False):
        """Forward propagation of a batch.
        """
        if explain==False:
            x = F.one_hot(x.to(torch.int64), num_classes=4).to(torch.float)
        x = x.swapaxes(1,2)
        if self.training:
            if torch.randint(2, (1,)).item():
                x = torch.flip(x, dims=(1,2))
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
        
        #spline_out = self.spline_tr(out)
        #reshape_out = spline_out.view(spline_out.size(0), 960 * self._spline_df)
        
        return out.swapaxes(1,2)
