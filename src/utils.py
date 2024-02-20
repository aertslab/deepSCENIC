from typing import List, Mapping, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr
import seaborn as sns
import torch
from matplotlib import cm
from scenicplus.scenicplus_class import SCENICPLUS
from tqdm import tqdm
from pycisTopic.utils import region_names_to_coordinates
from matplotlib.colors import Normalize, rgb2hex
from scenicplus.utils import Groupby

Tensor = torch.cuda.FloatTensor


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=2, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

## Deepexplainer plotting utils ##

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))

def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))

def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))

def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))

default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}

def plot_weights_given_ax(ax, array,
                          height_padding_factor,
                          length_padding,
                          subticks_frequency,
                          highlight,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if letter[1] > 0:
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    return ax

def plot_weights(array, fig, n, n1, n2, title='', ylab='',
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=20,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    ax = fig.add_subplot(n, n1, n2)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    y = plot_weights_given_ax(ax=ax, array=array,
                              height_padding_factor=height_padding_factor,
                              length_padding=length_padding,
                              subticks_frequency=subticks_frequency,
                              colors=colors,
                              plot_funcs=plot_funcs,
                              highlight=highlight)
    return fig, ax

def plot_deepexplainer_givenax(explainer, fig, ntrack, track_no, seq_onehot, TF, TF_name, region_id):
    """
        When trying to use deepexplainer with a relu, we will get the following error message:
            RuntimeError: The size of tensor a (8) must match the size of tensor b (64) at non-singleton dimension 2
        Should be fixed by https://github.com/slundberg/shap/issues/2511
    """

    shap_values_, indexes_ = explainer.shap_values(seq_onehot,
                                                   output_rank_order=str(TF),
                                                   ranked_outputs=1,
                                                   check_additivity=False)
    seq_onehot = seq_onehot.numpy()
    _, ax1 = plot_weights(shap_values_[0][0]*seq_onehot,
                          fig, ntrack, 1, track_no,
                          title="TF_" + str(TF) + ' : ' + TF_name + ' for sequence region : ' + region_id, 
                          subticks_frequency=10, ylab="DeepExplainer")
    return ax1

def plot_mutagenesis_givenax(model, fig, ntrack, track_no, seq_onehot, num_classes, TF, TF_name=None, region_id=None, seq_len=640, mutated_seq_len=640):
    NUM_CLASSES = num_classes
    arrr_A = np.zeros((NUM_CLASSES, mutated_seq_len))
    arrr_C = np.zeros((NUM_CLASSES, mutated_seq_len))
    arrr_G = np.zeros((NUM_CLASSES, mutated_seq_len))
    arrr_T = np.zeros((NUM_CLASSES, mutated_seq_len))

    model.eval()
   
    seq_start = seq_len//2-(mutated_seq_len//2)
    seq_end = seq_len//2+(mutated_seq_len//2)
    with torch.no_grad():
        real_score = predict_mutated(model, seq_onehot)

        for i, mutloc in tqdm(enumerate(range(seq_start, seq_end))):
            new_X = np.copy(seq_onehot)
            if new_X[mutloc, :][0] == 0:
                new_X[mutloc, :] = np.array([1, 0, 0, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_A[:, i] = (prediction_mutated - real_score)

            if new_X[mutloc, :][1] == 0:
                new_X[mutloc, :] = np.array([0, 1, 0, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_C[:, i] = (prediction_mutated - real_score)

            if new_X[mutloc, :][2] == 0:
                new_X[mutloc, :] = np.array([0, 0, 1, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_G[:, i] = (prediction_mutated - real_score)

            if new_X[mutloc, :][3] == 0:
                new_X[mutloc, :] = np.array([0, 0, 0, 1], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_T[:, i] = (prediction_mutated - real_score)

    ax = fig.add_subplot(ntrack, 1, track_no)
    ax.set_ylabel('In silico\nMutagenesis\nTF_'+str(TF))
    if TF_name is not None:
        ax.set_title("TF_" + str(TF) + ' : ' + TF_name + ' for sequence region : ' + region_id)
    ax.scatter(range(mutated_seq_len), arrr_A[TF], label='A', color='green')
    ax.scatter(range(mutated_seq_len), arrr_C[TF], label='C', color='blue')
    ax.scatter(range(mutated_seq_len), arrr_G[TF], label='G', color='orange')
    ax.scatter(range(mutated_seq_len), arrr_T[TF], label='T', color='red')
    ax.legend()
    ax.axhline(y=0, linestyle='--', color='gray')
    ax.set_xlim((0, mutated_seq_len))
    _ = ax.set_xticks(np.arange(0, mutated_seq_len+1, 10))
    return ax

def plot_TF_ism_givenax(model, fig, ntrack, track_no, seq_onehot, num_classes, TF, TF_name=None, region_id=None, seq_len=640, mutated_seq_len=640):
    NUM_CLASSES = num_classes
    arrr_A = np.zeros((NUM_CLASSES, mutated_seq_len))
    arrr_C = np.zeros((NUM_CLASSES, mutated_seq_len))
    arrr_G = np.zeros((NUM_CLASSES, mutated_seq_len))
    arrr_T = np.zeros((NUM_CLASSES, mutated_seq_len))

    model.eval()
   
    seq_start = seq_len//2-(mutated_seq_len//2)
    seq_end = seq_len//2+(mutated_seq_len//2)
    with torch.no_grad():
        real_score = predict_mutated(model, seq_onehot)

        for i, mutloc in tqdm(enumerate(range(seq_start, seq_end))):
            new_X = np.copy(seq_onehot)
            if new_X[mutloc, :][0] == 0:
                new_X[mutloc, :] = np.array([1, 0, 0, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_A[:, i] = (prediction_mutated - real_score)

            if new_X[mutloc, :][1] == 0:
                new_X[mutloc, :] = np.array([0, 1, 0, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_C[:, i] = (prediction_mutated - real_score)

            if new_X[mutloc, :][2] == 0:
                new_X[mutloc, :] = np.array([0, 0, 1, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_G[:, i] = (prediction_mutated - real_score)

            if new_X[mutloc, :][3] == 0:
                new_X[mutloc, :] = np.array([0, 0, 0, 1], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_T[:, i] = (prediction_mutated - real_score)

    return arrr_A, arrr_C, arrr_G, arrr_T

    # ax = fig.add_subplot(ntrack, 1, track_no)
    # ax.set_ylabel('In silico\nMutagenesis\nTF_'+str(TF))
    # if TF_name is not None:
    #     ax.set_title("TF_" + str(TF) + ' : ' + TF_name + ' for sequence region : ' + region_id)
    # ax.scatter(range(mutated_seq_len), arrr_A[TF], label='A', color='green')
    # ax.scatter(range(mutated_seq_len), arrr_C[TF], label='C', color='blue')
    # ax.scatter(range(mutated_seq_len), arrr_G[TF], label='G', color='orange')
    # ax.scatter(range(mutated_seq_len), arrr_T[TF], label='T', color='red')
    # ax.legend()
    # ax.axhline(y=0, linestyle='--', color='gray')
    # ax.set_xlim((0, mutated_seq_len))
    # _ = ax.set_xticks(np.arange(0, mutated_seq_len+1, 10))
    # return ax

def plot_prediction_givenax(model, fig, ntrack, track_no, seq_onehot, num_classes):
    NUM_CLASSES = num_classes
    model.eval()
    real_score = model(seq_onehot)[0]
    with torch.no_grad():
        ax = fig.add_subplot(ntrack, 2, track_no*2-1)
        ax.margins(x=0)
        ax.set_ylabel('Prediction', color='red')
        ax.plot(real_score, '--', color='gray', linewidth=3)
        ax.scatter(range(NUM_CLASSES), real_score, marker='o', color='red', linewidth=11)
        ax.tick_params(axis='y', labelcolor='red')
        ax.set_xticks(range(NUM_CLASSES),)
        ax.set_xticklabels(range(1, NUM_CLASSES+1))
        ax.grid(True)
        return ax

def predict_mutated(model, new_X):
    float_x = torch.FloatTensor(new_X)
    if torch.cuda.is_available():
        float_x = float_x.type(Tensor)

    prediction_mutated = model(float_x)[0]
    if torch.cuda.is_available():
        prediction_mutated = prediction_mutated.cpu()

    return prediction_mutated.numpy()

def _region_to_chrom_start_end(x): return [x.replace(':', '-').split('-')[0],
                                          int(x.replace(
                                              ':', '-').split('-')[1]),
                                          int(x.replace(':', '-').split('-')[2])]

def activity_plot(SCENICPLUS_obj: SCENICPLUS,
                  bw_dict: Mapping[str, str],
                  region: str,
                  genes_violin_plot: Union[str, List] = None,
                  genes_arcs: Union[str, List] = None,
                  gene_height: int = 1,
                  exon_height: int = 4,
                  meta_data_key: str = None,
                  pr_consensus_bed: pr.PyRanges = None,
                  region_bed_height: int = 1,
                  pr_gtf: pr.PyRanges = None,
                  pr_interact: pr.PyRanges = None,
                  bw_ymax: float = None,
                  bw_ymin: float = None,
                  color_dict: Mapping[str, str] = None,
                  cmap='tab20',
                  plot_order: list = None,
                  figsize: tuple = (6, 8),
                  fontsize_dict={'bigwig_label': 9,
                                 'gene_label': 9,
                                 'violinplots_xlabel': 9,
                                 'violinplots_ylabel': 9,
                                 'title': 12,
                                 'bigwig_tick_label': 8},
                  gene_label_offset=3,
                  arc_rad=0.5,
                  arc_lw=1,
                  cmap_violinplots='Greys',
                  violinplots_means_color='black',
                  violinplots_edge_color='black',
                  violoinplots_alpha=1,
                  width_ratios_dict={'bigwig': 3,
                                     'violinplots': 1},
                  height_ratios_dict={'bigwig_violin': 1,
                                      'genes': 0.5,
                                      'arcs': 5,
                                      'custom_ax': 2},
                  sort_vln_plots: bool = False,
                  add_custom_ax: int = None) -> plt.Figure:
    fig_nrows = len(bw_dict.keys())
    if pr_interact is not None:
        fig_nrows += 1
    if pr_gtf is not None:
        fig_nrows += 1
    if add_custom_ax:
        fig_nrows += add_custom_ax
    height_ratios = [height_ratios_dict['bigwig_violin']
                     for i in range(len(bw_dict.keys()))]
    if pr_gtf is not None:
        height_ratios += [height_ratios_dict['genes']]
    if pr_interact is not None:
        height_ratios += [height_ratios_dict['arcs']]
    if add_custom_ax is not None:
        height_ratios += [height_ratios_dict['custom_ax']
                          for i in range(add_custom_ax)]

    if genes_violin_plot is not None:
        ncols = 2
        width_ratios = [width_ratios_dict['bigwig'],
                        width_ratios_dict['violinplots']]
    else:
        ncols = 1
        width_ratios = [1]
    fig, axs = plt.subplots(nrows=fig_nrows,
                            ncols=ncols,
                            gridspec_kw={
                                'height_ratios': height_ratios, 'width_ratios': width_ratios},
                            figsize=figsize)

    if genes_violin_plot is not None:
        axs_vln = axs[:, 1]
        axs_bw = axs[:, 0]
    else:
        axs_bw = axs

    subplot_idx = 0
    # get coordinates from region string
    chrom, start, end = _region_to_chrom_start_end(region)
    pr_region = pr.PyRanges(chromosomes=[chrom], starts=[start], ends=[end])
    # calculate bw_ymax for scaling
    bw_ymax_dict = {}
    bw_ymin_dict = {}
    for key in bw_dict.keys():
        bw_file = bw_dict[key]
        # calculate max value of the bigwig within our region
        bw = pyBigWig.open(bw_file)
        y = bw.values(chrom, start, end)
        y = np.nan_to_num(y)
        bw_ymax_dict[key] = y.max()
        bw_ymin_dict[key] = y.min()

    x = np.array(range(start, end, 1))

    # set y_max and min
    if bw_ymax is None:
        bw_ymax = max(bw_ymax_dict.values())
    if bw_ymin is None:
        bw_ymin = min(bw_ymin_dict.values())

    # set colors if not provided
    if color_dict is None:
        cmap = cm.get_cmap(cmap)
        color_dict = {k: cmap(i) for i, k in enumerate(bw_dict.keys())}

    # iterate over all bigwigs
    if plot_order is None:
        plot_order = bw_dict.keys()
    for key in plot_order:
        # create a new subplot
        ax = axs_bw[subplot_idx]
        subplot_idx += 1

        # open the bigwig
        bw_file = bw_dict[key]

        bw = pyBigWig.open(bw_file)
        y = bw.values(chrom, start, end)
        y = np.nan_to_num(y)

        # now plot the bigwig in the gridspec
        ax.fill_between(x, y1=y, y2=0, step="mid",
                        linewidth=0, color=color_dict[key])
        ax.patch.set_alpha(0)

        # figure settings
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim(
            [bw_ymin, bw_ymax]) if pr_consensus_bed is not None else ax.set_ylim([bw_ymin, bw_ymax])
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major',
                       labelsize=fontsize_dict['bigwig_tick_label'])

        ax.text(x=x.min(), y=bw_ymax + 1, s=key,
                fontsize=fontsize_dict['bigwig_label'])

        sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)

        if pr_consensus_bed is not None:
            consensus_bed_region_intersect = pr_consensus_bed.intersect(
                pr_region)
            for _, r in consensus_bed_region_intersect.df.iterrows():
                region_names = [x.split('_peak')[0]
                                for x in r['Name'].split(',')]
                if key in region_names:
                    bed_chrom = r['Chromosome']
                    bed_start = r['Start']
                    bed_end = r['End']
                    if pr_interact is not None:
                        pr_tmp = pr.PyRanges(chromosomes=[bed_chrom], starts=[
                                             bed_start], ends=[bed_end])
                        if len(pr_interact.intersect(pr_tmp)) > 0:
                            color = 'black'
                        else:
                            color = 'grey'
                    else:
                        color = 'grey'
                    rect = mpatches.Rectangle(
                        (bed_start, -2), bed_end-bed_start, region_bed_height, fill=True, color=color, linewidth=0)
                    ax.add_patch(rect)

    # draw the genes of interest, from our gtf
    # intersect genes gtf with the region of interest
    if pr_gtf is not None:
        # intersect gtf with region and get first 9 columns
        gtf_region_intersect = pr_gtf.intersect(pr_region)
        # only keep exon and gene info
        gtf_region_intersect = gtf_region_intersect[np.logical_and(
            np.logical_or(gtf_region_intersect.Feature == 'gene',
                          gtf_region_intersect.Feature == 'exon'),
            gtf_region_intersect.gene_type == 'protein_coding')]
        # iterate over all genes in intersect
        ax = axs_bw[subplot_idx]
        subplot_idx += 1
        genes_in_window = set(gtf_region_intersect.gene_name)
        n_genes_in_window = len(genes_in_window)
        for idx, _gene in enumerate(genes_in_window):
            _gene_height = gene_height / n_genes_in_window
            _gene_bottom = -gene_height/2 + _gene_height * idx
            _exon_bottom = _gene_bottom - \
                (((exon_height / n_genes_in_window) / 2) - _gene_height)
            _exon_height = (((exon_height / n_genes_in_window) / 2) - _gene_height) + \
                _gene_height + \
                (((exon_height / n_genes_in_window) / 2) - _gene_height)
            # plot the gene parts (gene body and gene exon)
            # iterate over all parts to plot them
            for _, part in gtf_region_intersect.df.loc[gtf_region_intersect.df['gene_name'] == _gene].iterrows():
                # make exons thick
                if part['Feature'] == 'exon':
                    exon_start = part['Start']
                    exon_end = part['End']
                    # draw rectangle for exon
                    rect = mpatches.Rectangle(
                        (exon_start, _exon_bottom), exon_end-exon_start, _exon_height, fill=True, color="k", linewidth=0)
                    ax.add_patch(rect)
                # make the gene body a thin line, drawn at the end so it will always display on top
                elif part['Feature'] == 'gene':
                    gene_start = part['Start']
                    gene_end = part['End']
                    rect = mpatches.Rectangle(
                        (gene_start, _gene_bottom), gene_end-gene_start, _gene_height, fill=True, color="k", linewidth=0)
                    ax.add_patch(rect)
                ax.text(gene_start, _gene_bottom - gene_label_offset,_gene, fontsize=fontsize_dict['gene_label'])
            # figure settings
            ax.set_ylim([-exon_height/2, exon_height/2])
            #ax.set_xlabel(gene, fontsize=10)
            ax.set_xlim([x.min(), x.max()])
            sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_alpha(0)  # make sure that each individual subplot is transparent! otherwise the underlying plots won't be shown. this is important e.g. for the dar/peak visualisaton, since the DARs are drawn directly on top of the peaks. if the DAR plot is not transparent, no peaks will be visible!!

    # Draw arcs from region to gene
    if pr_interact is not None and pr_gtf is not None:
        ax = axs_bw[subplot_idx]
        subplot_idx += 1
        # intersect region with pr_interact
        pr_region = pr.PyRanges(
            chromosomes=[chrom], starts=[start], ends=[end])
        region_interact_intersect = pr_interact.intersect(pr_region)
        # only keep interactions to genes within window
        if genes_arcs is not None:
            if type(genes_arcs) == str:
                genes_arcs = [genes_arcs]
            region_interact_intersect = region_interact_intersect[np.isin(
                region_interact_intersect.targetName, list(set(genes_in_window) & set(genes_arcs)))]
        else:
            region_interact_intersect = region_interact_intersect[np.isin(
                region_interact_intersect.targetName, list(genes_in_window))]
        for _, r2g in region_interact_intersect.df.sort_values('value').iterrows():
            posA = (int(r2g['sourceStart']), 0)
            posB = (int(r2g['targetStart']), 0)
            # this to ensure arcs are always down
            sign = '-' if posA[0] > posB[0] else ''
            color = r2g['color']
            if (posA[0] > x.min() and posA[0] < x.max()) and (posB[0] > x.min() and posB[0] < x.max()):
                arrow = mpatches.FancyArrowPatch(posA=posA,
                                                 posB=posB,
                                                 connectionstyle=f"arc3,rad={sign}{arc_rad}",
                                                 color=color,
                                                 lw=arc_lw)
                ax.add_patch(arrow)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim(-1, 0)
        sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])

    if genes_violin_plot is not None:
        # plot expression of gene/genes
        if meta_data_key not in SCENICPLUS_obj.metadata_cell.columns:
            raise ValueError(f'key {meta_data_key} not found in SCENICPLUS_obj.metadata_cell.columns')
            
        # check that bw_dict keys is a subset of scplus_obj.metadata_cell[meta_data_key]
        annotations = SCENICPLUS_obj.metadata_cell[meta_data_key].to_numpy()
        if len(set(plot_order) - set(annotations)) != 0:
            not_found = set(plot_order) - set(annotations)
            raise ValueError(f'Following keys were not found in SCENICPLUS_obj.metadata_cell[{meta_data_key}]\n{", ".join(not_found)}')

        # normalize scores
        mtx = np.log1p((SCENICPLUS_obj.X_EXP.T /
                       SCENICPLUS_obj.X_EXP.sum(1) * (10 ** 4)).T)

        # get expression values for gene/genes
        idx_gene = list(SCENICPLUS_obj.gene_names).index(genes_violin_plot) if type(
            genes_violin_plot) == str else [list(SCENICPLUS_obj.gene_names).index(g) for g in genes_violin_plot]
        expr_vals = mtx[:, idx_gene]

        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(plot_order))
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_violinplots)
        expr_min = np.Inf
        expr_max = np.NINF
        for idx, annotation in enumerate(plot_order):
            ax = axs_vln[idx]
            cells = np.where(annotations == annotation)
            expr_vals_sub = expr_vals[cells]

            if sort_vln_plots:
                idx_sorted = np.argsort(expr_vals_sub.mean(0))[::-1]
                expr_vals_sub = expr_vals_sub[:, idx_sorted]
                genes_violin_plot = np.array(
                    list(genes_violin_plot))[idx_sorted]

            expr_min = min(expr_min, expr_vals_sub.min())
            expr_max = max(expr_max, expr_vals_sub.max())

            vln_plot_part = ax.violinplot(expr_vals_sub, vert=False, showmeans=True, showextrema=False)
            for i, pc in enumerate(vln_plot_part['bodies']):
                facecolor = mapper.to_rgba(i)
                pc.set_facecolor(color_dict[annotation])
                pc.set_edgecolor(violinplots_edge_color)
                pc.set_alpha(violoinplots_alpha)
            vln_plot_part['cmeans'].set_edgecolor(violinplots_means_color)
            n_labels = 1 if type(genes_violin_plot) == str else len(genes_violin_plot)
            ax.set_yticks(ticks=np.arange(1, n_labels + 1))
            ax.set_yticklabels(genes_violin_plot, fontsize=fontsize_dict['violinplots_ylabel'])
            ax.set_xlim(xmin=round(expr_min), xmax=round(expr_max + 0.5))
            if idx < len(plot_order) - 1:
                sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
                ax.set_xticks([])
            else:
                sns.despine(top=True, right=True, left=True, bottom=False, ax=ax)
                ax.set_xlabel('log-normalised\nExpression counts', fontsize=fontsize_dict['violinplots_xlabel'])
        for i in range(idx + 1, len(axs_vln)):
            ax = axs_vln[i]
            sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])

    # finally, add a little text that shows which regions you're plotting
    length = round((end-start)/1000)
    label = f'{region} ({length} kb)'

    fig.suptitle(label, fontsize=fontsize_dict['title'])

    if add_custom_ax:
        return axs[axs.shape[0] - add_custom_ax: axs.shape[0]], fig
    else:
        return fig

def generate_bigWig(regs, scores, chrs_size_file, save_dir):
    """
    Generate bigWig file from input data

    Args:
        regs (list): List of genomic regions in the format 'chr:start-end'
        scores (list): List of scores corresponding to each region
        chrs_size_file (str): Path to a file containing chromosome sizes
        save_dir (str): Directory where the bigWig file will be saved
    """    
    chrs_ = []
    starts = []
    ends = []
    chr_nrs = []

    for r in regs:
        chr_, coords = r.split(':')
        start, end = coords.split('-')
        chrs_.append(chr_)
        chr_nrs.append(chr_[3:])
        starts.append(start)
        ends.append(end)

    print('\nCreating dataframe...\n')
    df = pd.DataFrame(
        {
        'Chromosome': chrs_,
        'Start': starts,
        'End': ends,
        'Score':scores,
        'Chromosome number':chr_nrs
        }
    )

    df = df.sort_values(by =['Chromosome number','Start'])
    df = df.drop(['Chromosome number'], axis = 1)
    gr = pr.from_dict(df)
    df_chrom = pd.read_csv(chrs_size_file, sep="\t", names=['Chromosome','End'])
    df_chrom.insert(1, "Start", [0]*df_chrom.shape[0])

    print(df_chrom)
    g = pr.from_dict(df_chrom)
    gr = gr.sort()
    g = g.sort()
    print(gr)
    print(g)

    print('Writing to Bigwig')
    fname = save_dir + ".bw"
    pr.to_bigwig(gr, fname, g)
    print('Done!')

ASM_SYNONYMS = {
    'hg38': 'GRCh38',
    'hg19': 'GRCh37',
    'mm9': 'MGSCv37',
    'mm10': 'GRCm38',
    'mm39': 'GRCm39',
    'dm6': 'BDGP6',
    'galGal6': 'GRCg6a'}

def get_interaction_pr(region_to_gene_df,
                       species,
                       assembly,
                       pbm_host='http://www.ensembl.org',
                       key_for_color='r2g_score',
                       cmap_pos='Blues',
                       cmap_neg='Reds',
                       scale_by_gene=True,
                       vmin=0, vmax=1):

    region_to_gene_df = region_to_gene_df.rename(columns={'gene':'Gene'})
    import pybiomart as pbm
    dataset_name = '{}_gene_ensembl'.format(species)
    server = pbm.Server(host=pbm_host, use_cache=False)
    mart = server['ENSEMBL_MART_ENSEMBL']
    dataset_display_name = getattr(mart.datasets[dataset_name], 'display_name')
    if not (ASM_SYNONYMS[assembly] in dataset_display_name or assembly in dataset_display_name):
        print(
            f'\u001b[31m!! The provided assembly {assembly} does not match the biomart host ({dataset_display_name}).\n Please check biomart host parameter\u001b[0m\nFor more info see: https://m.ensembl.org/info/website/archives/assembly.html')
    dataset = mart[dataset_name]
    if 'external_gene_name' not in dataset.attributes.keys():
        external_gene_name_query = 'hgnc_symbol'
    else:
        external_gene_name_query = 'external_gene_name'
    if 'transcription_start_site' not in dataset.attributes.keys():
        transcription_start_site_query = 'transcript_start'
    else:
        transcription_start_site_query = 'transcription_start_site'
    annot = dataset.query(attributes=['chromosome_name',
                                      'start_position',
                                      'end_position',
                                      'strand',
                                      external_gene_name_query,
                                      transcription_start_site_query,
                                      'transcript_biotype'])
    annot.columns = ['Chromosome', 'Start', 'End', 'Strand',
                     'Gene', 'Transcription_Start_Site', 'Transcript_type']
    annot['Chromosome'] = 'chr' + \
        annot['Chromosome'].astype(str)
    annot = annot[annot.Transcript_type == 'protein_coding']
    annot.Strand[annot.Strand == 1] = '+'
    annot.Strand[annot.Strand == -1] = '-'       
    annot['TSSeqStartEnd'] = np.logical_or(
        annot['Transcription_Start_Site'] == annot['Start'], annot['Transcription_Start_Site'] == annot['End'])
    gene_to_tss = annot[['Gene', 'Transcription_Start_Site']].groupby(
        'Gene').agg(lambda x: list(map(str, x)))
    startEndEq = annot[['Gene', 'TSSeqStartEnd']
                       ].groupby('Gene').agg(lambda x: list(x))
    gene_to_tss['Transcription_Start_Site'] = [np.array(tss[0])[eq[0]][0] if sum(
        eq[0]) >= 1 else tss[0][0] for eq, tss in zip(startEndEq.values, gene_to_tss.values)]
    gene_to_tss.columns = ['TSS_Gene']

    # get gene to strand mapping
    gene_to_strand = annot[['Gene', 'Strand']].groupby(
        'Gene').agg(lambda x: list(map(str, x))[0])

    # get gene to chromosome mapping (should be the same as the regions mapped to the gene)
    gene_to_chrom = annot[['Gene', 'Chromosome']].groupby(
        'Gene').agg(lambda x: list(map(str, x))[0])

    # add TSS for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_tss, on='Gene')

    # add strand for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_strand, on='Gene')

    # add chromosome for each gene to region_to_gene_df
    region_to_gene_df = region_to_gene_df.join(gene_to_chrom, on='Gene')

    region_to_gene_df.dropna(axis=0, how='any', inplace=True)
    arr = region_names_to_coordinates(region_to_gene_df['region']).to_numpy()
    chrom, chromStart, chromEnd = np.split(arr, 3, 1)
    chrom = chrom[:, 0]
    chromStart = chromStart[:, 0]
    chromEnd = chromEnd[:, 0]

    sourceChrom = chrom
    sourceStart = np.array(
        list(map(int, chromStart + (chromEnd - chromStart)/2 - 1)))
    sourceEnd = np.array(
        list(map(int, chromStart + (chromEnd - chromStart)/2)))

    # get target chrom, chromStart, chromEnd (i.e. TSS)
    targetChrom = region_to_gene_df['Chromosome']
    targetStart = region_to_gene_df['TSS_Gene'].values
    targetEnd = list(map(str, np.array(list(map(int, targetStart))) + np.array(
        [1 if strand == '+' else -1 for strand in region_to_gene_df['Strand'].values])))

    norm = Normalize(vmin=vmin, vmax=vmax)

    if scale_by_gene:
        grouper = Groupby(
            region_to_gene_df.loc[region_to_gene_df['r2g_score'] >= 0, 'Gene'].to_numpy())
        scores = region_to_gene_df.loc[region_to_gene_df['r2g_score']
                                       >= 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_pos)

        def _value_to_color(scores):
            S = (scores - scores.min()) / (scores.max() - scores.min())
            return [rgb2hex(mapper.to_rgba(s)) for s in S]

        colors_pos = np.zeros(len(scores), dtype='object')
        for idx in grouper.indices:
            colors_pos[idx] = _value_to_color(scores[idx])

        scores = region_to_gene_df.loc[region_to_gene_df['r2g_score']
                                       < 0, key_for_color].to_numpy()
    else:
        scores = region_to_gene_df.loc[region_to_gene_df['r2g_score']
                                       >= 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_pos)
        colors_pos = [rgb2hex(mapper.to_rgba(s)) for s in scores]

        scores = region_to_gene_df.loc[region_to_gene_df['r2g_score']
                                       < 0, key_for_color].to_numpy()
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_neg)

    region_to_gene_df.loc[region_to_gene_df['r2g_score'] >= 0, 'color'] = colors_pos
    region_to_gene_df['color'] = region_to_gene_df['color'].fillna('#525252')

    # get name for regions (add incremental number to gene in range of regions linked to gene)
    counter = 1
    previous_gene = region_to_gene_df['Gene'].values[0]
    names = []
    for gene in region_to_gene_df['Gene'].values:
        if gene != previous_gene:
            counter = 1
        else:
            counter += 1
        names.append(gene + '_' + str(counter))
        previous_gene = gene
    df_interact = pd.DataFrame(
        data={
            'Chromosome':        chrom,
            'Start':   chromStart,
            'End':     chromEnd,
            'name':         names,
            'score':        np.repeat(0, len(region_to_gene_df)),
            'value':        region_to_gene_df['r2g_score'].values,
            'exp':          np.repeat('.', len(region_to_gene_df)),
            'color':        region_to_gene_df['color'].values,
            'sourceChrom':  sourceChrom,
            'sourceStart':  sourceStart,
            'sourceEnd':    sourceEnd,
            'sourceName':   names,
            'sourceStrand': np.repeat('.', len(region_to_gene_df)),
            'targetChrom':  targetChrom,
            'targetStart':  targetStart,
            'targetEnd':    targetEnd,
            'targetName':   region_to_gene_df['Gene'].values,
            'targetStrand': region_to_gene_df['Strand'].values
        }
    )
    pr_interact = pr.PyRanges(df_interact)

    return pr_interact

def format_region_to_bed(region_string, output_file, seq_len=500):
    chrom, start_end = region_string.split(":")
    start, end = start_end.split("-")
    if (int(end)-int(start)) < seq_len:
        center = int(start) + ((int(end)-int(start))//2)
        start = center - seq_len//2
        end = center + seq_len//2
    bed_string = f"{chrom}\t{start}\t{end}\n"
    output_file.write(bed_string)
