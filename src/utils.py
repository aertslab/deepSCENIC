import numpy as np
import pandas as pd
import copy

import matplotlib
import shap
import torch
import  torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm


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


def plot_deepexplainer_givenax(explainer: shap.explainers, fig, ntrack, track_no, seq_onehot, TF, TF_name, region_id, negate=False):
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

def plot_mutagenesis_givenax(model, fig, ntrack, track_no, seq_onehot, num_classes, TF, TF_name=None, region_id=None):
    seq_shape = seq_onehot.shape
    NUM_CLASSES = num_classes
    arrr_A = np.zeros((NUM_CLASSES, seq_shape[1]))
    arrr_C = np.zeros((NUM_CLASSES, seq_shape[1]))
    arrr_G = np.zeros((NUM_CLASSES, seq_shape[1]))
    arrr_T = np.zeros((NUM_CLASSES, seq_shape[1]))

    model.eval()
    
    with torch.no_grad():
        real_score = predict_mutated(model, seq_onehot)

        for mutloc in tqdm(range(seq_shape[1])):
            new_X = np.copy(seq_onehot)
            if new_X[0][mutloc, :][0] == 0:
                new_X[0][mutloc, :] = np.array([1, 0, 0, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_A[:, mutloc] = (real_score - prediction_mutated)

            if new_X[0][mutloc, :][1] == 0:
                new_X[0][mutloc, :] = np.array([0, 1, 0, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)
                arrr_C[:, mutloc] = (real_score - prediction_mutated)

            if new_X[0][mutloc, :][2] == 0:
                new_X[0][mutloc, :] = np.array([0, 0, 1, 0], dtype='int8')
                prediction_mutated = predict_mutated(model, new_X)

                arrr_G[:, mutloc] = (real_score - prediction_mutated)

            if new_X[0][mutloc, :][3] == 0:
                new_X[0][mutloc, :] = np.array([0, 0, 0, 1], dtype='int8')
                
                prediction_mutated = predict_mutated(model, new_X)
                arrr_T[:, mutloc] = (real_score - prediction_mutated)

    ax = fig.add_subplot(ntrack, 1, track_no)
    ax.set_ylabel('In silico\nMutagenesis\nTF_'+str(TF))
    if TF_name is not None:
        ax.set_title("TF_" + str(TF) + ' : ' + TF_name + ' for sequence region : ' + region_id)
    ax.scatter(range(seq_shape[1]), -1*arrr_A[TF], label='A', color='green')
    ax.scatter(range(seq_shape[1]), -1*arrr_C[TF], label='C', color='blue')
    ax.scatter(range(seq_shape[1]), -1*arrr_G[TF], label='G', color='orange')
    ax.scatter(range(seq_shape[1]), -1*arrr_T[TF], label='T', color='red')
    ax.legend()
    ax.axhline(y=0, linestyle='--', color='gray')
    ax.set_xlim((0, seq_shape[1]))
    _ = ax.set_xticks(np.arange(0, seq_shape[1]+1, 10))
    return ax

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
