import os

import math
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional  as F

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_array, load_npz

from src.model import VAE
from src.utils import EarlyStopping
from src.tf2rNet.model import TF2rNet, final
from src.tf2rNet.utils import *

from tqdm import tqdm
import datetime
import re
import json

## Build dataloaders ##
def build_seq_dataloader(opt, regions, split=False):
    """ Build dataloader for TF2r network.
        
        Params
        ------
        regions: List of regions for training/testing.
        split: wheather to split train/test set or not (True/False).
    """

    class TensorDatasetWithIndex(Dataset[Tuple[torch.Tensor, ...]]):
        tensors: Tuple[torch.Tensor, ...]

        def __init__(self, *tensors: torch.Tensor) -> None:
            assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
            self.tensors = tensors

        def __getitem__(self, index):
            return (tuple(tensor[index] for tensor in self.tensors), index)

        def __len__(self):
            return self.tensors[0].size(0)

    peak = [re.split("[\-:]+", elem) for elem in regions]
    peak = pd.DataFrame(peak, columns=['chr', 'start', 'end'])

    if split:
        train_ids, test_ids = split_train_test_val(np.arange(ad.shape[1]))

        f = h5py.File('%s/splits.h5'%opt.save_name, "w")
        f.create_dataset("train_ids", data=train_ids)
        f.create_dataset("test_ids", data=test_ids)
        f.close()

        make_h5_sparse(peak[:, train_ids] , '%s/train_seqs.h5'%opt.save_name, opt.fasta, opt.seq_len, opt.TF2rNet_batch_size)
        make_h5_sparse(peak[:, test_ids], '%s/test_seqs.h5'%opt.save_name, opt.fasta, opt.seq_len, opt.TF2rNet_batch_size)

        with h5py.File('%s/splits.h5'%opt.save_name, 'r') as hf:
            train_ids = hf['train_ids'][:]
            test_ids = hf['test_ids'][:]

        with h5py.File('%s/train_seqs.h5'%opt.save_name, 'r') as hf:
            X_train = hf['X'][:].astype('float32')

        with h5py.File('%s/test_seqs.h5'%opt.save_name, 'r') as hf:
            X_test = hf['X'][:].astype('float32')

        m_train = ad[:, train_ids].X
        m_test = ad[:, test_ids].X
       
        train_data = TensorDatasetWithIndex(torch.FloatTensor(X_train), torch.FloatTensor(m_train.T))
        train_dataloader =  DataLoader(train_data, batch_size=opt.TF2rNet_batch_size, shuffle=False, num_workers=0)
        train_dataloader_shuffle =  DataLoader(train_data, batch_size=opt.TF2rNet_batch_size, shuffle=True, num_workers=0)

        test_data = TensorDatasetWithIndex(torch.FloatTensor(X_test), torch.FloatTensor(m_test.T))
        test_dataloader =  DataLoader(test_data, batch_size=opt.TF2rNet_batch_size, shuffle=False, num_workers=0)
        test_dataloader_shuffle =  DataLoader(test_data, batch_size=opt.TF2rNet_batch_size, shuffle=True, num_workers=0)
    else:
        make_h5_sparse(peak, '%s/all_seqs.h5'%opt.save_name, opt.fasta, opt.seq_len, opt.TF2rNet_batch_size)
        with h5py.File('%s/all_seqs.h5'%opt.save_name, 'r') as hf:
            X = hf['X'][:].astype('float32')

        train_data = TensorDatasetWithIndex(torch.FloatTensor(X))
        train_dataloader =  DataLoader(train_data, batch_size=opt.TF2rNet_batch_size, shuffle=False, num_workers=0)
        train_dataloader_shuffle =  DataLoader(train_data, batch_size=opt.TF2rNet_batch_size, shuffle=True, num_workers=0)
        test_dataloader = None
        test_dataloader_shuffle = None

    return train_dataloader, test_dataloader, train_dataloader_shuffle, test_dataloader_shuffle
 
def build_dataloader(data_rna, data_atac, opt):
    """ Build dataloader for VAE
  
        Params
        ------
        data_rna: scRNA-seq data
        data_atac: scATAC-seq data
        opt: model hyperparams
    """
    rna_gene_name = list(data_rna.var_names)
    atac_region_name = list(data_atac.var_names)

    # Check if sparse data
    if type(data_rna.layers['log_norm'])!=np.ndarray:
        data_rna.layers['log_norm'] = data_rna.layers['log_norm'].toarray()

    if type(data_atac.X)!=np.ndarray:
        data_atac.X = data_atac.X.toarray()

    # Build gene exp. dropout mask
    Dropout_Mask_rna = (data_rna.layers['log_norm'] != 0).astype(np.float) # binary mask of exp matrix

    # Get scaled values
    data_rna = pd.DataFrame(data_rna.layers['Z_scaled'], index=list(data_rna.obs_names), columns=rna_gene_name)
    data_atac = pd.DataFrame(data_atac.layers['Z_scaled'], index=list(data_atac.obs_names), columns=atac_region_name)

    num_genes_rna = data_rna.shape[1]
    num_regions_atac = data_atac.shape[1]

    feat_rna = torch.FloatTensor(data_rna.values)
    feat_atac = torch.FloatTensor(data_atac.values) 
    data = TensorDataset(feat_rna, feat_atac, torch.LongTensor(list(range(len(feat_rna)))),
                         torch.FloatTensor(Dropout_Mask_rna))

    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=opt.train, num_workers=1)

    return {'dataloader': dataloader, 'num_genes_rna': num_genes_rna, 'num_regions_atac': num_regions_atac, 'data_rna': data_rna, 'data_atac': data_atac, 'rna_gene_name': rna_gene_name, 'atac_region_name':atac_region_name}


## deepSCENIC ##
class deepSCENIC:
    def __init__(self, opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')

    def init_data(self):
        # Read TFs list
        TFs_df = pd.read_csv(self.opt.TF_file, header=None, names=["name"], usecols=[0])
        TFs = set(TFs_df['name'])

        # Read region to gene mask
        r2g_dist_coo = load_npz(self.opt.r2g_mask)

        print("reading data...")
        # Read multimodal data
        data_rna = sc.read(self.opt.data_rna_file)
        data_atac = sc.read(self.opt.data_atac_file)
        print("data read!")
        print(data_rna)
        print(data_atac)

        TFs_idx = np.where(data_rna.var_names.isin(TFs))[0]

        # Check if sparse data
        if type(data_rna.layers['log_norm'])!=np.ndarray:
            data_rna_values = data_rna.layers['log_norm'].toarray()
        else:
            data_rna_values = data_rna.layers['log_norm']
        if type(data_atac.X)!=np.ndarray:
            data_atac_values = data_atac.X.toarray()
        else:
            data_atac_values = data_atac.X

        # Binarize ATAC data
        if self.opt.bin_acc==True:
            data_atac_values[data_atac_values > 0] = 1

        # normalize expression data    
        data_rna_values = data_rna_values / data_rna_values.std(0)
        if self.opt.bin_acc==False:
            data_atac_values = data_atac_values / data_atac_values.std(0)

        data_rna.layers['Z_scaled'] = data_rna_values
        data_atac.layers['Z_scaled'] = data_atac_values

        # Split rna dataset in train and test
        if self.opt.test_size > 0:
            cell_idxs = data_rna.obs.index.values
            cell_idxs_train, cell_idxs_test = train_test_split(cell_idxs, test_size=self.opt.test_size, random_state=42)
            test_rna = data_rna[cell_idxs_test,:].copy()
            data_rna = data_rna[cell_idxs_train,:].copy()
            test_atac = data_atac[cell_idxs_test,:].copy()
            data_atac = data_atac[cell_idxs_train,:].copy()

        # Build dataloaders
        train_dataloader = build_dataloader(data_rna, data_atac, self.opt)        
        if self.opt.test_size > 0:
            test_dataloader = build_dataloader(test_rna, test_atac, self.opt)
        else:
            test_dataloader = None

        if self.opt.tf2r_df:
            regions = pd.read_pickle(self.opt.tf2r_df).columns.to_list()
        else:
            regions = data_atac.var_names.to_list()
        
        train_seq_dataloader, _, train_seq_data_shuffle, _ = build_seq_dataloader(self.opt, regions=regions, split=False)

        return train_dataloader, test_dataloader, TFs_idx, r2g_dist_coo, train_seq_dataloader, train_seq_data_shuffle

    def pretrain_TF2r(self, tf2r_df, n_epochs=200, patience=10):
        """ Pretrain TF2r data on the tf2r_df dataframe. Can be any kind of region data
            (scATAC-seq, Chip-seq, pycisTarget output,...).
   
            Params
            ------
            tf2r_df: Region dataframe for training.
            n_epochs: Number of epochs
            patience: patience parameter for early stopping
        """
        
        opt = self.opt
        if opt.device=='cuda':
            Tensor = torch.cuda.FloatTensor
        elif opt.device=='cpu':
            Tensor = torch.FloatTensor
        early_stopping = EarlyStopping(patience=patience)
        writer = SummaryWriter(opt.logs + '/logs/TF2rNet/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        ad_tf2r = sc.AnnData(tf2r_df)
        _, _, train_seq_data_shuffle, test_seq_data_shuffle = build_seq_dataloader(self.opt, ad=ad_tf2r, split=True)
        n_TFs = ad_tf2r.shape[0]

        model = TF2rNet(opt.TF2rNet_bottleneck_size, n_TFs).float().to(opt.device)
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)

        def tf2r_loss(pred, real):
            if opt.TF2rNet_loss == 'mse':
                return torch.mean((real - pred).pow(2))
            elif opt.TF2rNet_loss == 'bce':
                F.binary_cross_entropy_with_logits(predicted, real, reduction='none').mean()
            else:
                raise NameError('Loss function must be either mse or bce!')
        loss_tf2rnet = tf2r_loss

        # Training
        for epoch in range(n_epochs):
            model.train()
            loss_l = []
            for i, (data_batch, data_batch_idx) in tqdm(enumerate(train_seq_data_shuffle, 0), unit="batch", total=len(train_seq_data_shuffle)):
                optimizer.zero_grad()
                X,y = data_batch
                X = Variable(X.type(Tensor).to(opt.device))
                pred, _ = model(X)

                loss = tf2r_loss(pred, y.to(opt.device))
                loss.backward()

                optimizer.step()

                loss_l.append(loss.detach().item())
                # Tensorboard logs
                n_iter = (epoch*len(train_seq_data_shuffle)) + i
                writer.add_scalar('Loss/train', loss.item(), n_iter)
                del loss
            print('epoch:', epoch, 'loss:', np.mean(loss_l))

            # Evaluate validation set
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, opt.save_name + '/TF2rNET_model_pretrain.pth')

            with torch.no_grad():
                model.eval()
                loss_l = []
                for i, (data_batch, data_batch_idx) in tqdm(enumerate(test_seq_data_shuffle, 0), unit="batch", total=len(test_seq_data_shuffle)):
                    X,y = data_batch
                    X = Variable(X.type(Tensor).to(opt.device))
                    pred, _ = model(X)
                    loss = tf2r_loss(pred, y.to(opt.device))
                    loss_l.append(loss.detach().item())
                    del loss

                print('epoch:', epoch, 'test_loss:', np.mean(loss_l))
                writer.add_scalar('Loss/val', np.mean(loss_l), epoch)

                early_stopping(np.mean(loss_l))
                if early_stopping.early_stop:
                    break

    def to_latent(self, vae_model_path, tf2r_model_path):
        """ Function for saving model embeddings.
            
            Params
            ------
            vae_model_path: VAE saved model path
            tf2r_model_path: TF2rNet saved model path
        """
        if self.opt.device=='cuda':
            Tensor = torch.cuda.FloatTensor
        elif self.opt.device=='cpu':
            Tensor = torch.FloatTensor

        ### Initialize dataloaders
        self.opt.test_size = 0
        self.opt.train = False
        dataloader, _, TFs_idx, r2g_dist_coo, train_seq_dataloader, _ = self.init_data()

        # Initialize TF2rNet model
        model_dict = torch.load(tf2r_model_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet = TF2rNet(self.opt.TF2rNet_bottleneck_size, len(TFs_idx)).float().to(self.opt.device)
        tf2rNet.load_state_dict(model_dict) # Load weights

        # Initialize VAE
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)
        vae.load_state_dict(torch.load(vae_model_path,  map_location=torch.device(self.opt.device))['model_state_dict'])

        with torch.no_grad(): 
            vae.eval()
            tf2rNet.eval()
    
            # TF2rNet forward pass
            tf_pred_l = []
            for j, (seq_data_batch, seq_data_batch_idx) in enumerate(train_seq_dataloader, 0):
                X = Variable(seq_data_batch[0].type(Tensor))
                tf_pred, _ = tf2rNet(X)
                tf_pred_l.append(tf_pred)# * (torch.sigmoid(tf_pred_mask)>0.5).float()*1)
            adj_E1 = torch.cat(tf_pred_l).T
            print("VAE forward...")

            # Z_rna & Z_atac
            z_rna_l = []
            z_tf_mu_l = []
            z_tf_var_l = []
            z_rna_atac_l = []
            dec_rna_l = []
            for i, data_batch in tqdm(enumerate(dataloader['dataloader'], 0), unit="batch", total=len(dataloader['dataloader'])):
                # VAE forward pass
                inputs_rna, inputs_atac, data_id, dropout_mask_rna = data_batch
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                _, _, _, _, dec_rna, hidden_rna, hidden_tf_reg, hidden_tf_var, hidden_rna_atac, _  = vae(
                        inputs_rna, inputs_atac, dropout_mask_rna=None,
                        dropout_mask_atac=None, opt=self.opt, adj_E1=adj_E1)

                hidden_rna_atac = hidden_rna_atac * inputs_atac 
                z_rna_l += [hidden_rna.cpu().numpy()]
                z_tf_mu_l += [hidden_tf_reg.cpu().numpy()]
                z_tf_var_l += [hidden_tf_var.cpu().numpy()]
                z_rna_atac_l += [hidden_rna_atac.cpu().numpy()]
                dec_rna_l += [dec_rna.cpu().numpy()]


            z_rna_l = np.vstack(z_rna_l)
            z_rna_atac_l = np.vstack(z_rna_atac_l)
            z_tf_mu_l = np.vstack(z_tf_mu_l)
            z_tf_var_l = np.vstack(z_tf_var_l)
            dec_rna_l = np.vstack(dec_rna_l)
            np.save(self.opt.save_name + 'z_rna.npy', z_rna_l)
            np.save(self.opt.save_name + 'z_tf_reg.npy', z_tf_mu_l)
            np.save(self.opt.save_name + 'z_tf_var.npy', z_tf_var_l)
            np.save(self.opt.save_name + 'z_rna_atac.npy', z_rna_atac_l)
            np.save(self.opt.save_name + 'y_rna.npy', dec_rna_l)

    def get_beta_motif(self, epoch, mode='exp'):
        """ Function for decaying beta_motif parameter."""
        min_beta = 1e-8
        schedules = {
            'exp': np.maximum(min_beta, self.opt.beta_motif * ((min_beta / self.opt.beta_motif) ** (epoch / self.opt.n_epochs))),
            'lin': np.maximum(min_beta, self.opt.beta_motif - (self.opt.beta_motif - min_beta) * (epoch / self.opt.n_epochs)),
            'cos': min_beta + 0.5 * (self.opt.beta_motif - min_beta) * (1. + np.cos(epoch * math.pi / self.opt.n_epochs))
        }
        return schedules[mode]

    def train_model(self):
        opt = self.opt

        if opt.device=='cuda':
            Tensor = torch.cuda.FloatTensor
        elif opt.device=='cpu':
            Tensor = torch.FloatTensor

        # Save hyperparams to file
        with open(opt.save_name + "/hyperparma.txt", 'w') as f:
            json.dump(opt.__dict__, f, indent=2)

        # Initialize Tensorboard logger
        writer = SummaryWriter(opt.logs + '/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # Initialize dataloaders
        print("Initializing data...")
        train_dataloader, test_dataloader, TFs_idx, r2g_dist_coo, train_seq_dataloader, train_seq_dataloader_shuffle = self.init_data()

        eps = 1e-8

        # Initialize tf2r prior
        if opt.tf2r_df is not None:
            with torch.no_grad():
                adj_E1_prior = torch.tensor(pd.read_pickle(self.opt.tf2r_df).values, requires_grad=False)
                ranks = adj_E1_prior[adj_E1_prior>0]
                adj_E1_prior = (adj_E1_prior - ranks.min()) / (ranks.max() - ranks.min())
                adj_E1_prior[adj_E1_prior<0] = 1
                adj_E1_prior = 1 - adj_E1_prior
                adj_E1_prior = torch.log(adj_E1_prior / torch.unsqueeze(adj_E1_prior.sum(1) + eps, -1) + eps)
                adj_E1_prior = adj_E1_prior.to(opt.device)

        # Initialize TF2rNet model
        tf2rNet = TF2rNet(opt.TF2rNet_bottleneck_size, len(TFs_idx), seq_len=opt.seq_len).float().to(opt.device) # Initialize TF2r model
        if opt.load_model is not None:
            model_dict = torch.load(opt.load_model + 'model_tf2r.pth',  map_location=torch.device(opt.device))['model_state_dict'] # Load pretrained sequence model
            tf2rNet.load_state_dict(model_dict) # Load weights
            print("loaded weights for tf2r")

        # Initialize VAE model
        vae = VAE(TFs_idx, r2g_dist_coo, 1, opt.n_hidden, dev=opt.device).float().to(opt.device)
        if opt.load_model is not None:
            vae.load_state_dict(torch.load(opt.load_model + 'model.pth',  map_location=torch.device(opt.device))['model_state_dict'])
            print("loaded weights for vae")

        # Initialize optimizer
        optimizer = optim.Adam([{'params': vae.parameters(), 'lr':opt.lr}, {'params':tf2rNet.parameters(), 'lr':opt.lr}])
        if opt.load_model is not None:
            optimizer.load_state_dict(torch.load(opt.load_model + 'model.pth',  map_location=torch.device(opt.device))['optimizer_state_dict'])
            print("loaded optim state")

        adj_E1 = None
        best_loss = 0
        train_seq_dataloader_shuffle_iterator = iter(train_seq_dataloader_shuffle)
        
        for epoch in range(opt.n_epochs):
            vae.train()
            tf2rNet.train()
            
            for i, data_batch in tqdm(enumerate(train_dataloader['dataloader'], 0), unit="batch", total=len(train_dataloader['dataloader'])):
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                optimizer.zero_grad(True)

                if (epoch >= opt.warmup_vae) | (adj_E1 is None):
                    # TF2rNet forward pass
                    tf2rNet.eval()
                    with torch.no_grad():
                        tf_pred_l = []
                        for j, (seq_data_batch, seq_data_batch_idx) in enumerate(train_seq_dataloader, 0):
                            X = Variable(seq_data_batch[0].type(Tensor).to(opt.device))
                            tf_pred, _ = tf2rNet(X)
                            tf_pred_l.append(tf_pred)
                        adj_E1 = torch.cat(tf_pred_l).T

                    if (epoch >= opt.warmup_vae):
                        # train on n random regions for backpropagating gradients
                        tf2rNet.train()
                        tf_pred_l = []
                        try:
                            seq_data_batch, seq_data_batch_idx = next(train_seq_dataloader_shuffle_iterator)
                        except StopIteration:
                            train_seq_dataloader_shuffle_iterator = iter(train_seq_dataloader_shuffle)
                            seq_data_batch, seq_data_batch_idx = next(train_seq_dataloader_shuffle_iterator)
                        X = Variable(seq_data_batch[0].type(Tensor).to(opt.device))
                        tf_pred, _ = tf2rNet(X)
                        tf_pred_l.append(tf_pred)
                        adj_E1 = adj_E1.clone()
                        adj_E1[:, seq_data_batch_idx] = torch.cat(tf_pred_l).T

                # VAE forward pass
                inputs_rna, inputs_atac, data_id, dropout_mask_rna = data_batch
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                if opt.dropout_loss:
                    loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, dec_rna,  _, _, _, _, f1_atac = vae(
                        inputs_rna, inputs_atac, dropout_mask_rna=dropout_mask_rna.to(opt.device),
                        dropout_mask_atac=None, opt=opt, adj_E1=adj_E1)
                else:
                    loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, dec_rna,  _, _, _, _, f1_atac = vae(
                        inputs_rna, inputs_atac, dropout_mask_rna=None,
                        dropout_mask_atac=None, opt=opt, adj_E1=adj_E1)

                # Compute sparse loss
                E1_sparse = adj_E1.abs().mean()
                E2_sparse = F.relu(vae.adj_E2).mean()
                sparse_loss = opt.alpha * (E1_sparse + E2_sparse)
                loss = loss  + sparse_loss

                # Compute prior loss
                if opt.tf2r_df is not None:
                    E1_kl = F.kl_div(torch.log(adj_E1.abs() / torch.unsqueeze(adj_E1.abs().sum(1) + eps, -1) + eps), adj_E1_prior, reduction='batchmean', log_target=True) * self.opt.beta_motif # self.get_beta_motif(epoch)
                    loss = loss + E1_kl

                loss.backward()
                optimizer.step()
              
                # Tensorboard logs
                n_iter = (epoch*len(train_dataloader['dataloader'])) + i 
                writer.add_scalar('Loss/total', loss.detach().item(), n_iter)
                writer.add_scalar('Loss/rec_rna', loss_rec_rna.item(), n_iter)
                writer.add_scalar('Loss/rec_atac', loss_rec_atac.item(), n_iter)
                writer.add_scalar('Loss/kl_rna', loss_gauss_rna.item(), n_iter)
                writer.add_scalar('Loss/l1_A', sparse_loss.detach().item(), n_iter)
                writer.add_scalar('Loss/l1_E1', E1_sparse.detach().item(), n_iter)
                writer.add_scalar('Loss/l1_E2', E2_sparse.detach().item(), n_iter)
                writer.add_scalar('Loss/f1_atac', f1_atac.detach().item(), n_iter)
                if opt.tf2r_df is not None:
                    writer.add_scalar('Loss/kl_E1', E1_kl.detach().item(), n_iter)
            print('epoch:', epoch)#,

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, opt.save_name + '/model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': tf2rNet.state_dict(),
            }, opt.save_name + '/model_tf2r.pth')
           
            # Evaluate test set
            if opt.test_size > 0:
                with torch.no_grad():
                    vae.eval()
                    tf2rNet.eval()
                    loss_all, rec_rna, rec_atac, loss_kl_rna, loss_sparse = [], [], [], [], []
                    for i, data_batch in tqdm(enumerate(test_dataloader['dataloader'], 0), unit="batch", total=len(test_dataloader['dataloader'])):
#                        # TF2rNet forward pass
#                        if epoch > opt.warmup_vae:
#                            tf_pred_l = []
#                            for j, (seq_data_batch, seq_data_batch_idx) in enumerate(train_seq_dataloader, 0): 
#                                X = Variable(seq_data_batch[0].type(Tensor))
#                                tf_pred, _ = tf2rNet(X)
#                                tf_pred_l.append(tf_pred)
#                            adj_E1 = torch.cat(tf_pred_l).T

                        inputs_rna, inputs_atac, data_id, dropout_mask_rna  = data_batch
                        inputs_rna = Variable(inputs_rna.type(Tensor))
                        inputs_atac = Variable(inputs_atac.type(Tensor))

                        if opt.dropout_loss:
                            loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, dec_rna, _, _, _, _, _ = vae(
                                inputs_rna, inputs_atac, dropout_mask_rna=dropout_mask_rna.to(opt.device),
                                dropout_mask_atac=None, opt=opt, adj_E1=adj_E1)
                        else:
                            loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, dec_rna, _, _, _, _, _ = vae(
                                inputs_rna, inputs_atac, dropout_mask_rna=None,
                                dropout_mask_atac=None, opt=opt, adj_E1=adj_E1)
    
                        sparse_loss = opt.alpha * (adj_E1.abs().mean() + F.relu(vae.adj_E2).mean())
                        loss = loss + sparse_loss

                        if opt.tf2r_df is not None:
                            loss = loss + F.kl_div(torch.log(adj_E1.abs() / torch.unsqueeze(adj_E1.abs().sum(1) + eps, -1) + eps), adj_E1_prior, reduction='batchmean', log_target=True) * self.opt.beta_motif#* self.get_beta_motif(epoch)
                        if best_loss < loss.detach().item():
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': vae.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }, opt.save_name + '/best_model.pth')
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': tf2rNet.state_dict(),
                            }, opt.save_name + '/best_model_tf2r.pth')
                            best_loss = loss.detach().item()
                             
                        rec_rna.append(loss_rec_rna.item())
                        rec_atac.append(loss_rec_atac.item())
                        loss_all.append(loss.detach().item())
                        loss_kl_rna.append(loss_gauss_rna.item())
                        loss_sparse.append(sparse_loss.detach().item())
   
                    del loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, dec_rna, sparse_loss

                    # Tensorboard logs
                    writer.add_scalar('Test/loss_total', np.mean(loss_all), epoch)
                    writer.add_scalar('Test/rec_rna', np.mean(rec_rna), epoch)
                    writer.add_scalar('Test/rec_atac', np.mean(rec_atac), epoch)
                    writer.add_scalar('Test/kl_rna', np.mean(loss_kl_rna), epoch)
                    writer.add_scalar('Test/l1_A', np.mean(loss_sparse), epoch)

                
                    del loss_all, rec_rna, rec_atac, loss_kl_rna, loss_sparse
