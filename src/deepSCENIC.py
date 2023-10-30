import datetime
import json
import os
import pickle
import re
from typing import Tuple
import h5py
import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score
from tqdm import tqdm
from enformer_pytorch import Enformer

from src.tf2rNet.models import MotifNet, final, Sei
from src.tf2rNet.utils import *
from src.model import VAE
from src.utils import EarlyStopping


## Build dataloaders ##
def build_seq_dataloader(opt, ad=None, pretrain=False):
    """ Build dataloader for TF2rNet
            Params
            ------
            opt: model hyperparams
            ad: scATAC-seq data
            pretrain: if True, build dataloader for TF2rNet pretraining
    """
    class TensorDatasetWithIndex(Dataset[Tuple[torch.Tensor, ...]]):
        """ Dataset wrapping tensors.
        """
        tensors: Tuple[torch.Tensor, ...]

        def __init__(self, *tensors: torch.Tensor) -> None:
            assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
            self.tensors = tensors

        def __getitem__(self, index):
            return (tuple(tensor[index] for tensor in self.tensors), index)

        def __len__(self):
            return self.tensors[0].size(0)

    peak = [re.split("[\-:]+", elem) for elem in ad.var_names]
    peak = pd.DataFrame(peak, columns=['chr', 'start', 'end'])

    make_h5_sparse(peak, '%s/all_seqs.h5'%opt.save_name, opt.fasta, opt.seq_len, opt.TF2rNet_batch_size)
    with h5py.File('%s/all_seqs.h5'%opt.save_name, 'r') as hf:
        X_seq = hf['X'][:].astype('float32')

    if opt.enformer_embs_file:    
        hf = h5py.File('%s'%opt.enformer_embs_file, 'r')
        X_emb = hf['X'][:]
        hf.close()
        data = TensorDatasetWithIndex(torch.FloatTensor(X_seq), torch.FloatTensor(X_emb))
    else:
        if pretrain:
            data = TensorDatasetWithIndex(torch.FloatTensor(X_seq), torch.FloatTensor(ad.X.T))
        else:
            data = TensorDatasetWithIndex(torch.FloatTensor(X_seq))

    train_dataloader =  DataLoader(data, batch_size=opt.TF2rNet_batch_size, shuffle=False, num_workers=1)
    test_dataloader = None

    if pretrain:
        # Define the sizes of the train and test splits
        train_size = int((1-opt.test_size) * len(data))
        test_size = len(data) - train_size

        # Split the dataset into train and test sets
        train_data, test_data = random_split(data, [train_size, test_size])
        train_dataloader_shuffle =  DataLoader(train_data, batch_size=opt.TF2rNet_batch_size, shuffle=True, num_workers=1)
        test_dataloader_shuffle = DataLoader(test_data, batch_size=opt.TF2rNet_batch_size, shuffle=True, num_workers=1)
    else:
        train_dataloader_shuffle =  DataLoader(data, batch_size=opt.TF2rNet_batch_size, shuffle=True, num_workers=1)
        test_dataloader_shuffle = None

    return train_dataloader, test_dataloader, train_dataloader_shuffle, test_dataloader_shuffle
 
def build_dataloader(data_rna, data_atac, batch_size, opt):
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
    if type(data_rna.X)!=np.ndarray:
        data_rna.X = data_rna.X.toarray()
    if type(data_atac.X)!=np.ndarray:
        data_atac.X = data_atac.X.toarray()

    # Build gene exp. dropout mask
    Dropout_Mask_rna = (data_rna.X != 0).astype(np.float) # binary mask of exp matrix

    # Get scaled values
    # data_rna = pd.DataFrame(data_rna.X, index=list(data_rna.obs_names), columns=rna_gene_name)
    # data_atac = pd.DataFrame(data_atac.X, index=list(data_atac.obs_names), columns=atac_region_name)

    num_genes_rna = data_rna.shape[1]
    num_regions_atac = data_atac.shape[1]

    feat_rna = torch.FloatTensor(data_rna.X)
    feat_atac = torch.FloatTensor(data_atac.X) 
    data = TensorDataset(feat_rna, feat_atac, torch.LongTensor(list(range(len(feat_rna)))),
                         torch.FloatTensor(Dropout_Mask_rna))

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=opt.train, num_workers=1)

    return {'dataloader': dataloader, 'num_genes_rna': num_genes_rna, 'num_regions_atac': num_regions_atac, 'data_rna': data_rna, 'data_atac': data_atac, 'rna_gene_name': rna_gene_name, 'atac_region_name':atac_region_name}


## deepSCENIC ##
class deepSCENIC:
    """ Class for deepSCENIC model.
        Params
        ------
        opt: model hyperparams
    """
    def __init__(self, opt):
        self.opt = opt
        # Create save dir
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')

    def init_data(self, test=False):
        # Read TFs list
        TFs_df = pd.read_csv(self.opt.TF_file, header=None, names=["name"], usecols=[0])
        TFs = set(TFs_df['name'])

        # Read region to gene mask
        r2g_dist_coo = load_npz(self.opt.r2g_mask)

        if test==False:
            # Read data
            print("reading data...")
            data_rna = sc.read(self.opt.data_rna_file)
            data_atac = sc.read(self.opt.data_atac_file)
            print("data read!")
            print(data_rna)
            print(data_atac)
        else:
            # Read data
            print("reading data...")
            data_rna = sc.read(self.opt.data_rna_file_test)
            data_atac = sc.read(self.opt.data_atac_file_test)
            print("data read!")
            print(data_rna)
            print(data_atac)

        # Get TFs index in rna data
        TFs_idx = np.where(data_rna.var_names.isin(TFs))[0]
        TFs = data_rna.var_names[data_rna.var_names.isin(TFs)].to_list()
        self.TFs = TFs

        # Check if sparse data
        if type(data_rna.X)!=np.ndarray:
            data_rna.X = data_rna.X.toarray()
        else:
            data_rna.X = data_rna.X
        if type(data_atac.X)!=np.ndarray:
            data_atac.X = data_atac.X.toarray()
        else:
            data_atac.X = data_atac.X

        self.n_cells = data_rna.shape[0]
        self.n_genes = data_rna.shape[1]
        self.n_regions = data_atac.shape[1]

        # Binarize ATAC data
        if self.opt.bin_acc==True:
            data_atac.X[data_atac.X > 0] = 1

        # positive scaling of rna data    
        data_rna.X = data_rna.X / data_rna.X.std(0)
        if self.opt.bin_acc==False:
            # positive scaling of atac data
            sc.pp.normalize_total(data_atac)
            sc.pp.log1p(data_atac)
            data_atac.X = data_atac.X / data_atac.X.std(0)
        # data_rna.X = data_rna_values
        # data_atac.X = data_atac_values

        # Split data in train and test
        if self.opt.test_size > 0:
            cell_idxs = data_rna.obs.index.values
            cell_idxs_train, cell_idxs_test = train_test_split(cell_idxs, test_size=self.opt.test_size, random_state=42)
            test_rna = data_rna[cell_idxs_test,:].copy()
            data_rna = data_rna[cell_idxs_train,:].copy()
            test_atac = data_atac[cell_idxs_test,:].copy()
            data_atac = data_atac[cell_idxs_train,:].copy()

        # Build RNA/ATAC dataloader
        train_dataloader = build_dataloader(data_rna, data_atac, self.opt.batch_size, self.opt)        
        if self.opt.test_size > 0:
            test_dataloader = build_dataloader(test_rna, test_atac, 16, self.opt)
        else:
            test_dataloader = None

        # Build sequence dataloader
        train_seq_dataloader, _, train_seq_data_shuffle, _ = build_seq_dataloader(self.opt, ad=data_atac)

        return train_dataloader, test_dataloader, TFs_idx, r2g_dist_coo, train_seq_dataloader, train_seq_data_shuffle

    def pretrain_TF2r(self, ad, n_epochs=200, patience=10):
        """ Pretrain TF2r data on the df dataframe. Can be any kind of region data (with regions as rows and features as columns)
            (scATAC-seq, Chip-seq, pycisTarget output,...).
   
            Params
            ------
            df: dataframe for training.
            n_epochs: Number of epochs
            patience: patience parameter for early stopping
        """
        
        opt = self.opt
        if opt.device=='cuda':
            Tensor = torch.cuda.FloatTensor
        elif opt.device=='cpu':
            Tensor = torch.FloatTensor
        if patience is not None:
            early_stopping = EarlyStopping(patience=patience)
        # Initialize Tensorboard writer
        writer = SummaryWriter(opt.logs + '/logs/TF2rNet/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Read dataframe
        if opt.TF2rNet_loss == 'bce':
            # Binarized values        
            ad.X[ad.X>0]=1
        # Build sequence dataloader
        _, _, train_seq_data_shuffle, test_seq_data_shuffle = build_seq_dataloader(self.opt, ad=ad, pretrain=True)
        out_dim = ad.shape[0]
        # Initialize TF2rNet model    
        model = Sei().float().to(opt.device)
        final1  = final(opt.TF2rNet_bottleneck_size * opt.emb_len, n_units=out_dim, flatten=False).float().to(opt.device)
        # Initialize optimizer
        optimizer = optim.Adam([{'params':model.parameters(), 'lr':opt.lr}, {'params':final1.parameters(), 'lr':opt.lr}])

        # Define loss function
        def tf2r_loss(pred, real):
            """ TF2rNet loss function.
            """
            if opt.TF2rNet_loss == 'mse':
                return torch.mean((real - pred).pow(2))
            elif opt.TF2rNet_loss == 'bce':
                return torch.mean(F.binary_cross_entropy_with_logits(pred, real))
            else:
                raise NameError('Loss function must be either mse or bce!')

        # Initialize F1 score computation
        f1 = F1Score(task='binary', num_classes=1).to(opt.device)
        # Training
        for epoch in range(n_epochs):
            model.train()
            loss_l = []
            for i, (data_batch, _) in tqdm(enumerate(train_seq_data_shuffle, 0), unit="batch", total=len(train_seq_data_shuffle)):
                optimizer.zero_grad()
                X, y = data_batch
                X = Variable(X.type(Tensor).to(opt.device))
                X = model(X)
                pred, _ = final1(X.reshape(-1, opt.TF2rNet_bottleneck_size * opt.emb_len))
                loss = tf2r_loss(pred, y.to(opt.device))
                loss.backward()
                optimizer.step()
                loss_l.append(loss.detach().item())
                # Tensorboard logs
                n_iter = (epoch*len(train_seq_data_shuffle)) + i
                writer.add_scalar('Loss/train', loss.item(), n_iter)
                if opt.TF2rNet_loss == 'bce':
                    f1_pred = f1(pred.ravel(), y.to(opt.device).int().ravel())
                    writer.add_scalar('F1/train', f1_pred, n_iter)
                del loss
            print('epoch:', epoch, 'loss:', np.mean(loss_l))

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, opt.save_name + '/TF2rNET_func_enc_pretrain.pth')

            # Evaluate validation set
            with torch.no_grad():
                model.eval()
                loss_l = []
                for i, (data_batch, _) in tqdm(enumerate(test_seq_data_shuffle, 0), unit="batch", total=len(test_seq_data_shuffle)):
                    X,y = data_batch
                    X = Variable(X.type(Tensor).to(opt.device))
                    X = model(X)   
                    pred, _ = final1(X.reshape(-1, opt.TF2rNet_bottleneck_size * opt.emb_len))
                    loss = tf2r_loss(pred, y.to(opt.device))
                    loss_l.append(loss.detach().item())
                    del loss

                print('epoch:', epoch, 'test_loss:', np.mean(loss_l))
                writer.add_scalar('Loss/val', np.mean(loss_l), epoch)
                # Evaluate early stopping
                if patience is not None:
                    early_stopping(np.mean(loss_l))
                    if early_stopping.early_stop:
                        break

    def pretrain(self):        
        # self.opt.train=False
        opt = self.opt
        if opt.device=='cuda':
            Tensor = torch.cuda.FloatTensor
        elif opt.device=='cpu':
            Tensor = torch.FloatTensor

        # Initialize Tensorboard writer
        writer = SummaryWriter(opt.logs + '/logs/TF2rNet/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Read dataframe
        # Initialize dataloaders
        print("Initializing data...")
        self.opt.test_size=0
        test_dataloader, _, TFs_idx, r2g_dist_coo, test_seq_dataloader, test_seq_dataloader_shuffle = self.init_data(test=True)
        train_dataloader, _, TFs_idx, r2g_dist_coo, train_seq_dataloader, train_seq_dataloader_shuffle = self.init_data()
        

        # Initialize TF2rNet model   
        with open(self.opt.ppms_file, 'rb') as f:
            PPMs = pickle.load(f) 
        tf2rNet_func_encoder = Sei().float().to(self.opt.device)
        tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device)
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)

        if self.opt.load_model is not None:
            # Load tf2r fuctional encoder
            model_dict_tf2r_func_enc = torch.load(self.opt.load_model + 'model_tf2r_encoder.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
            tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
            model_dict_tf2r = torch.load(self.opt.load_model + 'model_tf2r.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
            tf2rNet.load_state_dict(model_dict_tf2r)
            vae.load_state_dict(torch.load(self.opt.load_model + 'model.pth',  map_location=torch.device(self.opt.device))['model_state_dict'])
            print("loaded models")   


        # Initilialize explanation model 
        # Define the concatenated model with a custom forward method
        class ConcatModel(torch.nn.Module):
            def __init__(self, models):
                super(ConcatModel, self).__init__()
                self.models = torch.nn.ModuleList(models)

            def forward(self, seq, motif=False, explain=True):
                if motif==False:
                    # seq = torch.unsqueeze(seq, dim=0)
                    for model in self.models:
                        if isinstance(model, Enformer):
                            x = model(seq, return_only_embeddings=True)
                        elif isinstance(model, MotifNet):                 
                            x = model(emb=x, motif=motif)
                            # x = torch.unsqueeze(x, dim=0)
                        else:
                            x = model(seq, explain=explain)
                    return x
                else:
                    # seq = torch.unsqueeze(seq, dim=0)
                    for model in self.models:
                        if isinstance(model, MotifNet):
                            x = model(seq=seq, motif=motif, explain=explain)
                    return x

        # Initialize optimizer
        optimizer =  optim.Adam([{'params':vae.parameters(), 'lr':1e-4}, {'params':tf2rNet_func_encoder.parameters(), 'lr':self.opt.lr }, {'params':tf2rNet.parameters(), 'lr':self.opt.lr}])

        adj_E1_mtf = None
        best_loss = np.inf
        train_dataloader_iter = iter(train_dataloader['dataloader']) # Initialize iterator for vae dataloader
        for epoch in range(opt.n_epochs):
            vae.train()
            tf2rNet.train()
            tf2rNet_func_encoder.train()
 
            # if (adj_E1_mtf is None):
            #     # Infer motif matching scores
            #     with torch.no_grad():
            #         mtf_pred_l = []
            #         for _, (X, seq_data_batch_idx) in tqdm(enumerate(train_seq_dataloader, 0)):
            #             mtf_pred  = tf2rNet(seq=X[0].to(self.opt.device), motif=True)
            #             mtf_pred_l.append(mtf_pred)
            #         adj_E1_mtf = torch.cat(mtf_pred_l).T                        
            #         del mtf_pred_l, mtf_pred

            for j, (X, seq_data_batch_idx) in tqdm(enumerate(train_seq_dataloader_shuffle, 0)):
                seq = X[0].to(torch.int64).to(self.opt.device)
                seq = F.one_hot(seq, num_classes=4).to(torch.float)
                seq.requires_grad = True
                tf_pred = tf2rNet_func_encoder(seq, explain=True)
                tf_pred, seq_l1 = tf2rNet(emb=tf_pred, motif=False)
                adj_E1 = tf_pred.T
                # adj_E1 = adj_E1 * adj_E1_mtf[:, seq_data_batch_idx]
  
                # vae forward pass
                try:
                    data_batch = next(train_dataloader_iter)
                except StopIteration:
                    train_dataloader_iter = iter(train_dataloader['dataloader'])
                    data_batch = next(train_dataloader_iter)
                inputs_rna, inputs_atac, _, _ = data_batch
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                loss, loss_rec_atac, loss_gauss_rna, f1_atac = vae.pretrain(
                        inputs_rna, inputs_atac, opt=self.opt, adj_E1=adj_E1, idxs=seq_data_batch_idx)
                
                loss = seq_l1 + loss
                optimizer.zero_grad(True)
                loss.backward()
                # torch.nn.utils.clip_grad_value_(tf2rNet_func_encoder.parameters(), clip_value=1.0)
                # torch.nn.utils.clip_grad_value_(tf2rNet.parameters(), clip_value=1.0)
                if not j%self.opt.n_it_acc: 
                    optimizer.step()
                
                
                # Tensorboard logs
                n_iter = (epoch*len(train_seq_dataloader_shuffle)) + j
                writer.add_scalar('Loss/total', loss.detach().item(), n_iter)
                writer.add_scalar('Loss/rec_atac', loss_rec_atac.item(), n_iter)
                writer.add_scalar('Loss/seq_l1', seq_l1.item(), n_iter)
                writer.add_scalar('Loss/kl_rna', loss_gauss_rna.item(), n_iter)
                writer.add_scalar('Loss/f1_atac', f1_atac.detach().item(), n_iter)
            print('epoch:', epoch)

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, self.opt.save_name + '/model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': tf2rNet.state_dict(),
            }, self.opt.save_name + '/model_tf2r.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': tf2rNet_func_encoder.state_dict(),
            }, self.opt.save_name + '/model_tf2r_encoder.pth')

           
            # Evaluate test set
            if (self.opt.test_size > 0):
                with torch.no_grad():
                    vae.eval()
                    loss_all, rec_atac, loss_kl_rna = [], [], []
    
                    # TF2rNet forward pass
                    tf2rNet.eval()
                    tf2rNet_func_encoder.eval()
                    with torch.no_grad():
                        tf_pred_l = []
                        for j, (X, seq_data_batch_idx) in tqdm(enumerate(test_seq_dataloader, 0)):
                            tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device))
                            tf_pred, _ = tf2rNet(emb=tf_pred, motif=False)
                            tf_pred_l.append(tf_pred)
                        adj_E1 = torch.cat(tf_pred_l).T
                        # adj_E1 = adj_E1 * adj_E1_mtf
                        del tf_pred_l, tf_pred

                        for i, data_batch in tqdm(enumerate(test_dataloader['dataloader'], 0), unit="batch", total=len(test_dataloader['dataloader'])):
                            inputs_rna, inputs_atac, _, dropout_mask_rna  = data_batch
                            inputs_rna = Variable(inputs_rna.type(Tensor))
                            inputs_atac = Variable(inputs_atac.type(Tensor))

                            _, _, loss_rec_atac, loss_gauss_rna,_, _, _, _, _, _, f1_atac = vae(
                                inputs_rna, inputs_atac, dropout_mask_rna=dropout_mask_rna.to(self.opt.device), opt=self.opt, adj_E1=adj_E1)

                        loss = loss_rec_atac + loss_gauss_rna
                        if loss.detach().item() < best_loss:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': vae.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }, self.opt.save_name + '/best_model.pth')
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': tf2rNet.state_dict(),
                            }, self.opt.save_name + '/best_model_tf2r.pth')
                            best_loss = loss.detach().item()
                             
                        rec_atac.append(loss_rec_atac.item())
                        loss_all.append(loss.detach().item())
                        loss_kl_rna.append(loss_gauss_rna.item())
   
                    del loss, loss_rec_atac, loss_gauss_rna

                    # Tensorboard logs
                    writer.add_scalar('Test/loss_total', np.mean(loss_all), epoch)
                    writer.add_scalar('Test/rec_atac', np.mean(rec_atac), epoch)
                    writer.add_scalar('Test/kl_rna', np.mean(loss_kl_rna), epoch)

                    del loss_all, rec_atac, loss_kl_rna


       
    def simulate_perturbation(self, vae_model_path, tf2r_model_path, perturbation, n_iter=5, ad=None, n_neighs=15, tf2r_func_enc_model_path=None, keep_intermediate=False, adj_E1=None, adj_E1_pert=None, fc_upper_bound=99.9, fc_lower_bound=0):
        """ Function for simulating TF perturbations.
            
            Params
            ------
            vae_model_path: VAE saved model path
            tf2r_model_path: TF2rNet saved model path
            perturbation: A dictionary indexed by TF names with perturbation level as values. 
                e.g. {'SOX10': 0} will simulate a perturbation where the expression level of SOX10 is set to 0 in all cells.
            n_iter: Number of itertions to simulate. Default is 5.
            tf2r_func_enc_model_path: TF2rNet functional encoder saved model path. If None, TF2rNet will be used without functional encoder.
            keep_intermediate: If set to True simulated gene expression values for each iteration will be kept.
            adj_E1: Precomputed TF-region matrix. If None, matrix will be internally generated.
            adj_E1_pert: Perturbed TF-region matrix.
            fc_upper_bound: Upper bound for fold change values. Default is 0.99 percentile.
            fc_lower_bound: Lower bound for fold change values. Default is 0.01 percentile.
        """

        def generate_metacells(pred_mtx, conn_mtx):
            meta_mtx = pred_mtx.copy()
            meta_mtx.loc[:,:] = 0
            for cell_idx in range(pred_mtx.shape[0]):
                neigh_idxs = np.nonzero(conn_mtx[cell_idx,:].A)[1]
                meta_mtx.iloc[cell_idx,:] = pred_mtx.iloc[neigh_idxs,:].mean(0)
            return meta_mtx

        def _do_one_round_of_simulation(vae, perturbed_mtx, TFs_idx, adj_E1, opt):
            data = TensorDataset(torch.FloatTensor(perturbed_mtx.values))
            dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=False, num_workers=1)

            y_pert_l = []
            with torch.no_grad():
                vae.eval();
                for batch in dataloader:
                    perturbed_batch = batch[0]
                    y_pert, _, _, _, _ = vae.predict(perturbed_batch[:, TFs_idx].to(opt.device), adj_E1)
                    y_pert_l += [y_pert['x_rec'].cpu().numpy()]

            y_pert_l = np.vstack(y_pert_l)
            return pd.DataFrame(y_pert_l, index=perturbed_mtx.index, columns=perturbed_mtx.columns)

        ### Initialize dataloaders
        self.opt.test_size = 0
        self.opt.train = False
        _, _, TFs_idx, r2g_dist_coo, seq_dataloader, _ = self.init_data()

        # Initialize TF2rNet model
        with open(self.opt.ppms_file, 'rb') as f:
            PPMs = pickle.load(f)
        model_dict_tf2rNet = torch.load(tf2r_model_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        if not self.opt.enformer_embs_file:
            # Load pretrained TF2rNet functional encoder model
            model_dict_tf2r_func_enc = torch.load(tf2r_func_enc_model_path,  map_location=torch.device(self.opt.device))['model_state_dict']
            tf2rNet_func_encoder = Sei().float().to(self.opt.device)
            tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device)
            tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
            tf2rNet.load_state_dict(model_dict_tf2rNet)
            tf2rNet_func_encoder.eval()
        else:
            # Load pretrained MotifNet model
            tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device)
            tf2rNet.load_state_dict(model_dict_tf2rNet)

        # Load pretrained VAE model
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)
        vae.load_state_dict(torch.load(vae_model_path,  map_location=torch.device(self.opt.device))['model_state_dict'])

        with torch.no_grad():
            vae.eval()
            tf2rNet.eval()
            if adj_E1 is None:
                # TF2rNet forward pass
                tf_pred_mtf_l = []
                tf_pred_ctx_l = []
                for j, (X, _) in tqdm(enumerate(seq_dataloader, 0), total=len(seq_dataloader)):
                    tf_pred_mtf = tf2rNet(seq=X[0].to(self.opt.device), motif=True)
                    if not self.opt.enformer_embs_file:
                        tf_pred_ctx = tf2rNet_func_encoder(X[0].to(self.opt.device))
                        tf_pred_ctx = tf2rNet(emb=tf_pred_ctx, motif=False)
                    else:
                        tf_pred_ctx = tf2rNet(emb=X[1].to(self.opt.device), motif=False)
                    tf_pred_ctx_l.append(tf_pred_ctx)
                    tf_pred_mtf_l.append(tf_pred_mtf)
                adj_E1_mtf = torch.cat(tf_pred_mtf_l).T
                adj_E1_ctx = torch.cat(tf_pred_ctx_l).T
                adj_E1 = adj_E1_mtf * adj_E1_ctx
            else:
                adj_E1=adj_E1.to(self.opt.device)
            
            if keep_intermediate:
                perturbation_over_iter = {}
                fcs = {}
            #Reads original gene expression matrix
            if ad is None:
                ad = sc.read(self.opt.data_rna_file)
            # ad.X = ad.layers['log_norm']
            sc.pp.neighbors(ad, n_neighbors=n_neighs)
            conn_matrix = ad.obsp['connectivities']
            original_matrix = ad.to_df().copy()
            original_matrix.loc[:,:] = ad.X
            # Normalize data
            original_matrix = original_matrix / original_matrix.std(0)
            perturbed_matrix = original_matrix.copy()
            #do several iterations of perturbation
            perturbed_pred_matrix_t_1 = _do_one_round_of_simulation(vae, original_matrix, TFs_idx, adj_E1, self.opt)
            perturbed_pred_matrix_t_1[perturbed_pred_matrix_t_1<0] = 0
            perturbed_pred_matrix_t_1 = generate_metacells(perturbed_pred_matrix_t_1, conn_matrix)

            # knock down TFs
            if len(perturbation.keys())>0:
                for gene in perturbation.keys():
                    perturbed_matrix.loc[:, gene] = perturbation[gene]
            if adj_E1_pert is not None:
                adj_E1 = adj_E1_pert
            if keep_intermediate:
                 # Save original matrix
                 perturbation_over_iter['0'] = original_matrix.copy()
                 # Save predictions of unperturbed matrix
                 perturbation_over_iter['1'] = perturbed_pred_matrix_t_1.copy()
            for i in tqdm(range(n_iter)):
                if len(perturbation.keys())>0:
                    for gene in perturbation.keys():
                        perturbed_matrix.loc[:, gene] = perturbation[gene]

                # Save predictions of perturbed matrix
                perturbed_pred_matrix_t_2 = _do_one_round_of_simulation(vae, perturbed_matrix, TFs_idx, adj_E1, self.opt)
                perturbed_pred_matrix_t_2[perturbed_pred_matrix_t_2<0] = 0 # Remove negative values
                perturbed_pred_matrix_t_2 = generate_metacells(perturbed_pred_matrix_t_2, conn_matrix)
                
                fc = (perturbed_pred_matrix_t_2 + 1e-8)/(perturbed_pred_matrix_t_1 + 1e-8) # compute fold change
                perturbed_pred_matrix_t_1 = perturbed_pred_matrix_t_2.copy()

                perturbed_matrix = (perturbed_matrix * fc).copy() # Apply fold change compute new expression matrix

                if keep_intermediate:
                    perturbation_over_iter[str(i + 2)] = perturbed_matrix
                    fcs[str(i+2)] = fc
            if keep_intermediate:
                return  perturbation_over_iter, fcs
            else:
                return perturbed_matrix

    def to_latent(self, vae_model_path, tf2r_model_path, tf2r_func_enc_model_path=None):
        """ Function for saving model embeddings.
            
            Params
            ------
            vae_model_path: VAE saved model path
            tf2r_model_path: TF2rNet saved model path
            tf2r_func_enc_model_path: TF2rNet functional encoder saved model path. If None, TF2rNet will be used without functional encoder.
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
        with open(self.opt.ppms_file, 'rb') as f:
            PPMs = pickle.load(f)
        model_dict_tf2rNet = torch.load(tf2r_model_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        if not self.opt.enformer_embs_file:
            model_dict_tf2r_func_enc = torch.load(tf2r_func_enc_model_path,  map_location=torch.device(self.opt.device))['model_state_dict']
            tf2rNet_func_encoder = Sei().float().to(self.opt.device)
            tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device)
            tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
            tf2rNet.load_state_dict(model_dict_tf2rNet)
        else:
            tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device)
            tf2rNet.load_state_dict(model_dict_tf2rNet)

        # Initialize VAE
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)
        vae.load_state_dict(torch.load(vae_model_path,  map_location=torch.device(self.opt.device))['model_state_dict'])

        with torch.no_grad(): 
            vae.eval()
            tf2rNet.eval()

            # Infer tf2r scores
            mtf_pred_l = []
            for _, (X, _) in tqdm(enumerate(train_seq_dataloader, 0)):
                mtf_pred  = tf2rNet(seq=X[0].to(self.opt.device), motif=True)
                mtf_pred_l.append(mtf_pred)
            adj_E1_mtf = torch.cat(mtf_pred_l).T   
            tf_pred_l = []
            for _, (X, _) in tqdm(enumerate(train_seq_dataloader, 0)):
                if not self.opt.enformer_embs_file:
                    tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device))
                    tf_pred = tf2rNet(emb=tf_pred, motif=False)
                else:
                    tf_pred = tf2rNet(emb=X[1].to(self.opt.device), motif=False)
                tf_pred_l.append(tf_pred)
            adj_E1 = torch.cat(tf_pred_l).T
            adj_E1 = adj_E1 * adj_E1_mtf
    
            print("VAE forward...")
            z_rna_l = []
            z_tf_mu_l = []
            z_tf_var_l = []
            z_rna_atac_l = []
            dec_rna_l = []
            dec_atac_l = []
            for _, data_batch in tqdm(enumerate(dataloader['dataloader'], 0), unit="batch", total=len(dataloader['dataloader'])):
                # VAE forward pass
                inputs_rna, inputs_atac, _, _ = data_batch
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                _, _, _, _, dec_rna, dec_atac, hidden_rna, hidden_tf_reg, hidden_tf_var, hidden_rna_atac, _  = vae(
                        inputs_rna, inputs_atac, dropout_mask_rna=None,
                        dropout_mask_atac=None, opt=self.opt, adj_E1=adj_E1)

                z_rna_l += [hidden_rna.cpu().numpy()]
                z_tf_mu_l += [hidden_tf_reg.cpu().numpy()]
                z_tf_var_l += [hidden_tf_var.cpu().numpy()]
                z_rna_atac_l += [hidden_rna_atac.cpu().numpy()]
                dec_rna_l += [dec_rna.cpu().numpy()]
                dec_atac_l += [dec_atac.cpu().numpy()]

            z_rna_l = np.vstack(z_rna_l)
            z_rna_atac_l = np.vstack(z_rna_atac_l)
            z_tf_mu_l = np.vstack(z_tf_mu_l)
            z_tf_var_l = np.vstack(z_tf_var_l)
            dec_rna_l = np.vstack(dec_rna_l)
            dec_atac_l = np.vstack(dec_atac_l)
            np.save(self.opt.save_name + 'z_rna.npy', z_rna_l)
            np.save(self.opt.save_name + 'z_tf_reg.npy', z_tf_mu_l)
            np.save(self.opt.save_name + 'z_tf_var.npy', z_tf_var_l)
            np.save(self.opt.save_name + 'z_rna_atac.npy', z_rna_atac_l)
            np.save(self.opt.save_name + 'y_rna.npy', dec_rna_l)
            np.save(self.opt.save_name + 'y_atac.npy', dec_atac_l)

    def train_model(self):
        """ Function for training deepSCENIC model.
        """
        # Save hyperparams to file
        with open(self.opt.save_name + "/hyperparma.txt", 'w') as f:
            json.dump(self.opt.__dict__, f, indent=2)

        # Initialize Tensorboard logger
        writer = SummaryWriter(self.opt.logs + '/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Initialize dataloaders
        print("Initializing data...")
        train_dataloader, test_dataloader, TFs_idx, r2g_dist_coo, train_seq_dataloader, train_seq_dataloader_shuffle = self.init_data()

        if self.opt.device=='cuda':
            Tensor = torch.cuda.FloatTensor
        elif self.opt.device=='cpu':
            Tensor = torch.FloatTensor

        # Initialize TF2rNet model
        with open(self.opt.ppms_file, 'rb') as f:
            PPMs = pickle.load(f)
        if self.opt.load_tf2rNet_model is not None:
            tf2rNet_func_encoder = Sei().float().to(self.opt.device)
            model_dict_tf2r_func_enc = torch.load(self.opt.load_tf2rNet_model,  map_location=torch.device(self.opt.device))['model_state_dict']
            tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
            print("loaded weights for tf2r")
            tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device) 
        elif self.opt.load_model is not None:
            if not self.opt.enformer_embs_file:
                # Load tf2r fuctional encoder
                tf2rNet_func_encoder = Sei().float().to(self.opt.device)
                model_dict_tf2r_func_enc = torch.load(self.opt.load_model + 'model_tf2r_encoder.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
                tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
                # Load tf2r contex head
            tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device)
            model_dict_tf2r = torch.load(self.opt.load_model + 'model_tf2r.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
            tf2rNet.load_state_dict(model_dict_tf2r)   
        else:
            if not self.opt.enformer_embs_file:
                # Initialize tf2rNet cnn
                tf2rNet_func_encoder = Sei().float().to(self.opt.device)
                tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device)
            else:
                tf2rNet = MotifNet(PPMs, self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, explain=False, dev=self.opt.device).float().to(self.opt.device)

        # Initialize VAE model
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)
        if self.opt.load_model is not None:
            vae.load_state_dict(torch.load(self.opt.load_model + 'model.pth',  map_location=torch.device(self.opt.device))['model_state_dict'])
            print("loaded weights for vae")

        # Initialize optimizers
        if not self.opt.enformer_embs_file:
            optimizer = optim.Adam(vae.parameters(), self.opt.lr)
            optim_func_enc = optim.Adam([{'params':tf2rNet_func_encoder.parameters(), 'lr':self.opt.lr * 1e-3}, {'params':tf2rNet.parameters(), 'lr':self.opt.lr * 1e-3}])
        else:
            optimizer = optim.Adam([{'params': vae.parameters(), 'lr':self.opt.lr}, {'params':tf2rNet.parameters(), 'lr':self.opt.lr}])
#        if self.opt.load_model is not None:
#            optimizer.load_state_dict(torch.load(self.opt.load_model + 'model.pth',  map_location=torch.device(self.opt.device))['optimizer_state_dict'])
#            if not self.opt.enformer_embs_file:
#                optimizer.load_state_dict(torch.load(self.opt.load_model + 'model.pth',  map_location=torch.device(self.opt.device))['optimizer_tf2r_state_dict'])
#            print("loaded optim state")
#        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
#                              T_max = opt.n_epochs, # Maximum number of iterations.
#                              eta_min = 1e-5,
#                              verbose=True) # Minimum learning rate.

        adj_E1_mtf = None
        best_loss =  float('inf')
        train_seq_dataloader_shuffle_iterator = iter(train_seq_dataloader_shuffle) # Initialize iterator for sequence dataloader
        for epoch in range(self.opt.n_epochs):
            vae.train()
            tf2rNet.train()
            if not self.opt.enformer_embs_file:
                tf2rNet_func_encoder.train()
 
            for i, data_batch in tqdm(enumerate(train_dataloader['dataloader'], 0), unit="batch", total=len(train_dataloader['dataloader'])):
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

                if (adj_E1_mtf is None):
                    # Infer motif matching scores
                    tf2rNet.eval()
                    with torch.no_grad():
                        mtf_pred_l = []
                        for _, (X, seq_data_batch_idx) in tqdm(enumerate(train_seq_dataloader, 0)):
                            mtf_pred  = tf2rNet(seq=X[0].to(self.opt.device), motif=True)
                            mtf_pred_l.append(mtf_pred)
                        adj_E1_mtf = torch.cat(mtf_pred_l).T                        
                        del mtf_pred_l, mtf_pred

                # TF2rNet forward pass
                if not i%self.opt.n_it_acc: # Accumulate gradients for TF2rNet every n_it_acc iterations
                    tf2rNet.eval()
                    if not self.opt.enformer_embs_file:
                        tf2rNet_func_encoder.eval()
                    with torch.no_grad():
                        tf_pred_l = []
                        for j, (X, seq_data_batch_idx) in tqdm(enumerate(train_seq_dataloader, 0)):
                            if not self.opt.enformer_embs_file:
                                tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device))
                                tf_pred = tf2rNet(emb=tf_pred, motif=False)
                            else:
                                tf_pred = tf2rNet(emb=X[1].to(self.opt.device), motif=False)
                            tf_pred_l.append(tf_pred)
                        adj_E1_old = torch.cat(tf_pred_l).T
                    tf2rNet.train()
                    if not self.opt.enformer_embs_file:
                        tf2rNet_func_encoder.train()
                    del tf_pred_l, tf_pred

                # train on n random regions for backpropagating gradients
                try:
                    X, seq_data_batch_idx = next(train_seq_dataloader_shuffle_iterator)
                except StopIteration:
                    train_seq_dataloader_shuffle_iterator = iter(train_seq_dataloader_shuffle)
                    X, seq_data_batch_idx = next(train_seq_dataloader_shuffle_iterator)
                if not self.opt.enformer_embs_file:
                    tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device))
                    tf_pred = tf2rNet(emb=tf_pred, motif=False)
                else:
                    tf_pred = tf2rNet(emb=X[1].to(self.opt.device), motif=False)
                
                adj_E1 = adj_E1_old.clone()
                adj_E1[:, seq_data_batch_idx] = tf_pred.T
                if not (i+1)%self.opt.n_it_acc:
                    del adj_E1_old
                del tf_pred

                # Apply motif prior
                adj_E1 = adj_E1_mtf * adj_E1

                # VAE forward pass
                inputs_rna, inputs_atac, _, dropout_mask_rna = data_batch
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                if self.opt.dropout_loss:
                    loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, _,  _, _, _, _, _, f1_atac = vae(
                        inputs_rna, inputs_atac, dropout_mask_rna=dropout_mask_rna.to(self.opt.device),
                        dropout_mask_atac=True, opt=self.opt, adj_E1=adj_E1)
                else:
                    loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, _, _,  _, _, _, _, f1_atac = vae(
                        inputs_rna, inputs_atac, dropout_mask_rna=None,
                        dropout_mask_atac=None, opt=self.opt, adj_E1=adj_E1)

                # Compute sparse loss
                with torch.no_grad():
                    E1_sparse = adj_E1.abs().mean(1).mean()
                    E2_sparse = vae.adj_E2.abs().mean()

                loss.backward()
                if not i%self.opt.n_it_acc: # Accumulate gradients for TF2rNet every n_it_acc iterations
                    optimizer.step()
                    if (not self.opt.enformer_embs_file):
                        optim_func_enc.step()
                    # Reset optimizers
                    optimizer.zero_grad(True)
                    if (not self.opt.enformer_embs_file):
                        optim_func_enc.zero_grad(True)                        

                # Tensorboard logs
                n_iter = (epoch*len(train_dataloader['dataloader'])) + i 
                writer.add_scalar('Loss/total', loss.detach().item(), n_iter)
                writer.add_scalar('Loss/rec_rna', loss_rec_rna.item(), n_iter)
                writer.add_scalar('Loss/rec_atac', loss_rec_atac.item(), n_iter)
                writer.add_scalar('Loss/kl_rna', loss_gauss_rna.item(), n_iter)
                writer.add_scalar('Loss/l1_E1', E1_sparse.detach().item(), n_iter)
                writer.add_scalar('Loss/l1_E2', E2_sparse.detach().item(), n_iter)
                writer.add_scalar('Loss/f1_atac', f1_atac.detach().item(), n_iter)
            print('epoch:', epoch)

            # Save model
            if not self.opt.enformer_embs_file:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': vae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_tf2r_state_dict': optim_func_enc.state_dict(),
                }, self.opt.save_name + '/model.pth')
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': vae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, self.opt.save_name + '/model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': tf2rNet.state_dict(),
            }, self.opt.save_name + '/model_tf2r.pth')
            if not self.opt.enformer_embs_file:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': tf2rNet_func_encoder.state_dict(),
                }, self.opt.save_name + '/model_tf2r_encoder.pth')

           
            # Evaluate test set
            if (self.opt.test_size > 0):
                with torch.no_grad():
                    vae.eval()
                    loss_all, rec_rna, rec_atac, loss_kl_rna, loss_sparse = [], [], [], [], []
                    for i, data_batch in tqdm(enumerate(test_dataloader['dataloader'], 0), unit="batch", total=len(test_dataloader['dataloader'])):
                        inputs_rna, inputs_atac, _, dropout_mask_rna  = data_batch
                        inputs_rna = Variable(inputs_rna.type(Tensor))
                        inputs_atac = Variable(inputs_atac.type(Tensor))

                        if self.opt.dropout_loss:
                            loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, _, _, _, _, _, _, _ = vae(
                                inputs_rna, inputs_atac, dropout_mask_rna=dropout_mask_rna.to(self.opt.device),
                                dropout_mask_atac=True, opt=self.opt, adj_E1=adj_E1)
                        else:
                            loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, _, _, _, _, _, _, _ = vae(
                                inputs_rna, inputs_atac, dropout_mask_rna=None,
                                dropout_mask_atac=None, opt=self.opt, adj_E1=adj_E1)
    
                        sparse_loss = adj_E1.abs().mean(1).mean()

                        if loss.detach().item() < best_loss:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': vae.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                            }, self.opt.save_name + '/best_model.pth')
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': tf2rNet.state_dict(),
                            }, self.opt.save_name + '/best_model_tf2r.pth')
                            best_loss = loss.detach().item()
                             
                        rec_rna.append(loss_rec_rna.item())
                        rec_atac.append(loss_rec_atac.item())
                        loss_all.append(loss.detach().item())
                        loss_kl_rna.append(loss_gauss_rna.item())
                        loss_sparse.append(sparse_loss.detach().item())
   
                    del loss, loss_rec_rna, loss_rec_atac, loss_gauss_rna, sparse_loss

                    # Tensorboard logs
                    writer.add_scalar('Test/loss_total', np.mean(loss_all), epoch)
                    writer.add_scalar('Test/rec_rna', np.mean(rec_rna), epoch)
                    writer.add_scalar('Test/rec_atac', np.mean(rec_atac), epoch)
                    writer.add_scalar('Test/kl_rna', np.mean(loss_kl_rna), epoch)
                    writer.add_scalar('Test/l1_A', np.mean(loss_sparse), epoch)

                    del loss_all, rec_rna, rec_atac, loss_kl_rna, loss_sparse
#                    scheduler.step()
