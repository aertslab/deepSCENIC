import datetime
import json
import os
import pickle
from typing import Tuple
import h5py
import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import load_npz
from torch.autograd import Variable
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score
from tqdm import tqdm
from enformer_pytorch import Enformer
from enformer_pytorch import GenomeIntervalDataset

from src.tf2rNet.models import MotifNet
from src.tf2rNet.utils import *
from src.model import VAE
from src.utils import EarlyStopping, format_region_to_bed


## Build dataloaders ##
def build_seq_dataloader(opt, ad=None):
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
            assert all(len(tensors[0]) == len(tensor) for tensor in tensors), "Size mismatch between tensors"
            self.tensors = tensors

        def __getitem__(self, index):
            return (tuple(tensor[index] for tensor in self.tensors), index)

        def __len__(self):
            return len(self.tensors[0])

    output_file = open(opt.save_name + 'enhancer.bed', 'w')
    [format_region_to_bed(r, output_file, seq_len=opt.seq_len) for r in ad.var_names.to_list()]
    output_file.close()

    ds = GenomeIntervalDataset(
        bed_file = opt.save_name+ 'enhancer.bed',   # bed file - columns 0, 1, 2 must be <chromosome>, <start position>, <end position>
        fasta_file = opt.fasta, # path to fasta file
        return_seq_indices = False, # return nucleotide indices (ACGTN) or one hot encodings
        shift_augs = (-3, +3),    # random shift augmentations from -2 to +2 basepairs
        rc_aug = True, # use reverse complement augmentation with 50% probability
        context_length = opt.seq_len, #sequence length to extract
        return_augs = False  # return the augmentation meta data
    )
     
    data = TensorDatasetWithIndex(ds)
    dataloader =  DataLoader(data, batch_size=opt.TF2rNet_batch_size, shuffle=False, num_workers=0)
    dataloader_shuffle =  DataLoader(data, batch_size=opt.TF2rNet_batch_size, shuffle=True, num_workers=0)

    return dataloader, dataloader_shuffle
 
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

    # Get scaled values
    # data_rna = pd.DataFrame(data_rna.X, index=list(data_rna.obs_names), columns=rna_gene_name)
    # data_atac = pd.DataFrame(data_atac.X, index=list(data_atac.obs_names), columns=atac_region_name)

    num_genes_rna = data_rna.shape[1]
    num_regions_atac = data_atac.shape[1]

    feat_rna = torch.FloatTensor(data_rna.X)
    feat_atac = torch.FloatTensor(data_atac.X) 
    data = TensorDataset(feat_rna, feat_atac, torch.LongTensor(list(range(len(feat_rna)))))

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=opt.train, num_workers=0)

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

    def init_data(self, train=False, test=False):
        # Read TFs list
        TFs_df = pd.read_csv(self.opt.TF_file, header=None, names=["name"], usecols=[0])
        TFs = set(TFs_df['name'])

        # Read region to gene mask
        r2g_dist_coo = load_npz(self.opt.r2g_mask)

        if train==True:
            # Read data
            print("reading data...")
            data_rna_full = sc.read(self.opt.data_rna_file)
            data_rna = sc.read(self.opt.data_rna_file_train)
            try:
                data_stds = data_rna_full[:, data_rna.var_names].X.std(0)
            except AttributeError:
                data_stds = data_rna_full[:, data_rna.var_names].X.toarray().std(0)
            data_atac = sc.read(self.opt.data_atac_file_train)
            print("data read!")
            print(data_rna)
            print(data_atac)
        elif test==True:
            # Read data
            print("reading data...")
            data_rna_full = sc.read(self.opt.data_rna_file)
            data_rna = sc.read(self.opt.data_rna_file_test)
            try:
                data_stds = data_rna_full[:, data_rna.var_names].X.std(0)
            except AttributeError:
                data_stds = data_rna_full[:, data_rna.var_names].X.toarray().std(0)
            data_atac = sc.read(self.opt.data_atac_file_test)
            print("data read!")
            print(data_rna)
            print(data_atac)
        else:
            # Read data
            print("reading data...")
            data_rna = sc.read(self.opt.data_rna_file)
            try:
                data_stds = data_rna.X.std(0)
            except AttributeError:
                data_stds = data_rna.X.toarray().std(0)
            data_atac = sc.read(self.opt.data_atac_file)
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
        data_rna.X = data_rna.X / data_stds

        # Build RNA/ATAC dataloader
        dataloader = build_dataloader(data_rna, data_atac, self.opt.batch_size, self.opt)        

        # Build sequence dataloader
        train_seq_dataloader, train_seq_data_shuffle = build_seq_dataloader(self.opt, ad=data_atac)

        return dataloader, TFs_idx, r2g_dist_coo, train_seq_dataloader, train_seq_data_shuffle

    def pretrain(self):        
        opt = self.opt

        # Initialize Tensorboard writer
        writer = SummaryWriter(opt.logs + '/logs/TF2rNet/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Initialize data
        _, _, r2g_dist_coo, test_seq_dataloader, _ = self.init_data(test=True)
        dataloader, TFs_idx, r2g_dist_coo, seq_dataloader, _  = self.init_data(train=True)
        # Initialize models
        tf2rNet = MotifNet(self.TFs, opt.TF2rNet_bottleneck_size, emb_len=opt.emb_len, dev=opt.device).float().to(opt.device)
        tf2rNet_func_encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=opt.emb_len, dropout_rate = 0.1).to(opt.device)
        vae = VAE(TFs_idx, r2g_dist_coo, 1, opt.n_hidden, dev=opt.device).float().to(opt.device)
        # Load pretrained models
        if opt.load_model is not None:
            tf2rNet.load_state_dict(torch.load(opt.save_name + 'model_tf2r.pth',  map_location=torch.device(opt.device))['model_state_dict'])
            tf2rNet_func_encoder.load_state_dict(torch.load(opt.save_name + 'model_tf2r_encoder.pth',  map_location=torch.device(opt.device))['model_state_dict'])
            vae.load_state_dict(torch.load(opt.save_name + 'model.pth',  map_location=torch.device(opt.device))['model_state_dict'])       

        optimizer =  optim.Adam([{'params':tf2rNet_func_encoder.parameters(), 'lr':opt.lr}, {'params':tf2rNet.parameters(), 'lr':opt.lr}, {'params':vae.parameters(), 'lr':opt.lr}]) 

        train_dataloader_iter = iter(dataloader['dataloader']) # Initialize iterator for vae dataloader
        for epoch in range(opt.n_epochs):
            tf2rNet.train()
            tf2rNet_func_encoder.train()
            vae.train()
            for j, (seq, seq_data_batch_idx) in tqdm(enumerate(seq_dataloader, 0)):
                optimizer.zero_grad(True)

                x = tf2rNet_func_encoder(seq[0].to(opt.device), return_only_embeddings=True)
                X_E1 = tf2rNet(emb=x) 
                # vae forward pass
                try:
                    data_batch = next(train_dataloader_iter)
                except StopIteration:
                    train_dataloader_iter = iter(dataloader['dataloader'])
                    data_batch = next(train_dataloader_iter)
                inputs_rna, inputs_atac, _ = data_batch
                x_rna_tfs = inputs_rna[:, TFs_idx]
                
                loss, loss_rec_atac, loss_gauss_rna, f1_atac = vae.pretrain(
                x_rna_tfs.to(opt.device), inputs_atac.to(opt.device), opt=opt, adj_E1=X_E1, idxs=seq_data_batch_idx.to(opt.device))
                
                E1_sparse = X_E1.abs().mean(1).mean()
                loss = E1_sparse + loss
                
                loss.backward()
                optimizer.step()

                # Tensorboard logs
                n_iter = (epoch*len(seq_dataloader)) + j
                writer.add_scalar('Loss/total', loss.detach().item(), n_iter)
                writer.add_scalar('Loss/rec_atac', loss_rec_atac.item(), n_iter)
                writer.add_scalar('Loss/l1_E1', E1_sparse.item(), n_iter)
                writer.add_scalar('Loss/kl_rna', loss_gauss_rna.item(), n_iter)
                writer.add_scalar('Loss/f1_atac', f1_atac.detach().item(), n_iter)


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
            torch.save({
                'epoch': epoch,
                'model_state_dict': tf2rNet_func_encoder.state_dict(),
            }, opt.save_name + '/model_tf2r_encoder.pth')    

            tf2rNet.eval()
            tf2rNet_func_encoder.eval()
            vae.eval()
            loss_all, rec_atac, loss_kl_rna = [], [], []
            with torch.no_grad():
                for j, (seq, seq_data_batch_idx) in tqdm(enumerate(test_seq_dataloader, 0)):
                    x = tf2rNet_func_encoder(seq[0].to(opt.device), return_only_embeddings=True)
                    X_E1 = tf2rNet(emb=x)     
                    
                    # vae forward pass
                    try:
                        data_batch = next(train_dataloader_iter)
                    except StopIteration:
                        train_dataloader_iter = iter(dataloader['dataloader'])
                        data_batch = next(train_dataloader_iter)
                    inputs_rna, inputs_atac, _ = data_batch
                    x_rna_tfs = inputs_rna[:, TFs_idx]
                
                    loss, loss_rec_atac, loss_gauss_rna, f1_atac = vae.pretrain(
                        x_rna_tfs.to(opt.device), inputs_atac.to(opt.device), opt=opt, adj_E1=X_E1, idxs=seq_data_batch_idx.to(opt.device))

                    loss = loss_rec_atac + loss_gauss_rna + X_E1.abs().mean(1).mean()
                    rec_atac.append(loss_rec_atac.item())
                    loss_all.append(loss.detach().item())
                    loss_kl_rna.append(loss_gauss_rna.item())
                    # Tensorboard logs
                    writer.add_scalar('Test/loss_total', np.mean(loss_all), epoch)
                    writer.add_scalar('Test/rec_atac', np.mean(rec_atac), epoch)
                    writer.add_scalar('Test/kl_rna', np.mean(loss_kl_rna), epoch)
    
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
            dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=False, num_workers=0)

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
        self.opt.train = False
        _, TFs_idx, r2g_dist_coo, seq_dataloader, _ = self.init_data()

        # Initialize TF2rNet model
        model_dict_tf2rNet = torch.load(tf2r_model_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        model_dict_tf2r_func_enc = torch.load(tf2r_func_enc_model_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet_func_encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=5, dropout_rate = 0.1).to(self.opt.device)
        tf2rNet = MotifNet(self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, dev=self.opt.device).float().to(self.opt.device)
        tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
        tf2rNet.load_state_dict(model_dict_tf2rNet)
        tf2rNet_func_encoder.eval()

        # Load pretrained VAE model
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)
        vae.load_state_dict(torch.load(vae_model_path,  map_location=torch.device(self.opt.device))['model_state_dict'])

        with torch.no_grad():
            vae.eval()
            tf2rNet.eval()
            if adj_E1 is None:
                # TF2rNet forward pass
                tf_pred_l = []
                for j, (X, _) in tqdm(enumerate(seq_dataloader, 0), total=len(seq_dataloader)):
                    tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device))
                    tf_pred = tf2rNet(emb=tf_pred)
                    tf_pred_l.append(tf_pred)
                adj_E1 = torch.cat(tf_pred_l)
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

    def to_latent(self, adj_E1, adj_E2):
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
        self.opt.train = False
        dataloader, TFs_idx, r2g_dist_coo, _, _ = self.init_data()

        # Initialize TF2rNet model
        # Load tf2r fuctional encoder
        tf2rNet_func_encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=5, dropout_rate = 0.1).to(self.opt.device)
        model_dict_tf2r_func_enc = torch.load(self.opt.load_model + 'model_tf2r_encoder.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
        # Load tf2r contex head
        tf2rNet = MotifNet(self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, dev=self.opt.device).float().to(self.opt.device)
        model_dict_tf2r = torch.load(self.opt.load_model + 'model_tf2r.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet.load_state_dict(model_dict_tf2r)   
        print("loaded weights for TF2rNet")

        # Initialize VAE
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)
        vae.load_state_dict(torch.load(self.opt.save_name + 'model.pth',  map_location=torch.device(self.opt.device))['model_state_dict'])
        vae.adj_E2 = nn.Parameter(adj_E2)
        with torch.no_grad(): 
            vae.eval()
            print("VAE forward...")
            z_rna_l = []
            z_tf_mu_l = []
            z_tf_var_l = []
            z_rna_atac_l = []
            dec_rna_l = []
            dec_atac_l = []
            for _, data_batch in tqdm(enumerate(dataloader['dataloader'], 0), unit="batch", total=len(dataloader['dataloader'])):
                # VAE forward pass
                inputs_rna, inputs_atac, _ = data_batch
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                out_gen_rna, out_gen_atac, out_inf_rna, enh_act, z_rna = vae.predict(
                        inputs_rna, adj_E1=adj_E1, adj_E2=adj_E2)

                z_rna_l += [z_rna.cpu().numpy()]
                z_tf_mu_l += [out_inf_rna['mean'].cpu().numpy()]
                z_tf_var_l += [out_inf_rna['logvar'].cpu().numpy()]
                z_rna_atac_l += [enh_act.cpu().numpy()]
                dec_rna_l += [out_gen_rna['x_rec'].cpu().numpy()]
                dec_atac_l += [out_gen_atac['x_rec'].cpu().numpy()]

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

    def finetune_r2g_test(self):
        if self.opt.device=='cuda':
            Tensor = torch.cuda.FloatTensor
        elif self.opt.device=='cpu':
            Tensor = torch.FloatTensor

        # Initialize Tensorboard logger
        writer = SummaryWriter(self.opt.logs + '/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        ### Initialize dataloaderss
        self.opt.train = False
        dataloader, TFs_idx, r2g_dist_coo, seq_dataloader, _  = self.init_data(test=True)

        # Load tf2r fuctional encoder
        tf2rNet_func_encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=5, dropout_rate = 0.1).to(self.opt.device)
        model_dict_tf2r_func_enc = torch.load(self.opt.load_model + 'model_tf2r_encoder.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
        # Load tf2r contex head
        tf2rNet = MotifNet(self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, dev=self.opt.device).float().to(self.opt.device)
        model_dict_tf2r = torch.load(self.opt.load_model + 'model_tf2r.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet.load_state_dict(model_dict_tf2r)   
        print("loaded weights for TF2rNet")

        # Initialize VAE
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)
        # Exclude 'adj_E2' from the state dict
        state_dict = torch.load(self.opt.load_model + 'model.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
        if 'adj_E2' in state_dict:
            del state_dict['adj_E2']
        vae.load_state_dict(state_dict, strict=False)
        vae.adj_E2 = nn.Parameter(torch.zeros(r2g_dist_coo.size, device=self.opt.device, requires_grad=True) + vae.eps)
        
        # Freeze layers of vae
        for name, param in vae.named_parameters():
            if name == 'adj_E2':
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Freeze layers of tf2r
        for param in tf2rNet.parameters():
            param.requires_grad = False
        for param in tf2rNet_func_encoder.parameters():
            param.requires_grad = False
        
        # Initialize optimizers
        optimizer = optim.Adam([{'params': vae.adj_E2, 'lr':self.opt.lr}])

        # Training
        adj_E1 = None
        for epoch in range(self.opt.n_epochs):
            vae.train()
            tf2rNet.eval()
            tf2rNet_func_encoder.eval()

            for i, data_batch in tqdm(enumerate(dataloader['dataloader'], 0), unit="batch", total=len(dataloader['dataloader'])):
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

                # TF2rNet forward pass
                if adj_E1 is None:
                    with torch.no_grad():
                        tf_pred_l = []
                        for j, (X, _) in tqdm(enumerate(seq_dataloader, 0)):
                            tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device), return_only_embeddings=True)
                            tf_pred = tf2rNet(emb=tf_pred)
                            tf_pred_l.append(tf_pred)
                        adj_E1 = torch.cat(tf_pred_l)
                    del tf_pred_l, tf_pred

                # VAE forward pass
                inputs_rna, inputs_atac, _ = data_batch
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                loss_rec_rna, loss_rec_atac, loss_gauss_rna, _,  _, _, _, _, _, f1_atac = vae(
                    inputs_rna,
                    inputs_atac,
                    dropout_mask_rna=self.opt.dropout_loss,
                    dropout_mask_atac=self.opt.dropout_loss,
                    opt=self.opt, adj_E1=adj_E1)
                
                # Compute sparse loss
                E2_sparse = vae.adj_E2.abs().mean()

                loss = loss_rec_rna + loss_gauss_rna + E2_sparse
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(True)
                
                # Tensorboard logs
                n_iter = (epoch*len(dataloader['dataloader'])) + i 
                writer.add_scalar('Loss/total', loss.detach().item(), n_iter)
                writer.add_scalar('Loss/rec_rna', loss_rec_rna.item(), n_iter)
                writer.add_scalar('Loss/rec_atac', loss_rec_atac.item(), n_iter)
                writer.add_scalar('Loss/kl_rna', loss_gauss_rna.item(), n_iter)
                writer.add_scalar('Loss/l1_E2', E2_sparse.detach().item(), n_iter)
                writer.add_scalar('Loss/f1_atac', f1_atac.detach().item(), n_iter)
            print('epoch:', epoch)

            # Save tf2r matrix
            torch.save({'vae_E2_test': vae.adj_E2}, self.opt.save_name + '/E2_test.pth')
        
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
        test_dataloader, _, r2g_dist_coo, test_seq_dataloader, _ = self.init_data(test=True)
        train_dataloader, TFs_idx, r2g_dist_coo, train_seq_dataloader, train_seq_dataloader_shuffle = self.init_data(train=True)        

        if self.opt.device=='cuda':
            Tensor = torch.cuda.FloatTensor
        elif self.opt.device=='cpu':
            Tensor = torch.FloatTensor

        # Initialize TF2rNet model
        tf2rNet_func_encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=self.opt.emb_len, dropout_rate = 0.1).to(self.opt.device)
        tf2rNet = MotifNet(self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, dev=self.opt.device).float().to(self.opt.device)
        if self.opt.load_model is not None:
            # Load tf2r fuctional encoder
            model_dict_tf2r_func_enc = torch.load(self.opt.load_model + 'model_tf2r_encoder.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
            tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
            # Load tf2r contex head
            model_dict_tf2r = torch.load(self.opt.load_model + 'model_tf2r.pth',  map_location=torch.device(self.opt.device))['model_state_dict']
            tf2rNet.load_state_dict(model_dict_tf2r)   
            print("loaded weights for TF2rNet")
            

        # Initialize VAE model        
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, dev=self.opt.device).float().to(self.opt.device)
        if self.opt.load_model is not None:
            if os.path.exists(self.opt.load_model + 'model.pth'):
                vae_d = torch.load(self.opt.load_model + 'model.pth',   map_location=torch.device(self.opt.device))['model_state_dict']
                vae.load_state_dict(vae_d)
                print("loaded weights for vae")

        # Initialize optimizers
        optimizer = optim.Adam(vae.parameters(), self.opt.lr)
        optim_func_enc = optim.Adam([{'params':tf2rNet_func_encoder.parameters(), 'lr':self.opt.lr}, {'params':tf2rNet.parameters(), 'lr':self.opt.lr}])
        if self.opt.load_model is not None:
            state_dict = torch.load(self.opt.load_model + 'model.pth',   map_location=torch.device(self.opt.device))
            if 'optimizer_tf2r_state_dict' in state_dict:
                optimizer.load_state_dict(state_dict['optimizer_state_dict'])
                optim_func_enc.load_state_dict(state_dict['optimizer_tf2r_state_dict'])

        adj_E1 = None
        adj_E1_test = None
        adj_E1_old = None
        best_loss =  float('inf')
        train_seq_dataloader_shuffle_iterator = iter(train_seq_dataloader_shuffle) # Initialize iterator for sequence dataloader
        for epoch in range(self.opt.n_epochs):
            vae.train()
            tf2rNet.eval()
            tf2rNet_func_encoder.eval()
            with torch.no_grad():
                tf_pred_l = []
                # seq_l1_l = []
                for j, (X, seq_data_batch_idx) in tqdm(enumerate(train_seq_dataloader, 0)):
                    tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device), return_only_embeddings=True)
                    tf_pred = tf2rNet(emb=tf_pred)
                    tf_pred_l.append(tf_pred)
                adj_E1_old = torch.cat(tf_pred_l)
            del tf_pred_l, tf_pred
            adj_E1 = adj_E1_old.clone()
            tf2rNet.train()
            tf2rNet_func_encoder.train()            
 
            for i, data_batch in tqdm(enumerate(train_dataloader['dataloader'], 0), unit="batch", total=len(train_dataloader['dataloader'])):
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                # TF2rNet forward pass
                adj_E1 = adj_E1_old.clone()

                # train on n random regions for backpropagating gradients
                try:
                    X, seq_data_batch_idx = next(train_seq_dataloader_shuffle_iterator)
                except StopIteration:
                    train_seq_dataloader_shuffle_iterator = iter(train_seq_dataloader_shuffle)
                    X, seq_data_batch_idx = next(train_seq_dataloader_shuffle_iterator)
                tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device), return_only_embeddings=True)
                tf_pred = tf2rNet(emb=tf_pred)
                
                adj_E1[seq_data_batch_idx, :] = tf_pred
                with torch.no_grad():
                    adj_E1_old[seq_data_batch_idx, :] = tf_pred
                del tf_pred

                # Apply motif prior
                # adj_E1 = adj_E1_mtf * adj_E1

                # VAE forward pass
                inputs_rna, inputs_atac, _ = data_batch
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                loss_rec_rna, loss_rec_atac, loss_gauss_rna, _,  _, _, _, _, _, f1_atac = vae(
                    inputs_rna,
                    inputs_atac,
                    dropout_mask_rna=self.opt.dropout_loss,
                    dropout_mask_atac=self.opt.dropout_loss, 
                    opt=self.opt,
                    adj_E1=adj_E1)
                
                # Compute sparse loss
                # with torch.no_grad():
                E1_sparse = adj_E1.abs().mean(1).mean()
                E2_sparse = vae.adj_E2.abs().mean()

                if epoch>=self.opt.warmup_vae:
                    loss = loss_rec_rna + loss_gauss_rna + loss_rec_atac + E1_sparse + E2_sparse
                else:
                    loss = loss_rec_rna + loss_gauss_rna + E1_sparse + E2_sparse

                loss.backward()
                optimizer.step()
                optim_func_enc.step()
                # Reset optimizers
                optimizer.zero_grad(True)
                optim_func_enc.zero_grad(True)                        

                # Tensorboard logs
                n_iter = (epoch*len(train_dataloader['dataloader'])) + i 
                writer.add_scalar('Loss/total', loss.detach().item(), n_iter)
                writer.add_scalar('Loss/rec_rna', loss_rec_rna.detach().item(), n_iter)
                writer.add_scalar('Loss/rec_atac', loss_rec_atac.detach().item(), n_iter)
                writer.add_scalar('Loss/kl_rna', loss_gauss_rna.detach().item(), n_iter)
                writer.add_scalar('Loss/l1_E1', E1_sparse.detach().item(), n_iter)
                writer.add_scalar('Loss/l1_E2', E2_sparse.detach().item(), n_iter)
                writer.add_scalar('Loss/f1_atac', f1_atac.detach().item(), n_iter)

                # Save model
                if not i%100:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': vae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'optimizer_tf2r_state_dict': optim_func_enc.state_dict(),
                    }, self.opt.save_name + '/model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': tf2rNet.state_dict(),
                    }, self.opt.save_name + '/model_tf2r.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': tf2rNet_func_encoder.state_dict(),
                    }, self.opt.save_name + '/model_tf2r_encoder.pth')

            print('epoch:', epoch)
            # Evaluate test set
            with torch.no_grad():
                vae.eval()
                tf2rNet.eval()
                tf2rNet_func_encoder.eval()

                if adj_E1_test is None:
                    tf_pred_l = []
                    for j, (X, seq_data_batch_idx) in tqdm(enumerate(test_seq_dataloader, 0)):
                        tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device), return_only_embeddings=True)
                        tf_pred = tf2rNet(emb=tf_pred)
                        tf_pred_l.append(tf_pred)
                    adj_E1_test = torch.cat(tf_pred_l)
                    del tf_pred_l, tf_pred

                
                loss_all, f1_score, rec_atac, loss_kl_rna, loss_sparse = [], [], [], [], []
                for i, data_batch in tqdm(enumerate(test_dataloader['dataloader'], 0), unit="batch", total=len(test_dataloader['dataloader'])):
                    inputs_rna, inputs_atac, _  = data_batch
                    inputs_rna = Variable(inputs_rna.type(Tensor))
                    inputs_atac = Variable(inputs_atac.type(Tensor))

                    _, out_gen_atac, out_inf_rna, _, _= vae.predict(
                       inputs_rna, adj_E1=adj_E1_test)
                    
                    if self.opt.bin_acc==True:
                        loss_acc = 'bce'
                        f1 = F1Score(task='binary',num_classes=1).to(self.opt.device)
                        f1_atac = f1(out_gen_atac['x_rec'].ravel(), inputs_atac.int().ravel())
                    else:
                        loss_acc = 'mae'
                        f1_atac = torch.Tensor([0])
                    
                    loss_rec_atac = vae.losses.reconstruction_loss(inputs_atac, out_gen_atac['x_rec'], False, rec_type=loss_acc)
                    loss_gauss_rna = vae.losses.gaussian_loss(out_inf_rna['mean'], out_inf_rna['logvar']) * self.opt.beta

                    loss = loss_gauss_rna + loss_rec_atac

                    sparse_loss = adj_E1_test.abs().mean(1).mean()
                    loss = loss_rec_atac + loss_gauss_rna + sparse_loss

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
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': tf2rNet_func_encoder.state_dict(),
                        }, self.opt.save_name + '/best_model_tf2r_encoder.pth')                        
                        best_loss = loss.detach().item()
                            
                    rec_atac.append(loss_rec_atac.item())
                    loss_all.append(loss.detach().item())
                    loss_kl_rna.append(loss_gauss_rna.item())
                    loss_sparse.append(sparse_loss.detach().item())
                    f1_score.append(f1_atac.detach().item())

                del loss, loss_rec_atac, loss_gauss_rna, sparse_loss

                # Tensorboard logs
                writer.add_scalar('Test/loss_total', np.mean(loss_all), epoch)
                writer.add_scalar('Test/rec_atac', np.mean(rec_atac), epoch)
                writer.add_scalar('Test/kl_rna', np.mean(loss_kl_rna), epoch)
                writer.add_scalar('Test/l1_A', np.mean(loss_sparse), epoch)
                writer.add_scalar('Test/f1_atac', np.mean(f1_score), epoch)

                del loss_all, rec_atac, loss_kl_rna, loss_sparse
#                    scheduler.step()
