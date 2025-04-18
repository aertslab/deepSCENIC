import datetime
import json
import os
import pickle
from typing import Tuple
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.utils import compute_class_weight
from scipy.sparse import issparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import load_npz
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from src.utils import EarlyStopping, format_region_to_bed, build_ppi_network, nx_to_pyg_edge_index

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

## Build dataloaders ##
def build_seq_dataloader(opt, ad=None):
    """ Build dataloader for TF2rNet
            Params
            ------
            opt: model hyperparams
            ad: scATAC-seq data
            pretrain: if True, build dataloader for TF2rNet pretraining
    """
    output_file = open(opt.save_name + 'enhancer.bed', 'w')
    [format_region_to_bed(r, output_file, seq_len=opt.seq_len) for r in ad.var_names.to_list()]
    output_file.close()

    if opt.train==True:
        ds = GenomeIntervalDataset(
            bed_file = opt.save_name+ 'enhancer.bed',   # bed file - columns 0, 1, 2 must be <chromosome>, <start position>, <end position>
            fasta_file = opt.fasta, # path to fasta file
            return_seq_indices = False, # return nucleotide indices (ACGTN) or one hot encodings
            shift_augs = (-3, +3),    # random shift augmentations from -2 to +2 basepairs
            rc_aug = True, # use reverse complement augmentation with 50% probability
            context_length = opt.seq_len, #sequence length to extract
            return_augs = False  # return the augmentation meta data
        )
    else:
        ds = GenomeIntervalDataset(
            bed_file = opt.save_name+ 'enhancer.bed',   # bed file - columns 0, 1, 2 must be <chromosome>, <start position>, <end position>
            fasta_file = opt.fasta, # path to fasta file
            return_seq_indices = False, # return nucleotide indices (ACGTN) or one hot encodings
            shift_augs = (0, 0),    # random shift augmentations from -2 to +2 basepairs
            rc_aug = False, # use reverse complement augmentation with 50% probability
            context_length = opt.seq_len, #sequence length to extract
            return_augs = False  # return the augmentation meta data
        )
     
    data = TensorDatasetWithIndex(ds)
    dataloader =  DataLoader(data, batch_size=opt.seqs_batch_size, shuffle=False, num_workers=0)
    if opt.balance_dars==True:
        region_weights = sc.read(opt.data_atac_file).var.dar + 1
        region_weights = region_weights.loc[ad.var_names].values
        sampler = WeightedRandomSampler(region_weights, len(region_weights))

        dataloader_shuffle = DataLoader(data, batch_size=opt.seqs_batch_size, shuffle=False, num_workers=0, sampler=sampler)
    else:
        dataloader_shuffle =  DataLoader(data, batch_size=opt.seqs_batch_size, shuffle=True, num_workers=0)

    return dataloader, dataloader_shuffle
 
def build_dataloader(data_rna, data_atac, batch_size, opt):
    """ Build dataloader for VAE
  
        Params
        ------
        data_rna: scRNA-seq data
        data_atac: scATAC-seq data
        opt: model hyperparams
    """
    # Get batch info
    if opt.batch_key is not None:
        batch_ids = sc.read(opt.data_rna_file).obs.loc[:, opt.batch_key].unique()
        batch_id_d = dict(zip(batch_ids, torch.tensor(pd.get_dummies(batch_ids).values).float()))

    rna_gene_name = list(data_rna.var_names)
    atac_region_name = list(data_atac.var_names)

    # # Check if sparse data
    # if issparse(data_rna.X)==True:
    #     data_rna.X = data_rna.X.A.copy()
    #     print(data_rna.X)

    # if issparse(data_atac.X)==True:
    #     data_atac.X = data_atac.X.A

    # Get scaled values
    # data_rna = pd.DataFrame(data_rna.X, index=list(data_rna.obs_names), columns=rna_gene_name)
    # data_atac = pd.DataFrame(data_atac.X, index=list(data_atac.obs_names), columns=atac_region_name)

    num_genes_rna = data_rna.shape[1]
    num_regions_atac = data_atac.shape[1]

    if issparse(data_rna.X)==True:
        feat_rna = torch.FloatTensor(data_rna.X.A)
    else:
        feat_rna = torch.FloatTensor(data_rna.X)

    if issparse(data_atac.X)==True:
        feat_atac = torch.FloatTensor(data_atac.X.A)
    else:
        feat_atac = torch.FloatTensor(data_atac.X) 

    if opt.batch_key is not None:
        batch_id = torch.FloatTensor(np.vstack(data_rna.obs.loc[:, opt.batch_key].map(batch_id_d).values))
        data = TensorDataset(feat_rna, feat_atac, batch_id, torch.LongTensor(list(range(len(feat_rna)))))
        opt.batch_id_dim = batch_id.shape[1]
    else:
        data = TensorDataset(feat_rna, feat_atac, torch.LongTensor(list(range(len(feat_rna)))))

    if opt.train==True:
        if opt.balance_class==True:
            class_weight = compute_class_weight('balanced', classes=np.unique(data_rna.obs[opt.ann_key]), y=data_rna.obs[opt.ann_key].values)
            class_weight_d = dict(zip(np.unique(data_rna.obs[opt.ann_key]), class_weight))
            samples_weight = data_rna.obs[opt.ann_key].map(class_weight_d).values   
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            shuffle = False
        else:
            sampler = None
            shuffle = True
    else:
        sampler = None
        shuffle = False

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0, sampler=sampler)

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

    def cosine_similarity(self, mtx):
        # Transpose the matrix to have columns as vectors
        matrix_transposed = mtx.t()

        # Normalize the columns to unit vectors (optional but recommended)
        matrix_transposed_normalized = F.normalize(matrix_transposed, p=2, dim=1)

        # Compute cosine similarity
        cosine_sim_matrix = torch.mm(matrix_transposed_normalized, matrix_transposed_normalized.t())

        return torch.abs(cosine_sim_matrix - torch.eye(cosine_sim_matrix.shape[0]).to(self.opt.device)).mean()

    def sign_penalty(self, mtx):
        mtx = mtx.T
        mtx_pos = F.relu(mtx).mean(0)
        mtx_neg = F.relu(-1*mtx).mean(0)

        return torch.mean(mtx_pos * mtx_neg)

    def _cosine_similarity(self, mtx, prior_mtx=None, seq_idxs=None, cos_sims=None):
        # Transpose the matrix to have columns as vectors
        matrix_transposed = mtx.t()

        # Normalize the columns to unit vectors (optional but recommended)
        matrix_transposed_normalized = F.normalize(matrix_transposed, p=2, dim=1)

        if prior_mtx is None:
            prior_mtx_normalized = matrix_transposed_normalized.t()
        else:
            prior_mtx = prior_mtx.to(self.opt.device)
            if seq_idxs is not None:
                prior_mtx = prior_mtx[seq_idxs, :]    
            idxs = torch.where(prior_mtx.sum(0)==0)[0]
            prior_mtx[:, idxs] = mtx[:, idxs]
            prior_mtx_normalized = F.normalize(prior_mtx, p=2, dim=1)

        # for i in range(4):
        i = torch.randint(0, 4, (1,)).item()
        if i == 0:
            # pos-pos
            mtx = matrix_transposed_normalized.clone()
            prior_mtx = prior_mtx_normalized.clone()
            mtx_abs = matrix_transposed.clone()
            mtx_abs[mtx_abs<0] = 0
            mtx_abs = mtx_abs.T.abs().mean(0)[:,None]
            mtx[mtx<0] = 0
            prior_mtx[prior_mtx<0] = 0
            # Compute cosine similarity
            cosine_sim_matrix = torch.mm(mtx, prior_mtx)
            cosine_sim_matrix = cosine_sim_matrix - torch.eye(cosine_sim_matrix.shape[0]).to(self.opt.device)
            cosine_sim_matrix[cosine_sim_matrix<0] = 0
            cos_sims[i] = torch.mean((cosine_sim_matrix) * mtx_abs)
        elif i == 1:
            # neg-neg
            mtx = matrix_transposed_normalized.clone()
            prior_mtx = prior_mtx_normalized.clone()
            mtx_abs = matrix_transposed.clone()
            mtx_abs[mtx_abs>0] = 0
            mtx_abs = mtx_abs.T.abs().mean(0)[:,None]
            mtx[mtx>0] = 0
            prior_mtx[prior_mtx>0] = 0
            # Compute cosine similarity
            cosine_sim_matrix = torch.mm(mtx.abs(), prior_mtx.abs())
            cosine_sim_matrix = cosine_sim_matrix - torch.eye(cosine_sim_matrix.shape[0]).to(self.opt.device)
            cosine_sim_matrix[cosine_sim_matrix<0] = 0
            cos_sims[i] = torch.mean((cosine_sim_matrix) * mtx_abs)
        elif i == 2:    
            # pos-neg
            mtx = matrix_transposed_normalized.clone()
            prior_mtx = prior_mtx_normalized.clone()
            mtx_abs = matrix_transposed.clone()
            mtx_abs[mtx_abs<0] = 0
            mtx_abs = mtx_abs.T.abs().mean(0)[:,None]
            mtx[mtx<0] = 0
            prior_mtx[prior_mtx>0] = 0
            # Compute cosine similarity
            cosine_sim_matrix = torch.mm(mtx, prior_mtx.abs())
            cosine_sim_matrix = cosine_sim_matrix - torch.eye(cosine_sim_matrix.shape[0]).to(self.opt.device)
            cosine_sim_matrix[cosine_sim_matrix<0] = 0
            cos_sims[i] = torch.mean((cosine_sim_matrix) * mtx_abs)
        elif i == 3:
            # neg-pos
            mtx = matrix_transposed_normalized.clone()
            prior_mtx = prior_mtx_normalized.clone()
            mtx_abs = matrix_transposed.clone()
            mtx_abs[mtx_abs>0] = 0
            mtx_abs = mtx_abs.T.abs().mean(0)[:,None]
            mtx[mtx>0] = 0
            prior_mtx[prior_mtx<0] = 0
            # Compute cosine similarity
            cosine_sim_matrix = torch.mm(mtx.abs(), prior_mtx)
            cosine_sim_matrix = cosine_sim_matrix - torch.eye(cosine_sim_matrix.shape[0]).to(self.opt.device)
            cosine_sim_matrix[cosine_sim_matrix<0] = 0
            cos_sims[i] = torch.mean((cosine_sim_matrix) * mtx_abs)
        return torch.sum(cos_sims)

    def init_data(self, train=False, test=False):
        # Read TFs list
        TFs_df = pd.read_csv(self.opt.TF_file, header=None, names=["name"], usecols=[0])
        TFs = set(TFs_df['name'])
        


        # Read region to gene mask
        r2g_dist_coo = load_npz(self.opt.r2g_mask)

        # Load prior if any
        if self.opt.tf2r_prior is None:
            self.prior_E1 = None
        else:
            self.prior_E1 = torch.tensor(pd.read_pickle(self.opt.tf2r_prior).values).float()

        if self.opt.tf2r_prior_test is None:
            self.prior_E1_test = None
        else:
            self.prior_E1_test = torch.tensor(pd.read_pickle(self.opt.tf2r_prior_test).values).float()       

        if train==True:
            # Read data
            print("reading data...")
            data_rna = sc.read(self.opt.data_rna_file)
            self.opt.n_tot_genes = data_rna.shape[1]

            data_rna_train = sc.read(self.opt.data_rna_file_train)
            genes_idx = np.array([data_rna.var_names.get_loc(i) for i in data_rna_train.var_names])
            ppi_edge_df = pd.read_csv('https://raw.githubusercontent.com/madilabcode/scNET/11a400488c4f4f4e69b6945eb99a0dde0b8cf7c2/scNET/Data/format_h_sapiens.csv', index_col=0)
            _, ppi, node_features = build_ppi_network(data_rna, ppi_edge_df, human_flag=True)
            ppi_edge_index, _ = nx_to_pyg_edge_index(ppi)

            TFs_idx = np.where(data_rna.var_names.isin(TFs))[0]

            try:
                data_stds = data_rna.X.std(0)
            except AttributeError:
                data_stds = data_rna.X.toarray().std(0)

            data_rna = data_rna[data_rna_train.obs.index]
            data_atac = sc.read(self.opt.data_atac_file_train)
            print("data read!")
            print(data_rna)
            print(data_atac)
        elif test==True:
            # Read data
            print("reading data...")
            data_rna = sc.read(self.opt.data_rna_file)
            self.opt.n_tot_genes = data_rna.shape[1]

            data_rna_test = sc.read(self.opt.data_rna_file_test)
            genes_idx = np.arange(data_rna.shape[1])
            try:
                data_stds = data_rna.X.std(0)
            except AttributeError:
                data_stds = data_rna.X.toarray().std(0)

            data_rna = data_rna[data_rna_test.obs.index]
            data_atac = sc.read(self.opt.data_atac_file_test)
            ppi_edge_df = pd.read_csv('https://raw.githubusercontent.com/madilabcode/scNET/11a400488c4f4f4e69b6945eb99a0dde0b8cf7c2/scNET/Data/format_h_sapiens.csv', index_col=0)
            _, ppi, node_features = build_ppi_network(data_rna, ppi_edge_df, human_flag=True)
            ppi_edge_index, _ = nx_to_pyg_edge_index(ppi)
            print("data read!")
            print(data_rna)
            print(data_atac)            
        else:
            # Read data
            print("reading data...")
            data_rna = sc.read(self.opt.data_rna_file)
            self.opt.n_tot_genes = data_rna.shape[1]

            genes_idx = np.arange(data_rna.shape[1])
            try:
                data_stds = data_rna.X.std(0)
            except AttributeError:
                data_stds = data_rna.X.toarray().std(0)
            data_atac = sc.read(self.opt.data_atac_file)
            ppi_edge_df = pd.read_csv('https://raw.githubusercontent.com/madilabcode/scNET/11a400488c4f4f4e69b6945eb99a0dde0b8cf7c2/scNET/Data/format_h_sapiens.csv', index_col=0)
            _, ppi, node_features = build_ppi_network(data_rna, ppi_edge_df, human_flag=True)
            ppi_edge_index, _ = nx_to_pyg_edge_index(ppi)

            print("data read!")
            print(data_rna)
            print(data_atac)

        # Get TFs index in rna data
        TFs_idx = np.where(data_rna.var_names.isin(TFs))[0]
        TFs = data_rna.var_names[data_rna.var_names.isin(TFs)].to_list()
        self.TFs = TFs
        ppi_tfs_idx = dict([[node_features.index.get_loc(g), data_rna.var_names[TFs_idx].get_loc(g)] for g in data_rna.var_names[TFs_idx] if g in node_features.index])
        ppi_genes_idx = np.array([data_rna.var_names.get_loc(i) for i in node_features.index])


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

        return dataloader, TFs_idx, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index, r2g_dist_coo, train_seq_dataloader, train_seq_data_shuffle


    def simulate_perturbation(self, vae, perturbation={}, clip_val=99.9, n_iter=5, original_matrix=None, keep_intermediate=False, adj_E1=None, adj_E2=None, adj_E1_pert=None, eps=1e-8):
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

        ### Initialize dataloaders
        self.opt.train = False
        with torch.no_grad():
            vae.eval()
            
            if keep_intermediate:
                perturbation_over_iter = {}
                fcs = {}
            # Initialize perturbed matrix                
            perturbed_matrix = original_matrix.copy()

            # Perform rounds of perturbation
            original_matrix_t = torch.tensor(original_matrix.values).to(torch.float).to(self.opt.device)
            original_matrix_t = original_matrix_t[:, vae.TFs_idx]

            z_tf_orig = vae.inference_rna(original_matrix_t.view(original_matrix_t.size(0), -1, 1))['mean'].to(torch.float)                
            Wrna = adj_E1 @ adj_E2
            Wrna_ct = z_tf_orig @ Wrna

            if adj_E1_pert is not None:
                Wrna_pert = adj_E1_pert @ adj_E2
            else:
                Wrna_pert = adj_E1 @ adj_E2

            orig_mtx_p99 = np.percentile(original_matrix, clip_val)
            for i in tqdm(range(n_iter)):          
                if len(perturbation.keys())>0:
                    for gene in perturbation.keys():
                        perturbed_matrix.loc[:, gene] = perturbation[gene]

                perturbed_matrix_t = torch.tensor(perturbed_matrix.values).to(torch.float).to(self.opt.device)
                perturbed_matrix_t = perturbed_matrix_t[:, vae.TFs_idx]                      
                z_tf_perturbed = vae.inference_rna(perturbed_matrix_t.view(perturbed_matrix_t.size(0), -1, 1))['mean'].to(torch.float)
                Wrna_pert_ct = z_tf_perturbed @ Wrna_pert

                logFC = (Wrna_pert_ct - Wrna_ct)
                logFC = vae.generative_rna(logFC.to(torch.float))
                logFC = logFC['x_rec']
                # logFC = torch.clamp(logFC, -clip_val, clip_val)
                perturbed_matrix = original_matrix + logFC.cpu().numpy() # Apply fold change compute new expression matrix
                mask = perturbed_matrix!=original_matrix
                perturbed_matrix[mask] = np.clip(perturbed_matrix[mask], 0, orig_mtx_p99)

                if keep_intermediate:
                    perturbation_over_iter[str(i + 2)] = perturbed_matrix
                    fcs[str(i+2)] = logFC.cpu().numpy()
            if keep_intermediate:
                return  perturbation_over_iter, fcs
            else:
                # return z_tf_perturbed, Wrna_pert
                return perturbed_matrix, logFC.cpu().numpy()

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
        dataloader, TFs_idx, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index, r2g_dist_coo, _, _ = self.init_data()

        if self.opt.use_best==True:
            model_dict_tf2r_func_enc_path = self.opt.save_name + 'best_model_tf2r_encoder.pth'
            model_dict_tf2r_path = self.opt.save_name + 'best_model_tf2r.pth'
            vae_model_path = self.opt.save_name + 'best_model.pth'
        else:
            model_dict_tf2r_func_enc_path = self.opt.save_name + 'model_tf2r_encoder.pth'
            model_dict_tf2r_path = self.opt.save_name + 'model_tf2r.pth'
            vae_model_path = self.opt.save_name + 'model.pth'        

        # Initialize TF2rNet model
        # Load tf2r fuctional encoder
        tf2rNet_func_encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=5, dropout_rate = 0.).to(self.opt.device)
        model_dict_tf2r_func_enc = torch.load(model_dict_tf2r_func_enc_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
        # Load tf2r contex head
        tf2rNet = MotifNet(self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, dev=self.opt.device).float().to(self.opt.device)
        model_dict_tf2r = torch.load(model_dict_tf2r_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet.load_state_dict(model_dict_tf2r)   
        print("loaded weights for TF2rNet")

        # Initialize VAE
        vae = VAE(TFs_idx, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index, r2g_dist_coo, 1, self.opt.n_hidden, opt=self.opt).float().to(self.opt.device)
        vae.load_state_dict(torch.load(vae_model_path,  map_location=torch.device(self.opt.device))['model_state_dict'])
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
            rna_ppi_l = []
            for _, data_batch in tqdm(enumerate(dataloader['dataloader'], 0), unit="batch", total=len(dataloader['dataloader'])):
                # VAE forward pass
                if self.opt.batch_key is not None:
                    inputs_rna, inputs_atac, inputs_batch, _ = data_batch
                    inputs_batch = Variable(inputs_batch.type(Tensor))
                else:
                    inputs_rna, inputs_atac, _ = data_batch
                    inputs_batch = None
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                out_gen_rna, out_gen_atac, out_inf_rna, enh_act, z_rna, x_rna_ppi = vae.predict(
                        inputs_rna, adj_E1=adj_E1, adj_E2=adj_E2)

                z_rna_l += [z_rna.cpu().numpy()]
                rna_ppi_l += [x_rna_ppi.cpu().numpy()]
                z_tf_mu_l += [out_inf_rna['mean'].cpu().numpy()]
                z_tf_var_l += [out_inf_rna['logvar'].cpu().numpy()]
                z_rna_atac_l += [enh_act.cpu().numpy()]
                dec_rna_l += [out_gen_rna['x_rec'].cpu().numpy()]
                dec_atac_l += [out_gen_atac['x_rec'].cpu().numpy()]

            z_rna_l = np.vstack(z_rna_l)
            rna_ppi_l = np.vstack(rna_ppi_l)
            z_rna_atac_l = np.vstack(z_rna_atac_l)
            z_tf_mu_l = np.vstack(z_tf_mu_l)
            z_tf_var_l = np.vstack(z_tf_var_l)
            dec_rna_l = np.vstack(dec_rna_l)
            dec_atac_l = np.vstack(dec_atac_l)
            np.save(self.opt.save_name + 'z_rna.npy', z_rna_l)
            np.save(self.opt.save_name + 'rna_ppi.npy', rna_ppi_l)
            np.save(self.opt.save_name + 'z_tf_reg.npy', z_tf_mu_l)
            np.save(self.opt.save_name + 'z_tf_var.npy', z_tf_var_l)
            np.save(self.opt.save_name + 'z_rna_atac.npy', z_rna_atac_l)
            np.save(self.opt.save_name + 'y_rna.npy', dec_rna_l)
            np.save(self.opt.save_name + 'y_atac.npy', dec_atac_l)

    def to_latent_h5(self, adj_E1, adj_E2):
        """ Function for saving model embeddings.
            
            Params
            ------
            vae_model_path: VAE saved model path
            tf2r_model_path: TF2rNet saved model path
            tf2r_func_enc_model_path: TF2rNet functional encoder saved model path. If None, TF2rNet will be used without functional encoder.
        """
        if self.opt.device == 'cuda':
            Tensor = torch.cuda.FloatTensor
        elif self.opt.device == 'cpu':
            Tensor = torch.FloatTensor

        ### Initialize dataloaders
        self.opt.train = False
        dataloader, TFs_idx, r2g_dist_coo, _, _ = self.init_data()

        if self.opt.use_best:
            model_dict_tf2r_func_enc_path = self.opt.save_name + 'best_model_tf2r_encoder.pth'
            model_dict_tf2r_path = self.opt.save_name + 'best_model_tf2r.pth'
            vae_model_path = self.opt.save_name + 'best_model.pth'
        else:
            model_dict_tf2r_func_enc_path = self.opt.save_name + 'model_tf2r_encoder.pth'
            model_dict_tf2r_path = self.opt.save_name + 'model_tf2r.pth'
            vae_model_path = self.opt.save_name + 'model.pth'        

        # Initialize TF2rNet model
        # Load tf2r functional encoder
        tf2rNet_func_encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=5, dropout_rate=0.1).to(self.opt.device)
        model_dict_tf2r_func_enc = torch.load(model_dict_tf2r_func_enc_path, map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
        # Load tf2r context head
        tf2rNet = MotifNet(self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, dev=self.opt.device).float().to(self.opt.device)
        model_dict_tf2r = torch.load(model_dict_tf2r_path, map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet.load_state_dict(model_dict_tf2r)   
        print("Loaded weights for TF2rNet")

        # Initialize VAE
        vae = VAE(TFs_idx, r2g_dist_coo, 1, self.opt.n_hidden, opt=self.opt).float().to(self.opt.device)
        vae.load_state_dict(torch.load(vae_model_path, map_location=torch.device(self.opt.device))['model_state_dict'])
        vae.adj_E2 = nn.Parameter(adj_E2)

        # Create HDF5 file and datasets
        with h5py.File(self.opt.save_name + 'latent_data.h5', 'w') as h5f:
            # Assume the total number of samples can be determined (e.g., len(dataloader['dataloader'].dataset))
            total_samples = len(dataloader['dataloader'].dataset)
            h5f.create_dataset('z_rna', (total_samples, self.n_genes), dtype='float32')
            h5f.create_dataset('z_tf_mu', (total_samples, len(TFs_idx)), dtype='float32')
            h5f.create_dataset('z_tf_var', (total_samples, len(TFs_idx)), dtype='float32')
            h5f.create_dataset('z_rna_atac', (total_samples, self.n_regions), dtype='float32')
            h5f.create_dataset('dec_rna', (total_samples, self.n_genes), dtype='float32')
            h5f.create_dataset('dec_atac', (total_samples, self.n_regions), dtype='float32')

            start_idx = 0

            with torch.no_grad(): 
                vae.eval()
                print("VAE forward...")
                for _, data_batch in tqdm(enumerate(dataloader['dataloader'], 0), unit="batch", total=len(dataloader['dataloader'])):
                    # VAE forward pass
                    if self.opt.batch_key is not None:
                        inputs_rna, inputs_atac, inputs_batch, _ = data_batch
                        inputs_batch = Variable(inputs_batch.type(Tensor))
                    else:
                        inputs_rna, inputs_atac, _ = data_batch
                        inputs_batch = None
                    inputs_rna = Variable(inputs_rna.type(Tensor))
                    inputs_atac = Variable(inputs_atac.type(Tensor))

                    out_gen_rna, out_gen_atac, out_inf_rna, enh_act, z_rna = vae.predict(
                        inputs_rna, adj_E1=adj_E1, adj_E2=adj_E2)

                    end_idx = start_idx + inputs_rna.size(0)
                    
                    h5f['z_rna'][start_idx:end_idx] = z_rna.cpu().numpy()
                    h5f['z_tf_mu'][start_idx:end_idx] = out_inf_rna['mean'].cpu().numpy()
                    h5f['z_tf_var'][start_idx:end_idx] = out_inf_rna['logvar'].cpu().numpy()
                    h5f['z_rna_atac'][start_idx:end_idx] = enh_act.cpu().numpy()
                    h5f['dec_rna'][start_idx:end_idx] = out_gen_rna['x_rec'].cpu().numpy()
                    h5f['dec_atac'][start_idx:end_idx] = out_gen_atac['x_rec'].cpu().numpy()
                    
                    start_idx = end_idx

    def finetune_r2g_test(self):
        if self.opt.device=='cuda:0':
            Tensor = torch.cuda.FloatTensor
        elif self.opt.device=='cpu':
            Tensor = torch.FloatTensor

        # Initialize Tensorboard logger
        writer = SummaryWriter(self.opt.logs + '/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        ### Initialize dataloaderss
        self.opt.train = False
        test_dataloader, _, genes_idx, ppi_tfs_idx, ppi_genes_idx,  ppi_edge_index, r2g_dist_coo, test_seq_dataloader, _ = self.init_data(test=True)
        train_dataloader, TFs_idx, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index, r2g_dist_coo, train_seq_dataloader, train_seq_dataloader_shuffle = self.init_data(train=True)        


        if self.opt.use_best==True:
            model_dict_tf2r_func_enc_path = self.opt.save_name + 'best_model_tf2r_encoder.pth'
            model_dict_tf2r_path = self.opt.save_name + 'best_model_tf2r.pth'
            vae_model_path = self.opt.save_name + 'best_model.pth'
        else:
            model_dict_tf2r_func_enc_path = self.opt.save_name + 'model_tf2r_encoder.pth'
            model_dict_tf2r_path = self.opt.save_name + 'model_tf2r.pth'
            vae_model_path = self.opt.save_name + 'model.pth'

        # Load tf2r fuctional encoder
        tf2rNet_func_encoder = Enformer.from_pretrained('EleutherAI/enformer-official-rough', target_length=5, dropout_rate = 0.1).to(self.opt.device)
        model_dict_tf2r_func_enc = torch.load(model_dict_tf2r_func_enc_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet_func_encoder.load_state_dict(model_dict_tf2r_func_enc)
        # Load tf2r contex head
        tf2rNet = MotifNet(self.TFs, self.opt.TF2rNet_bottleneck_size, emb_len=self.opt.emb_len, dev=self.opt.device).float().to(self.opt.device)
        model_dict_tf2r = torch.load(model_dict_tf2r_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        tf2rNet.load_state_dict(model_dict_tf2r)   
        print("loaded weights for TF2rNet")

        # Initialize VAE
        vae = VAE(TFs_idx, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index,  r2g_dist_coo, 1, self.opt.n_hidden, opt=self.opt).float().to(self.opt.device)
        vae.PPInet = vae.PPInet.to(self.opt.device1)
        # Exclude 'adj_E2' from the state dict
        state_dict = torch.load(vae_model_path,  map_location=torch.device(self.opt.device))['model_state_dict']
        print("Best training epoch: ", torch.load(vae_model_path)['epoch'])
        if 'adj_E2' in state_dict:
            del state_dict['adj_E2']
        if 'batch_layer_rna' in state_dict:
            del state_dict['batch_layer_rna']            
        vae.load_state_dict(state_dict, strict=False)
        vae.adj_E2 = nn.Parameter(torch.zeros(r2g_dist_coo.size, device=self.opt.device, requires_grad=True) + vae.eps)
        if self.opt.batch_key is not None:
            vae.batch_layer_rna = nn.Sequential(nn.Linear(self.opt.batch_id_dim + self.n_genes, 128),
                                                    nn.Tanh(),
                                                    nn.Linear(128, self.n_genes))
        print("loaded weights for vae")
        
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
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.opt.lr_patience, verbose=True)

        # Training
        adj_E1 = None
        best_loss = float('inf')
        for epoch in range(self.opt.n_epochs):
            vae.train()
            tf2rNet.eval()
            tf2rNet_func_encoder.eval()

            loss_l, loss_rna_l, loss_atac_l, loss_gauss_rna_l, E2_sparse_l, f1_l = [], [], [], [], [], []
            for i, data_batch in tqdm(enumerate(train_dataloader['dataloader'], 0), unit="batch", total=len(train_dataloader['dataloader'])):
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

                # TF2rNet forward pass
                if adj_E1 is None:
                    with torch.no_grad():
                        tf_pred_l = []
                        for j, (X, _) in tqdm(enumerate(train_seq_dataloader, 0)):
                            tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device), return_only_embeddings=True)
                            tf_pred = tf2rNet(emb=tf_pred)
                            tf_pred_l.append(tf_pred)
                        adj_E1 = torch.cat(tf_pred_l)
                    del tf_pred_l, tf_pred

                # VAE forward pass
                if self.opt.batch_key is not None:
                    inputs_rna, inputs_atac, inputs_batch, _ = data_batch
                    inputs_batch = Variable(inputs_batch.type(Tensor))
                else:
                    inputs_rna, inputs_atac, _ = data_batch
                    inputs_batch = None
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                loss_rec_rna, loss_rec_atac, loss_rec_ppi, loss_gauss_rna, E2_sparse, _,  _, _, _, _, _, f1_atac = vae(
                    inputs_rna,
                    inputs_atac,
                    inputs_batch=inputs_batch,
                    dropout_mask_rna=self.opt.dropout_loss,
                    dropout_mask_atac=self.opt.dropout_loss,
                    adj_E1=adj_E1)
                
                # Compute sparse loss
                # E2_sparse = (vae.adj_E2.abs() * vae.r2g_dist.values()).mean() * self.opt.alpha

                loss = loss_rec_rna + E2_sparse
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(True)
                
                # Tensorboard logs
                loss_l.append(loss.detach().item())
                loss_rna_l.append(loss_rec_rna.detach().item())
                loss_atac_l.append(loss_rec_atac.detach().item())
                loss_gauss_rna_l.append(loss_gauss_rna.detach().item())
                E2_sparse_l.append(E2_sparse.detach().item())
                f1_l.append(f1_atac.detach().item())

            writer.add_scalar('Loss/total', np.mean(loss_l), epoch)
            writer.add_scalar('Loss/rec_rna', np.mean(loss_rna_l), epoch)
            writer.add_scalar('Loss/rec_atac', np.mean(loss_atac_l), epoch)
            writer.add_scalar('Loss/kl_rna', np.mean(loss_gauss_rna_l), epoch)
            writer.add_scalar('Loss/l1_E2', np.mean(E2_sparse_l), epoch)
            writer.add_scalar('Loss/f1_atac', np.mean(f1_l), epoch)
            print('epoch:', epoch)
            scheduler.step(np.mean(loss_l))
            print('Updating lr to: ', scheduler.get_last_lr())

            with torch.no_grad():
                vae.eval()
                loss_l, loss_rna_l, E2_sparse_l = [], [], []
                for i, data_batch in tqdm(enumerate(test_dataloader['dataloader'], 0), unit="batch", total=len(test_dataloader['dataloader'])):
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                    # VAE forward pass
                    if self.opt.batch_key is not None:
                        inputs_rna, inputs_atac, inputs_batch, _ = data_batch
                        inputs_batch = Variable(inputs_batch.type(Tensor))
                    else:
                        inputs_rna, inputs_atac, _ = data_batch
                        inputs_batch = None

                    inputs_rna = Variable(inputs_rna.type(Tensor))
                    inputs_atac = Variable(inputs_atac.type(Tensor))

                    loss_rec_rna, loss_rec_atac, loss_rec_ppi, loss_gauss_rna, E2_sparse, _,  _, _, _, _, _, f1_atac = vae(
                        inputs_rna,
                        inputs_atac,
                        inputs_batch=inputs_batch,
                        dropout_mask_rna=self.opt.dropout_loss,
                        dropout_mask_atac=self.opt.dropout_loss,
                        adj_E1=adj_E1)
                    
                    # Compute sparse loss
                    # E2_sparse = (vae.adj_E2.abs() * vae.r2g_dist.values()).mean() * self.opt.alpha

                    loss_l.append((loss_rec_rna + E2_sparse).detach().item())
                    loss_rna_l.append(loss_rec_rna.detach().item())
                    E2_sparse_l.append(E2_sparse.detach().item())
                    
                # Tensorboard logs
                writer.add_scalar('Test/loss_total', np.mean(loss_l), epoch)
                writer.add_scalar('Test/rec_rna', np.mean(loss_rna_l), epoch)
                writer.add_scalar('Test/l1_E2', np.mean(E2_sparse_l), epoch)
                if np.mean(loss_l) <= best_loss:
                    best_loss = np.mean(loss_l)
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
        test_dataloader, _, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index, r2g_dist_coo, test_seq_dataloader, _ = self.init_data(test=True)
        train_dataloader, TFs_idx, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index, r2g_dist_coo, train_seq_dataloader, train_seq_dataloader_shuffle = self.init_data(train=True)        

        if self.opt.device=='cuda:0':
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
        if self.opt.load_model is not None:
            vae = VAE(TFs_idx, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index, r2g_dist_coo, 1, self.opt.n_hidden, opt=self.opt).float()
            if os.path.exists(self.opt.load_model + 'model.pth'):
                vae_d = torch.load(self.opt.load_model + 'model.pth')['model_state_dict']
                vae.load_state_dict(vae_d)
                vae = vae.to(self.opt.device)
                vae.PPInet = vae.PPInet.to(self.opt.device1)
                print("loaded weights for vae")
        else:
            vae = VAE(TFs_idx, genes_idx, ppi_tfs_idx, ppi_genes_idx, ppi_edge_index, r2g_dist_coo, 1, self.opt.n_hidden, opt=self.opt).float().to(self.opt.device)
            vae.PPInet = vae.PPInet.to(self.opt.device1)

        # Initialize optimizers
        optimizer = optim.Adam([{'params':vae.parameters(), 'lr':self.opt.lr}, {'params':tf2rNet_func_encoder.parameters(), 'lr':self.opt.lr}, {'params':tf2rNet.parameters(), 'lr':self.opt.lr}])
        # if self.opt.load_model is not None:
        #     state_dict = torch.load(self.opt.load_model + 'model.pth',   map_location=torch.device(self.opt.device))
        #     if 'optimizer_state_dict' in state_dict:
        #         optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        #         print("loaded optimizer state")

        # Initialize early stopping
        adj_E1 = None
        adj_E1_test = None
        adj_E1_old = None
        best_loss =  float('inf')
        train_seq_dataloader_shuffle_iterator = iter(train_seq_dataloader_shuffle) # Initialize iterator for sequence dataloader
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.opt.lr_patience, verbose=True)
        cos_sim_train = torch.zeros((4,)).to(self.opt.device)
        cos_sim_test = torch.zeros((4,)).to(self.opt.device)
        for epoch in range(self.opt.n_epochs):
            vae.train()
            tf2rNet.eval()
            tf2rNet_func_encoder.eval()
            # if (epoch==0) | (epoch >= self.opt.warmup_vae):
            if (epoch==0):
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

            if epoch < self.opt.warmup_vae:
                # if warmup vae, freeze layers of tf2r
                for param in tf2rNet.parameters():
                    param.requires_grad = False
                for param in tf2rNet_func_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in tf2rNet.parameters():
                    param.requires_grad = True
                for param in tf2rNet_func_encoder.parameters():
                    param.requires_grad = True
            
            loss_l, loss_rec_rna_l, loss_rec_ppi_l, loss_rec_atac_l, loss_gauss_rna_l,E1_sparse_l, E2_sparse_l, cos_loss_l, f1_score_l = [], [], [], [], [], [], [], [], []
            for i, data_batch in tqdm(enumerate(train_dataloader['dataloader'], 0), unit="batch", total=len(train_dataloader['dataloader'])):
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True

                # TF2rNet forward pass
                adj_E1 = adj_E1_old.clone()
                

                if epoch >= self.opt.warmup_vae:
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
                    del tf_pred #, seq_data_batch_idx #, adj_E1_old

                # VAE forward pass
                if self.opt.batch_key is not None:
                    inputs_rna, inputs_atac, inputs_batch, _ = data_batch
                    inputs_batch = Variable(inputs_batch.type(Tensor))
                else:
                    inputs_rna, inputs_atac, _ = data_batch
                    inputs_batch = None
                inputs_rna = Variable(inputs_rna.type(Tensor))
                inputs_atac = Variable(inputs_atac.type(Tensor))

                loss_rec_rna, loss_rec_atac, loss_rec_ppi, loss_gauss_rna, E2_sparse, _,  _, _, _, _, _, f1_atac = vae(
                    inputs_rna,
                    inputs_atac,
                    inputs_batch=inputs_batch,
                    dropout_mask_rna=self.opt.dropout_loss,
                    dropout_mask_atac=self.opt.dropout_loss, 
                    adj_E1=adj_E1,
                    )
                
                # Compute sparse loss
                # with torch.no_grad():
                # E1_sparse = (adj_E1[seq_data_batch_idx, :].abs().mean()) * self.opt.alpha
                with torch.no_grad():
                    cos_sim_train = cos_sim_train.clone()
                E1_sparse = (self._cosine_similarity(adj_E1, self.prior_E1, seq_data_batch_idx, cos_sims=cos_sim_train)) * self.opt.alpha
                # E1_sparse = E1_sparse + (self.sign_penalty(adj_E1[seq_data_batch_idx, :])) * self.opt.alpha
                with torch.no_grad():
                    cos_loss = self.cosine_similarity(adj_E1[seq_data_batch_idx, :]) 
                # E2_sparse = (vae.adj_E2.abs() * vae.r2g_dist.values()).mean() * self.opt.alpha

                if epoch >= self.opt.warmup_vae:
                    loss = loss_rec_rna + loss_gauss_rna + loss_rec_atac + E1_sparse + E2_sparse # + cos_loss
                else:
                    loss = loss_rec_rna + loss_gauss_rna + E2_sparse

                loss.backward()
                optimizer.step()
                # Reset optimizers
                optimizer.zero_grad(True)         

                loss_l.append(loss.detach().item())
                loss_rec_rna_l.append(loss_rec_rna.detach().item())
                loss_rec_ppi_l.append(loss_rec_ppi.detach().item())
                loss_rec_atac_l.append(loss_rec_atac.detach().item())
                loss_gauss_rna_l.append(loss_gauss_rna.detach().item())
                E1_sparse_l.append(E1_sparse.detach().item())
                E2_sparse_l.append(E2_sparse.detach().item())
                cos_loss_l.append(cos_loss.detach().item())
                f1_score_l.append(f1_atac.detach().item())            

            # Tensorboard logs
            writer.add_scalar('Loss/total', np.mean(loss_l), epoch)
            writer.add_scalar('Loss/rec_rna', np.mean(loss_rec_rna_l), epoch)
            writer.add_scalar('Loss/rec_ppi', np.mean(loss_rec_ppi_l), epoch)
            writer.add_scalar('Loss/rec_atac', np.mean(loss_rec_atac_l), epoch)
            writer.add_scalar('Loss/kl_rna', np.mean(loss_gauss_rna_l), epoch)
            writer.add_scalar('Loss/l1_E1', np.mean(E1_sparse_l), epoch)
            writer.add_scalar('Loss/l1_E2', np.mean(E2_sparse_l), epoch)
            writer.add_scalar('Loss/cos_loss', np.mean(cos_loss_l), epoch)
            writer.add_scalar('Loss/f1_atac', np.mean(f1_score_l), epoch)

            del loss_l, loss_rec_rna_l, loss_rec_ppi_l, loss_rec_atac_l, loss_gauss_rna_l, E1_sparse_l, E2_sparse_l, cos_loss_l, f1_score_l

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

            print('epoch:', epoch)
            # Evaluate test set
            with torch.no_grad():
                vae.eval()
                tf2rNet.eval()
                tf2rNet_func_encoder.eval()

                tf_pred_l = []
                for j, (X, seq_data_batch_idx) in tqdm(enumerate(test_seq_dataloader, 0)):
                    tf_pred = tf2rNet_func_encoder(X[0].to(self.opt.device), return_only_embeddings=True)
                    tf_pred = tf2rNet(emb=tf_pred)
                    tf_pred_l.append(tf_pred)
                adj_E1_test = torch.cat(tf_pred_l)
                del tf_pred_l, tf_pred

                
                loss_all, f1_score, rec_rna, rec_atac, rec_ppi, loss_kl_rna, loss_sparse = [], [], [], [], [], [], []
                for i, data_batch in tqdm(enumerate(test_dataloader['dataloader'], 0), unit="batch", total=len(test_dataloader['dataloader'])):
                    if self.opt.batch_key is not None:
                        inputs_rna, inputs_atac, inputs_batch, _ = data_batch
                        inputs_batch = Variable(inputs_batch.type(Tensor))
                    else:
                        inputs_rna, inputs_atac, _ = data_batch
                        inputs_batch = None
                    inputs_rna = Variable(inputs_rna.type(Tensor))
                    inputs_atac = Variable(inputs_atac.type(Tensor))

                    _, out_gen_atac, out_inf_rna, _, _, _ = vae.predict(
                       inputs_rna, inputs_batch=inputs_batch, adj_E1=adj_E1_test)
                    out_gen_rna, _, _, _, _, x_rna_ppi= vae.predict(
                       inputs_rna, inputs_batch=inputs_batch, adj_E1=adj_E1, adj_E2=vae.adj_E2)
               
                    if self.opt.bin_acc==True:
                        f1 = F1Score(task='binary',num_classes=1).to(self.opt.device)
                        mask = ~(inputs_atac == -1).all(dim=1)
                        if inputs_atac[mask].shape[0] !=0:
                            f1_atac = f1(out_gen_atac['x_rec'][mask].ravel(), inputs_atac[mask].int().ravel())
                        del mask    
                    else:
                        f1_atac = torch.Tensor([0]).to(self.opt.device)  
                    
                    loss_rec_rna = vae.losses.reconstruction_loss(inputs_rna[:, genes_idx], out_gen_rna['x_rec'], self.opt.dropout_loss, rec_type=self.opt.loss_rna) * self.opt.rna_tau
                    loss_rec_atac = vae.losses.reconstruction_loss(inputs_atac, out_gen_atac['x_rec'], False, rec_type=self.opt.loss_atac) * self.opt.atac_tau
                    loss_rec_ppi = vae.losses.reconstruction_loss(inputs_rna[:, TFs_idx], x_rna_ppi, False, rec_type='mse')
                    loss_gauss_rna = vae.losses.gaussian_loss(out_inf_rna['mean'], out_inf_rna['logvar']) * self.opt.beta

                    # sparse_loss = (adj_E1_test.abs().mean()) * self.opt.alpha
                    cos_sim_test = cos_sim_test.clone()
                    sparse_loss = (self._cosine_similarity(adj_E1_test, self.prior_E1_test, cos_sims=cos_sim_test)) * self.opt.alpha
                    # sparse_loss = sparse_loss + (self.sign_penalty(adj_E1_test)) * self.opt.alpha
                    loss = loss_rec_rna + loss_rec_atac + loss_gauss_rna #+ loss_rec_ppi
                            
                    rec_rna.append(loss_rec_rna.item())
                    rec_ppi.append(loss_rec_ppi.item())
                    rec_atac.append(loss_rec_atac.item())
                    loss_all.append(loss.detach().item())
                    loss_kl_rna.append(loss_gauss_rna.item())
                    loss_sparse.append(sparse_loss.detach().item())
                    f1_score.append(f1_atac.detach().item())

                if np.mean(rec_atac) <= best_loss:
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
                    best_loss = np.mean(rec_atac)

                # Tensorboard logs
                writer.add_scalar('Test/loss_total', np.mean(loss_all), epoch)
                writer.add_scalar('Test/rec_atac', np.mean(rec_atac), epoch)
                writer.add_scalar('Test/rec_rna', np.mean(rec_rna), epoch)
                writer.add_scalar('Test/rec_ppi', np.mean(rec_ppi), epoch)
                writer.add_scalar('Test/kl_rna', np.mean(loss_kl_rna), epoch)
                writer.add_scalar('Test/l1_A', np.mean(loss_sparse), epoch)
                writer.add_scalar('Test/f1_atac', np.mean(f1_score), epoch)

                # scheduler.step(np.mean(rec_atac))
                # print('Updating lr to: ', scheduler.get_last_lr())
                # if epoch >= self.opt.warmup_vae:
                #     early_stopping(sparse_loss)
                del loss, f1_score, loss_all, rec_atac, loss_kl_rna, loss_sparse, loss_rec_rna, loss_rec_ppi,loss_rec_atac, loss_gauss_rna, sparse_loss, adj_E1_test, rec_rna
                del inputs_rna, inputs_atac, out_gen_atac, out_gen_rna, out_inf_rna, E1_sparse, f1_atac, x_rna_ppi
