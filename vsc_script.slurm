#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=dedicated_big_gpu
#SBATCH --time=168:00:00
#SBATCH --cluster=wice
#SBATCH -A lp_big_wice_gpu
#SBATCH --mem=160gb

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $SLURM_SUBMIT_DIR

# Activate conda environment with tensorflow and required libraries
source /user/leuven/340/vsc34097/.bash_profile

my_conda_initialize
conda activate DeepSEM

python -u /lustre1/project/stg_00002/lcb/gpartel/software/deepSCENIC/main.py \
        --task deepSCENIC \
        --n_epochs 1500 \
        --batch_size 16 \
        --TF2rNet_batch_size 256 \
        --data_rna_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/DPCL/raw_exprMat.h5ad \
        --data_atac_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/DPCL/fragment_matrix.h5ad  \
        --TF_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/DPCL/TFs.txt \
        --save_name /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/DPCL_2/ \
        --logs /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/DPCL/ \
        --r2g_mask /staging/leuven/stg_00002/lcb/gpartel/projects/deepSCENIC/data/DPCL/r2gdist.npz \
        --fasta /lustre1/project/stg_00002/lcb/gpartel/resources/genomes/hg38.fa \
        --beta 1e-2 \
        --bin_acc \
        --train \
        --test_size 0.2 \
        --lr 1e-3 \
        --ppms_file /staging/leuven/stg_00002/lcb/gpartel/projects/deepSCENIC/data/PPMs_public.pkl \
        ### Comments either of the following lines depending on the model you want to use            
        # If TF2rNet pretrained model
        # --n_it_acc 10 \
        # --seq_len 500 \
        # --emb_len 31 \
        # --TF2rNet_bottleneck_size 960 \    
        # --load_tf2rNet_model /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/DPCL/TF2rNET_func_enc_pretrain.pth
        # Otherwise use Enformer embeddings
        --seq_len 640 \
        --emb_len 5 \
        --TF2rNet_bottleneck_size 3072 \
        --enformer_embs_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/DPCL/enformer_embs.h5
