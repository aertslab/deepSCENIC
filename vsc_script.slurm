#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=dedicated_big_gpu
#SBATCH --time=72:00:00
#SBATCH --cluster=wice
#SBATCH -A lp_big_wice_gpu
#SBATCH --mem=160gb

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $SLURM_SUBMIT_DIR

# Activate conda environment with tensorflow and required libraries
source /user/leuven/340/vsc34097/.bash_profile

my_conda_initialize
conda activate DeepSEM

python -u /lustre1/project/stg_00002/lcb/gpartel/software/DeepSEM_multimodal/main.py --task deepSCENIC --batch_size 64 --data_rna_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/PBMC/data/deepSCENIC/raw_exprMat.h5ad --data_atac_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/PBMC/data/deepSCENIC/fragment_matrix.h5ad  --TF_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/PBMC/data/deepSCENIC/TFs.txt --save_name /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/DeepSEM_out_500bp_prior_5/ --logs /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/DeepSEM_out --r2g_mask /staging/leuven/stg_00002/lcb/gpartel/projects/deepSCENIC/data/PBMC/data/deepSCENIC/r2gdist.npz --n_epochs 300 --fasta /lustre1/project/stg_00002/lcb/gpartel/resources/genomes/hg38.fa --TF2rNet_batch_size 1000 --beta 1e-2 --alpha 1 --TF2rNet_bottleneck_size 128 --bin_acc --train --test_size 0.2 --atac_gamma 1 --lr 1e-4 --seq_len 500 --beta_motif 1e-2 --tf2r_df /staging/leuven/stg_00002/lcb/gpartel/projects/deepSCENIC/data/PBMC/data/deepSCENIC/TF2R_df.pkl
