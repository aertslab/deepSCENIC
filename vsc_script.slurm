#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=dedicated_big_gpu
#SBATCH --time=24:00:00
#SBATCH --cluster=wice
#SBATCH -A lp_big_wice_gpu
#SBATCH --mem=80gb

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
cd $SLURM_SUBMIT_DIR

# Activate conda environment with tensorflow and required libraries
source /user/leuven/340/vsc34097/.bash_profile

my_conda_initialize
conda activate DeepSEM

python -u /lustre1/project/stg_00002/lcb/gpartel/software/deepSCENIC/main.py \
        --task deepSCENIC \
        --seqs_batch_size 256 \
        --TF2rNet_bottleneck_size 3072 \
        --TF_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/TFs.txt \
        --batch_size 16 \
        --beta 0.01 \
        --alpha 0.01 \
        --gamma 1 \
        --data_atac_file_train /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/fragment_matrix_train.h5ad \
        --data_atac_file_test /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/fragment_matrix_test.h5ad \
        --data_rna_file_train /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/raw_exprMat_train.h5ad \
        --data_rna_file_test /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/raw_exprMat_train_eval.h5ad \
        --data_atac_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/fragment_matrix.h5ad \
        --data_rna_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/raw_exprMat.h5ad \
        --emb_len 5 \
        --fasta /lustre1/project/stg_00002/lcb/gpartel/resources/genomes/hg38.fa \
        --logs /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/MM_lines_Enformer/ \
        --lr 1e-4 \
        --n_epochs 1000 \
        --r2g_mask /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/r2gpenalty_train.npz \
        --save_name /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/MM_lines_Enformer3/ \
        --seq_len 640 \
        --disable_dropout_loss \

python -u /lustre1/project/stg_00002/lcb/gpartel/software/deepSCENIC/main.py \
        --task finetune_E2 \
        --seqs_batch_size 256 \
        --TF2rNet_bottleneck_size 3072 \
        --TF_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/TFs.txt \
        --batch_size 16 \
        --beta 0.01 \
        --alpha 0.01 \
        --gamma 1 \
        --data_atac_file_train /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/fragment_matrix_train.h5ad \
        --data_atac_file_test /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/fragment_matrix_test.h5ad \
        --data_rna_file_train /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/raw_exprMat_train.h5ad \
        --data_rna_file_test /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/raw_exprMat_test.h5ad \
        --data_atac_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/fragment_matrix.h5ad \
        --data_rna_file /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/raw_exprMat.h5ad \
        --emb_len 5 \
        --fasta /lustre1/project/stg_00002/lcb/gpartel/resources/genomes/hg38.fa \
        --logs /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/MM_lines_Enformer/ \
        --lr 1e-6 \
        --n_epochs 10000 \
        --r2g_mask /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/data/MM_lines/r2gpenalty_test.npz \
        --save_name /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/MM_lines_Enformer3/ \
        --seq_len 640 \
        --load_model /lustre1/project/stg_00002/lcb/gpartel/projects/deepSCENIC/results/MM_lines_Enformer3/ \
        --use_best \
        --disable_dropout_loss \