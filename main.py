import argparse
from src.deepSCENIC import deepSCENIC

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--warmup_vae', type=int, default=0, help='Number of warmup epochs for VAE')
parser.add_argument('--task', type=str, default='deepSCENIC',
                    help='Determine which task to run. Select from (deepSCENIC: train the whole model; pretrain: pretrain tf2r network; finetune_E2: finetune the E2 network on test data).')
parser.add_argument('--batch_size', type=int, default=64, help='VAE batch size (number of cells in batch).')
parser.add_argument('--seqs_batch_size', type=int, default=1000, help='The batch size used for TF2rNet (number of sequences in batch).')
parser.add_argument('--data_rna_file', type=str, help='The input scRNA-seq file (.h5ad).')
parser.add_argument('--data_atac_file', type=str, help='The input scATAC-seq file (.h5ad).')
parser.add_argument('--data_rna_file_train', type=str, help='The input scRNA-seq file (.h5ad) containing training data.')
parser.add_argument('--data_atac_file_train', type=str, help='The input scATAC-seq file (.h5ad) containing training data.')
parser.add_argument('--data_rna_file_test', type=str, help='The input scRNA-seq file (.h5ad) containing test data.')
parser.add_argument('--data_atac_file_test', type=str, help='The input scATAC-seq file (.h5ad) containing test data.')
parser.add_argument('--bin_acc', default=False, action='store_true', help='Binarize scATAC-seq data.')
parser.add_argument('--TF_file', type=str, help='List of trascription factors file.')
parser.add_argument('--tf2r_prior', default=None, type=str, help='TF to region prior matrix.')
parser.add_argument('--tf2r_prior_test', default=None, type=str, help='TF to region prior test matrix.')
parser.add_argument('--r2g_mask', type=str, help='Region to target gene mask.')
parser.add_argument('--Receptor2TF_df', type=str, help='Receptor to TF df.')
parser.add_argument('--TF2rNet_bottleneck_size', type=int, default=3072, help='TF2rNet input feature dimension.')
parser.add_argument('--fasta', type=str, help='Fasta file of reference genome.')
parser.add_argument('--seq_len', type=int, default=640, help='Sequence length for TF2rNet.')
parser.add_argument('--lr_patience', type=int, default=10, help='LR patience for decreasing learning rate.')
parser.add_argument('--emb_len', type=int, default=5, help='Embedding length for TF2rNet.')
parser.add_argument('--beta', type=float, default=1e-2, help='The loss coefficient for KL term (beta-VAE).')
parser.add_argument('--alpha', type=float, default=1e-2, help='Sparse loss coefficient.')
parser.add_argument('--gamma', type=float, default=1, help='Sparse loss coefficient.')
parser.add_argument('--atac_tau', type=float, default=1, help='ATAC loss coefficient.')
parser.add_argument('--rna_tau', type=float, default=1, help='RNA loss coefficient.')
parser.add_argument('--rec_tau', type=float, default=1, help='LR loss coefficient.')
parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate.')
parser.add_argument('--n_hidden', type=int, default=128, help='The Number of hidden neural used in MLP')
parser.add_argument('--load_model', type=str, default=None, help='Load pretrained model from dir path.')
parser.add_argument('--save_name', type=str, default='/tmp', help='Output directory.')
parser.add_argument('--logs', type=str, default='/tmp', help='Tensorboard log dir.')
parser.add_argument('--train', default=False, action='store_true', help='Specify if training.')
parser.add_argument('--disable_dropout_loss', default=False, action='store_true', help='Weather to use dropout loss for recostructing scRNA-seq.')
parser.add_argument('--use_best', default=False, action='store_true', help='Weather to use  best or last model epoch.')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--device1', type=str, default='cuda:1')
parser.add_argument('--loss_rna', type=str, default='mae')
parser.add_argument('--loss_atac', type=str, default='cos', help='TF2rNet loss function (mse: mean squared error; mae: mean absolute error; cos: cosine similarity).')
parser.add_argument('--batch_key', type=str, default=None, help='The key of batch in h5ad file.')
parser.add_argument('--ann_key', type=str, default=None, help='The key of cell type annotations in h5ad file used for balanced training.')
parser.add_argument('--balance_class', default=False, action='store_true', help='Weather to balance classes during training.')
parser.add_argument('--balance_dars', default=False, action='store_true', help='Weather to balance classes during training.')
opt = parser.parse_args()


if opt.disable_dropout_loss==True:
    opt.dropout_loss = False
else:
    opt.dropout_loss = True

if opt.task == 'deepSCENIC':
    print(opt)
    model = deepSCENIC(opt)
    model.train_model()
if opt.task == 'pretrain':
    print(opt)
    model = deepSCENIC(opt)
    model.pretrain()
if opt.task == 'finetune_E2':
    print(opt)
    model = deepSCENIC(opt)
    model.finetune_r2g_test()