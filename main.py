import argparse
from src.deepSCENIC import deepSCENIC

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--warmup_vae', type=int, default=0, help='Number of iterations for accumulating gradients')
parser.add_argument('--task', type=str, default='deepSCENIC',
                    help='Determine which task to run. Select from (deepSCENIC: train the whole model; pretrain_tf2r: pretrain tf2r network)')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size used in the training process.')
parser.add_argument('--TF2rNet_batch_size', type=int, default=1000, help='The batch size used for TF2rNet inference.')
parser.add_argument('--data_rna_file', type=str, help='The input scRNA-seq file (.h5ad).')
parser.add_argument('--data_atac_file', type=str, help='The input scATAC-seq file (.h5ad).')
parser.add_argument('--data_rna_file_train', type=str, help='The input scRNA-seq file (.h5ad) containing training data.')
parser.add_argument('--data_atac_file_train', type=str, help='The input scATAC-seq file (.h5ad) containing training data.')
parser.add_argument('--data_rna_file_test', type=str, help='The input scRNA-seq file (.h5ad) containing test data.')
parser.add_argument('--data_atac_file_test', type=str, help='The input scATAC-seq file (.h5ad) containing test data.')
parser.add_argument('--bin_acc', default=False, action='store_true', help='Binarize scATAC-seq data.')
parser.add_argument('--TF_file', type=str, help='List of trascription factors file.')
parser.add_argument('--r2g_mask', type=str, help='Region to target gene mask.')
parser.add_argument('--TF2rNet_loss', type=str, default='mse', help='TF2rNet loss function for pretraining.')
parser.add_argument('--TF2rNet_bottleneck_size', type=int, default=3072, help='TF2rNet input feature dimension.')
parser.add_argument('--fasta', type=str, help='Fasta file of reference genome.')
parser.add_argument('--seq_len', type=int, default=500, help='Sequence length for TF2rNet.')
parser.add_argument('--emb_len', type=int, default=5, help='Embedding length for TF2rNet.')
parser.add_argument('--beta', type=float, default=1e-2, help='The loss coefficient for KL term (beta-VAE).')
parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate.')
parser.add_argument('--n_hidden', type=int, default=128, help='The Number of hidden neural used in MLP')
parser.add_argument('--load_model', type=str, default=None, help='Load pretrained model from dir path.')
parser.add_argument('--save_name', type=str, default='/tmp', help='Output directory.')
parser.add_argument('--logs', type=str, default='/tmp', help='Tensorboard log dir.')
parser.add_argument('--train', default=False, action='store_true', help='Specify if training.')
parser.add_argument('--disable_dropout_loss', default=False, action='store_true', help='Weather to use dropout loss for recostructing scRNA-seq.')
parser.add_argument('--ppms_file', type=str, help='Path to PPMs.')
parser.add_argument('--device', type=str, default='cuda')
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