import argparse
import torch

parser = argparse.ArgumentParser(description='SAGEMoX', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# setting
parser.add_argument('--name', type=str, default="Human_Lymph_Node_A1")
parser.add_argument(
    '--output_root',
    type=str,
    default=None,
    help='Root dir for visualize_trained_results outputs; files go under <root>/<name>/ (default: outputs)',
)
parser.add_argument(
    '--viz_point_size',
    type=float,
    default=6.0,
    help='matplotlib scatter size (s) for spatial_pred_vs_gt figures in visualize_trained_results',
)
parser.add_argument('--show', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=700, help='Stage 2: full training epochs (loop is epoch+1 steps)')
parser.add_argument(
    '--pretrain_epoch_stage0',
    type=int,
    default=31,
    help='Stage 0: GAE-only pretrain iteration count',
)
parser.add_argument(
    '--pretrain_epoch',
    type=int,
    default=500,
    help='Stage 1: GAE + fusion pretrain; loop runs pretrain_epoch+1 steps (same as before)',
)
parser.add_argument('--alpha_value', type=float, default=1)
parser.add_argument('--view', type=int, default=2, help='number of omics')

# parameters
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--spatial_k', type=int, default=10)
parser.add_argument('--motif_ratio', type=float, default=0.5,
                    help='Fusion ratio for high-order graph: A_f=(1-r)*A+r*M')
parser.add_argument('--pretrain_lr', type=float, default=1e-3)
parser.add_argument('--train_lr', type=float, default=1e-4)

# dimension of input and latent representations
parser.add_argument('--n_omics1', type=int, default=100)
parser.add_argument('--n_omics2', type=int, default=31)
parser.add_argument(
    '--z_dim',
    type=int,
    default=20,
    help='Dimension of GAE latent embedding z (bottleneck representation per spot)',
)

parser.add_argument(
    '--lambda_0',
    type=float,
    default=1,
    help='Weight for reconstruction loss (lambda_0 * L_rec)',
)
parser.add_argument('--lambda_1', type=float, default=1,
                    help='Weight for cross-modal correlation loss on aligned-guide branches (lambda_1 * L_corr)')
parser.add_argument(
    '--lambda_2',
    type=float,
    default=1,
    help='Weight for cross-omics fused-graph mismatch loss (lambda_2 * L_cross_fusion)',
)
parser.add_argument(
    '--cross_modal_corr_warmup_epochs',
    type=int,
    default=500,
    help='Stage-2 epochs to linearly ramp cross-modal correlation weight (lambda_1) from cross_modal_corr_warmup_start to 1.0; 0 = no ramp',
)
parser.add_argument(
    '--cross_modal_corr_warmup_start',
    type=float,
    default=0.0,
    help='Start multiplier for cross-modal correlation weight at epoch 0; used with cross_modal_corr_warmup_epochs>0',
)
parser.add_argument(
    '--tau',
    type=float,
    default=0.1,
    help='Edge-similarity threshold in cross-fusion mismatch loss (paper: tau)',
)
parser.add_argument('--shared_ratio', type=float, default=0.5,
                    help='Ratio of z_dim allocated to 对齐引导分支 (aligned_guide); remainder is 非对齐补充分支')
parser.add_argument('--device', type=str, default='cuda:0')

# GAE structure parameters
parser.add_argument('--gae_n_enc_1', type=int, default=800)
parser.add_argument('--gae_n_enc_2', type=int, default=400)
parser.add_argument('--gae_n_dec_1', type=int, default=400)
parser.add_argument('--gae_n_dec_2', type=int, default=800)
parser.add_argument('--dropout', type=float, default=0.3)

args = parser.parse_args()

# Robust device fallback: allow running on machines without CUDA builds.
if isinstance(args.device, str) and args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"

