import random
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
import opt
import torch.nn as nn
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn import cluster
from termcolor import colored


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def cross_modal_corr_lambda_schedule_scale(epoch, warmup_epochs, warmup_start=0.0):
    """
    Scale factor for lambda_1 (cross-modal correlation weight): linear ramp from warmup_start to 1.0.
    If warmup_epochs <= 0, returns 1.0 (full weight from epoch 0).
    """
    w = int(warmup_epochs)
    if w <= 0:
        return 1.0
    s0 = float(warmup_start)
    s0 = min(max(s0, 0.0), 1.0)
    t = min(1.0, (float(epoch) + 1.0) / float(w))
    return s0 + (1.0 - s0) * t


def reconstruction_loss(X, A_norm, Z_hat, A_hat):
    loss_w = F.mse_loss(Z_hat, torch.mm(A_norm, X))
    loss_a = F.mse_loss(A_hat, A_norm)
    loss_igae = loss_w + opt.args.alpha_value * loss_a
    return loss_igae


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def construct_graph(count, k=10, mode="connectivity"):
    countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()

    adj = (adj.T + adj) / 2
    adj_n = norm_adj(adj)

    return adj, adj_n


def _motif_cooccurrence_matrix(adj):
    """
    Build a 3-node motif co-occurrence matrix from an undirected adjacency matrix.
    Each entry counts the number of shared neighbors between two cells.
    """
    adj_bin = (adj > 0).astype(np.float32)
    np.fill_diagonal(adj_bin, 0.0)
    motif = adj_bin @ adj_bin
    np.fill_diagonal(motif, 0.0)

    max_val = motif.max()
    if max_val > 0:
        motif = motif / max_val
    return motif


def construct_high_order_graph(count, k=10, mode="connectivity", motif_ratio=0.5):
    """
    Construct high-order feature graph A_f^v = (1-r) * A_v + r * M_v.
    """
    countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj = (adj.T + adj) / 2

    motif = _motif_cooccurrence_matrix(adj)
    adj_high_order = (1.0 - motif_ratio) * adj + motif_ratio * motif
    adj_high_order = (adj_high_order + adj_high_order.T) / 2
    adj_high_order_n = norm_adj(adj_high_order)

    return adj_high_order, adj_high_order_n


def refine_adj_spatial(feature_graph, spatial_graph):
    """Keep spatial kNN edges only where the feature graph also has support (hard mask)."""
    feature_mask = (feature_graph > 0).astype(np.float32)
    spatial_graph_refine = spatial_graph * feature_mask
    spatial_graph_refine = (spatial_graph_refine + spatial_graph_refine.T) / 2
    return norm_adj(spatial_graph_refine)


def convert_to_tensor(arrays, device=opt.args.device):
    return [torch.FloatTensor(arr).to(device) for arr in arrays]


class WeightFusion(nn.Module):
    def __init__(self, num_views):
        super(WeightFusion, self).__init__()
        # Simple learnable weights for views, with softmax normalization in forward.
        # This is a global (not node-wise) fusion, shared across all nodes.
        self.weights = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)

    def forward(self, z_feature, z_spatial):
        """
        Weighted fusion of two embeddings:
          Z_intra = a * Z_feature + b * Z_spatial, where [a,b] = softmax(weights)
        """
        normalized_weights = F.softmax(self.weights, dim=0)  # (num_views,)
        fused_feature = normalized_weights[0] * z_feature + normalized_weights[1] * z_spatial
        return fused_feature, normalized_weights


def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


def post_proC(C, K, d=11, alpha=4):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def print_metrics(ARI, NMI, AMI, V_measure, Homo):
    metrics_str = (
        f"| ARI: {ARI:.4f} | NMI: {NMI:.4f} | AMI: {AMI:.4f} | "
        f"V-measure: {V_measure:.4f} | Homo: {Homo:.4f} |"
    )
    border = "=" * len(metrics_str)

    colored_border = colored(border, 'white', attrs=['bold'])
    colored_metrics = colored(metrics_str, 'red', attrs=['bold'])

    print(colored_border)
    print(colored_metrics)
    print(colored_border)
