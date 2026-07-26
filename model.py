import torch
import torch.nn as nn
import opt
from utils import he_init_weights, reconstruction_loss, WeightFusion
import torch.nn.functional as F
from Layers import build_gae


class SAGEMoX(nn.Module):
    def __init__(self):
        super().__init__()

        self.gae_feature_omics1 = build_gae(n_input=opt.args.n_omics1)
        self.gae_feature_omics2 = build_gae(n_input=opt.args.n_omics2)
        self.gae_spatial_omics1 = build_gae(n_input=opt.args.n_omics1)
        self.gae_spatial_omics2 = build_gae(n_input=opt.args.n_omics2)

        self.fusion_1 = WeightFusion(opt.args.view)
        self.fusion_2 = WeightFusion(opt.args.view)
        self.local_scale_r = nn.Parameter(torch.zeros(1))
        # A_mix = alpha * A_f + (1 - alpha) * A_s for global-local diffusion (alpha = sigmoid(logit)).
        self.feature_spatial_mix_logit = nn.Parameter(torch.zeros(1))
        self.fusion_balance_omics1 = nn.Sequential(
            nn.Linear(opt.args.z_dim * 2, opt.args.z_dim),
            nn.ReLU(),
            nn.Linear(opt.args.z_dim, 1),
        )
        self.fusion_balance_omics2 = nn.Sequential(
            nn.Linear(opt.args.z_dim * 2, opt.args.z_dim),
            nn.ReLU(),
            nn.Linear(opt.args.z_dim, 1),
        )
        self.diffusion_hop_logits_omics1 = nn.Parameter(torch.zeros(2))
        self.diffusion_hop_logits_omics2 = nn.Parameter(torch.zeros(2))
        init_supplement = 0.5
        init_supplement = min(max(init_supplement, 1e-6), 1.0 - 1e-6)
        init_logit = torch.log(torch.tensor(init_supplement / (1.0 - init_supplement), dtype=torch.float32))
        self.nonaligned_supplement_gate_logit_omics1 = nn.Parameter(init_logit.clone())
        self.nonaligned_supplement_gate_logit_omics2 = nn.Parameter(init_logit.clone())

        self.apply(he_init_weights)
        self.beta_1, self.beta_2 = self.compute_weights(opt.args.n_omics1, opt.args.n_omics2)
        if opt.args.z_dim < 2:
            raise ValueError("z_dim must be at least 2 for aligned-guide / nonaligned-supplement decomposition.")
        self.aligned_guide_dim = max(1, min(opt.args.z_dim - 1, int(opt.args.z_dim * opt.args.shared_ratio)))
        self.nonaligned_supplement_dim = opt.args.z_dim - self.aligned_guide_dim

        self.aligned_guide_proj = nn.Linear(opt.args.z_dim, self.aligned_guide_dim)
        self.nonaligned_supplement_proj = nn.Linear(opt.args.z_dim, self.nonaligned_supplement_dim)
        he_init_weights(self.aligned_guide_proj)
        he_init_weights(self.nonaligned_supplement_proj)

        self.s = nn.Sigmoid()

    def forward(
        self,
        omics_1,
        omics_2,
        adj_feature_omics1,
        adj_feature_omics2,
        adj_spatial_omics1,
        adj_spatial_omics2,
        stage=0,
    ):
        z_omics1_feature, adj_omics1_feature_hat = self.gae_feature_omics1.encoder(
            omics_1, adj_feature_omics1
        )
        z_omics2_feature, adj_omics2_feature_hat = self.gae_feature_omics2.encoder(
            omics_2, adj_feature_omics2
        )
        z_omics1_spatial, adj_omics1_spatial_hat = self.gae_spatial_omics1.encoder(
            omics_1, adj_spatial_omics1
        )
        z_omics2_spatial, adj_omics2_spatial_hat = self.gae_spatial_omics2.encoder(
            omics_2, adj_spatial_omics2
        )

        if stage == 0:
            # Stage 0: GAE-only pretrain (no fusion / global-local / branch split).
            return self._reconstruction_loss_only(
                omics_1,
                omics_2,
                adj_feature_omics1,
                adj_feature_omics2,
                adj_spatial_omics1,
                adj_spatial_omics2,
                z_omics1_feature,
                adj_omics1_feature_hat,
                z_omics2_feature,
                adj_omics2_feature_hat,
                z_omics1_spatial,
                adj_omics1_spatial_hat,
                z_omics2_spatial,
                adj_omics2_spatial_hat,
            )

        # Stage 1 & 2 shared: feature-spatial fusion, global-local on A_mix, aligned/supplement split.
        H_omics1_base, _ = self.fusion_1(z_omics1_feature, z_omics1_spatial)
        H_omics2_base, _ = self.fusion_2(z_omics2_feature, z_omics2_spatial)
        H_omics1 = self._global_local_fusion(
            H_omics1_base,
            adj_feature_omics1,
            adj_spatial_omics1,
            self.diffusion_hop_logits_omics1,
            self.fusion_balance_omics1,
        )
        H_omics2 = self._global_local_fusion(
            H_omics2_base,
            adj_feature_omics2,
            adj_spatial_omics2,
            self.diffusion_hop_logits_omics2,
            self.fusion_balance_omics2,
        )

        z_aligned_guide_omics1, z_nonaligned_supplement_omics1 = (
            self._split_aligned_guide_nonaligned_supplement(H_omics1)
        )
        z_aligned_guide_omics2, z_nonaligned_supplement_omics2 = (
            self._split_aligned_guide_nonaligned_supplement(H_omics2)
        )
        H_omics1 = torch.cat([z_aligned_guide_omics1, z_nonaligned_supplement_omics1], dim=1)
        H_omics2 = torch.cat([z_aligned_guide_omics2, z_nonaligned_supplement_omics2], dim=1)

        if stage == 1:
            # Stage 1: fusion pretrain with reconstruction only (simple H H^T adjacency).
            adj_fusion_hat_omics1 = self.s(torch.mm(H_omics1, H_omics1.t()))
            adj_fusion_hat_omics2 = self.s(torch.mm(H_omics2, H_omics2.t()))
            return self._reconstruction_loss_only(
                omics_1,
                omics_2,
                adj_feature_omics1,
                adj_feature_omics2,
                adj_spatial_omics1,
                adj_spatial_omics2,
                H_omics1,
                adj_fusion_hat_omics1,
                H_omics2,
                adj_fusion_hat_omics2,
                H_omics1,
                adj_fusion_hat_omics1,
                H_omics2,
                adj_fusion_hat_omics2,
            )

        if stage == 2:
            # Stage 2: dual-branch graphs + cross-modal correlation + cross-fusion; returns clustering matrix S.
            adj_aligned_guide_omics1 = self.s(
                torch.mm(z_aligned_guide_omics1, z_aligned_guide_omics1.t())
            )
            adj_aligned_guide_omics2 = self.s(
                torch.mm(z_aligned_guide_omics2, z_aligned_guide_omics2.t())
            )
            adj_nonaligned_supplement_omics1 = self.s(
                torch.mm(z_nonaligned_supplement_omics1, z_nonaligned_supplement_omics1.t())
            )
            adj_nonaligned_supplement_omics2 = self.s(
                torch.mm(z_nonaligned_supplement_omics2, z_nonaligned_supplement_omics2.t())
            )

            w_nonaligned_supplement_omics1 = torch.sigmoid(
                self.nonaligned_supplement_gate_logit_omics1
            )
            w_nonaligned_supplement_omics2 = torch.sigmoid(
                self.nonaligned_supplement_gate_logit_omics2
            )
            w_aligned_guide_omics1 = 1.0 - w_nonaligned_supplement_omics1
            w_aligned_guide_omics2 = 1.0 - w_nonaligned_supplement_omics2
            adj_fusion_hat_omics1 = (
                w_aligned_guide_omics1 * adj_aligned_guide_omics1
                + w_nonaligned_supplement_omics1 * adj_nonaligned_supplement_omics1
            )
            adj_fusion_hat_omics2 = (
                w_aligned_guide_omics2 * adj_aligned_guide_omics2
                + w_nonaligned_supplement_omics2 * adj_nonaligned_supplement_omics2
            )

            loss_rec = self._reconstruction_loss_only(
                omics_1,
                omics_2,
                adj_feature_omics1,
                adj_feature_omics2,
                adj_spatial_omics1,
                adj_spatial_omics2,
                H_omics1,
                adj_fusion_hat_omics1,
                H_omics2,
                adj_fusion_hat_omics2,
                H_omics1,
                adj_fusion_hat_omics1,
                H_omics2,
                adj_fusion_hat_omics2,
            )
            loss_cross_fusion = self._cross_fusion_mismatch_loss(
                adj_fusion_hat_omics1, adj_fusion_hat_omics2
            )
            loss_cross_modal_corr = self._cross_modal_correlation_loss(
                z_aligned_guide_omics1, z_aligned_guide_omics2
            )
            S = 0.5 * adj_fusion_hat_omics1 + 0.5 * adj_fusion_hat_omics2
            return loss_rec, loss_cross_modal_corr, loss_cross_fusion, S

        raise ValueError(f"Invalid stage: {stage}. Expected 0, 1, or 2.")

    def _split_aligned_guide_nonaligned_supplement(self, H):
        z_aligned_guide = self.aligned_guide_proj(H)
        z_nonaligned_supplement = self.nonaligned_supplement_proj(H)
        return z_aligned_guide, z_nonaligned_supplement

    @staticmethod
    def _row_normalize_dense(A, eps=1e-8):
        row_sum = A.sum(dim=1, keepdim=True)
        return A / (row_sum + eps)

    @staticmethod
    def _unit_normalize_rows(H, eps=1e-8):
        return H / (H.norm(dim=1, keepdim=True) + eps)

    def _mixed_adjacency(self, adj_feature, adj_spatial):
        A_f = self._row_normalize_dense(adj_feature)
        A_s = self._row_normalize_dense(adj_spatial)
        alpha = torch.sigmoid(self.feature_spatial_mix_logit)
        return alpha * A_f + (1.0 - alpha) * A_s

    def _global_local_fusion(
        self, Z_base, adj_feature, adj_spatial, diffusion_hop_logits, fusion_balance
    ):
        A_mix = self._mixed_adjacency(adj_feature, adj_spatial)

        AZ = torch.mm(A_mix, Z_base)
        A2Z = torch.mm(A_mix, AZ)
        w = F.softmax(diffusion_hop_logits, dim=0)
        Z_global = w[0] * AZ + w[1] * A2Z

        residual = Z_base - AZ
        r_unit = self._unit_normalize_rows(residual)
        w_r = torch.sigmoid(self.local_scale_r)
        Z_local = AZ + w_r * r_unit

        eta = torch.sigmoid(fusion_balance(torch.cat([Z_global, Z_local], dim=1)))
        return (1.0 - eta) * Z_global + eta * Z_local

    def _reconstruction_loss_only(
        self,
        omics_1,
        omics_2,
        adj_feature_omics1,
        adj_feature_omics2,
        adj_spatial_omics1,
        adj_spatial_omics2,
        z_omics1_feature,
        a_1,
        z_omics2_feature,
        a_2,
        z_omics1_spatial,
        a_3,
        z_omics2_spatial,
        a_4,
    ):
        X_omics1_feature_hat, A1_hat = self.gae_feature_omics1.decoder(
            z_omics1_feature, adj_feature_omics1
        )
        X_omics2_feature_hat, A2_hat = self.gae_feature_omics2.decoder(
            z_omics2_feature, adj_feature_omics2
        )
        X_omics1_spatial_hat, A3_hat = self.gae_spatial_omics1.decoder(
            z_omics1_spatial, adj_spatial_omics1
        )
        X_omics2_spatial_hat, A4_hat = self.gae_spatial_omics2.decoder(
            z_omics2_spatial, adj_spatial_omics2
        )

        loss_rec_omics1_feature = reconstruction_loss(
            omics_1, adj_feature_omics1, X_omics1_feature_hat, (A1_hat + a_1) / 2
        )
        loss_rec_omics1_spatial = reconstruction_loss(
            omics_1, adj_spatial_omics1, X_omics1_spatial_hat, (A3_hat + a_3) / 2
        )
        loss_rec_omics1 = loss_rec_omics1_feature + loss_rec_omics1_spatial

        loss_rec_omics2_feature = reconstruction_loss(
            omics_2, adj_feature_omics2, X_omics2_feature_hat, (A2_hat + a_2) / 2
        )
        loss_rec_omics2_spatial = reconstruction_loss(
            omics_2, adj_spatial_omics2, X_omics2_spatial_hat, (A4_hat + a_4) / 2
        )
        loss_rec_omics2 = loss_rec_omics2_feature + loss_rec_omics2_spatial

        return self.beta_1 * loss_rec_omics1 + self.beta_2 * loss_rec_omics2

    @staticmethod
    def _sample_wise_standardize(H, eps=1e-8):
        mean = H.mean(dim=1, keepdim=True)
        std = H.std(dim=1, keepdim=True, unbiased=False)
        return (H - mean) / (std + eps)

    def _cross_modal_correlation_loss(self, H_omics1, H_omics2):
        ds = min(H_omics1.shape[1], H_omics2.shape[1])
        ds = max(1, ds)
        x = self._sample_wise_standardize(H_omics1[:, :ds])
        y = self._sample_wise_standardize(H_omics2[:, :ds])

        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        numerator = (x_centered * y_centered).sum(dim=0)
        denominator = torch.sqrt(
            (x_centered.pow(2).sum(dim=0) + 1e-8) * (y_centered.pow(2).sum(dim=0) + 1e-8)
        )
        corr = numerator / denominator
        return -corr.mean()

    def _cross_fusion_mismatch_loss(self, A1, A2):
        tau = opt.args.tau
        non_diag = ~torch.eye(A1.size(0), dtype=torch.bool, device=A1.device)

        mask_12 = (A1 < tau) & non_diag
        mask_21 = (A2 < tau) & non_diag
        penalty_12 = (A2 * mask_12.float()).sum() / (mask_12.float().sum() + 1e-8)
        penalty_21 = (A1 * mask_21.float()).sum() / (mask_21.float().sum() + 1e-8)
        return 0.5 * (penalty_12 + penalty_21)

    @staticmethod
    def compute_weights(n_omics1, n_omics2):
        denominator = n_omics1 + n_omics2
        if denominator == 0:
            raise ValueError("The sum of n_omics1 and n_omics2 should not be zero.")
        return n_omics2 / denominator, n_omics1 / denominator
