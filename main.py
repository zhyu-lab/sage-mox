import numpy as np
import torch
import opt
from utils import post_proC, print_metrics, set_seed, cross_modal_corr_lambda_schedule_scale
from model import SAGEMoX
from evaluation import eval
from load_data import load_data
import tqdm


if __name__ == '__main__':

    set_seed(seed=opt.args.seed)

    (
        X_omics1,
        X_omics2,
        adj_feature_omics1,
        adj_feature_omics2,
        label,
        adj_spatial_omics1,
        adj_spatial_omics2,
    ) = load_data()

    opt.args.n_omics1 = X_omics1.shape[1]
    opt.args.n_omics2 = X_omics2.shape[1]
    if label is None:
        raise ValueError(
            f"Dataset {opt.args.name} has no labels; n_cluster is set from len(unique(label))."
        )
    opt.args.n_cluster = int(len(np.unique(label)))

    print("=" * 10 + " Pretraining has begun! " + "=" * 10)

    model = SAGEMoX().to(opt.args.device)
    optimizer0 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(opt.args.pretrain_epoch_stage0), ncols=200)

    for _ in pbar:
        loss_rec = model(
            X_omics1,
            X_omics2,
            adj_feature_omics1,
            adj_feature_omics2,
            adj_spatial_omics1,
            adj_spatial_omics2,
            stage=0,
        )

        pretrain_loss = opt.args.lambda_0 * loss_rec
        optimizer0.zero_grad()
        pretrain_loss.backward()
        optimizer0.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(opt.args.pretrain_epoch + 1), ncols=200)
    for _ in pbar:
        loss_rec = model(
            X_omics1,
            X_omics2,
            adj_feature_omics1,
            adj_feature_omics2,
            adj_spatial_omics1,
            adj_spatial_omics2,
            stage=1,
        )

        pretrain_loss = opt.args.lambda_0 * loss_rec
        optimizer.zero_grad()
        pretrain_loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})

    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.train_lr)
    pbar2 = tqdm.tqdm(range(opt.args.epoch + 1), ncols=200)
    for epoch in pbar2:
        loss_rec, loss_cross_modal_corr, loss_cross_fusion, S = model(
            X_omics1,
            X_omics2,
            adj_feature_omics1,
            adj_feature_omics2,
            adj_spatial_omics1,
            adj_spatial_omics2,
            stage=2,
        )

        corr_scale = cross_modal_corr_lambda_schedule_scale(
            epoch, opt.args.cross_modal_corr_warmup_epochs, opt.args.cross_modal_corr_warmup_start
        )
        lambda_cross_modal_corr = opt.args.lambda_1 * corr_scale
        total_loss = (
            opt.args.lambda_0 * loss_rec
            + lambda_cross_modal_corr * loss_cross_modal_corr
            + opt.args.lambda_2 * loss_cross_fusion
        )

        optimizer2.zero_grad()
        total_loss.backward()
        optimizer2.step()

        pbar2.set_postfix({'loss': '{0:1.4f}'.format(total_loss)})

    S_cpu = S.cpu().detach().numpy()
    pred, _ = post_proC(S_cpu, opt.args.n_cluster)

    if label is not None:
        ari, nmi, ami, v_measure, homo = eval(label, pred)
        print_metrics(ari, nmi, ami, v_measure, homo)
