from utils import construct_graph, construct_high_order_graph, refine_adj_spatial, convert_to_tensor
from processing import preprocess
import scanpy as sc
import numpy as np
import opt
import os


SUPPORTED_DATASETS = [
    "Human_Lymph_Node_A1",
    "Human_Lymph_Node_D1",
    "Mouse_Brain_E13_S1",
    "Mouse_Brain_E15_S1",
]


def load_data(return_spatial: bool = False):
    """
    Load and preprocess data from supported datasets.

    Constructs feature graphs and spatial graphs, refines the spatial graphs,
    and returns PCA features, normalized graphs, and labels.
    """
    if opt.args.name in ["Human_Lymph_Node_A1", "Human_Lymph_Node_D1"]:
        adata_omics1, adata_omics2, label = load_human_lymph_node(opt.args.name)
        datatype = "RNA-ADT"
    elif opt.args.name in ["Mouse_Brain_E13_S1", "Mouse_Brain_E15_S1"]:
        adata_omics1, adata_omics2, label = load_mouse_brain_with_anno(opt.args.name)
        datatype = "RNA-ATAC"
    else:
        raise ValueError(
            f"Dataset {opt.args.name} not supported. "
            f"Supported: {', '.join(SUPPORTED_DATASETS)}"
        )

    data = preprocess(adata_omics1, adata_omics2, datatype=datatype)
    X_pca_omics1, X_pca_omics2 = (
        data["adata_omics1"].obsm["feat"].copy(),
        data["adata_omics2"].obsm["feat"].copy(),
    )

    adj_omics1, adj_norm_omics1 = construct_high_order_graph(
        X_pca_omics1, k=opt.args.k, motif_ratio=opt.args.motif_ratio
    )
    adj_omics2, adj_norm_omics2 = construct_high_order_graph(
        X_pca_omics2, k=opt.args.k, motif_ratio=opt.args.motif_ratio
    )

    spatial = adata_omics1.obsm["spatial"].copy()
    adj_spatial, _ = construct_graph(spatial, k=opt.args.spatial_k)

    adj_spatial_refine_omics1 = refine_adj_spatial(adj_omics1, adj_spatial)
    adj_spatial_refine_omics2 = refine_adj_spatial(adj_omics2, adj_spatial)

    adj_feature_omics1 = adj_norm_omics1
    adj_feature_omics2 = adj_norm_omics2

    X_pca_omics1, X_pca_omics2 = convert_to_tensor([X_pca_omics1, X_pca_omics2])
    adj_feature_omics1, adj_feature_omics2, adj_spatial_refine_omics1, adj_spatial_refine_omics2 = (
        convert_to_tensor(
            [
                adj_feature_omics1,
                adj_feature_omics2,
                adj_spatial_refine_omics1,
                adj_spatial_refine_omics2,
            ]
        )
    )

    if opt.args.show:
        print("-------------Details Of The Dataset------------")
        print("-----------------------------------------------")
        print("Dataset name         :", opt.args.name)
        print("Omics1 shape         :", X_pca_omics1.shape)
        print("Omics2 shape         :", X_pca_omics2.shape)
        if label is not None:
            print("Category num         :", max(label) - min(label) + 1)
            print("Category distribution:")
            for i in range(max(label + 1)):
                print("Label", i, end=":")
                print(len(label[np.where(label == i)]))
        print("-----------------------------------------------")

    if return_spatial:
        return (
            X_pca_omics1,
            X_pca_omics2,
            adj_feature_omics1,
            adj_feature_omics2,
            label,
            adj_spatial_refine_omics1,
            adj_spatial_refine_omics2,
            spatial,
        )

    return (
        X_pca_omics1,
        X_pca_omics2,
        adj_feature_omics1,
        adj_feature_omics2,
        label,
        adj_spatial_refine_omics1,
        adj_spatial_refine_omics2,
    )


def load_human_lymph_node(name):
    """Load human lymph node (A1 / D1) RNA + ADT from ./data/10X/{name}/."""
    base_dir = f"./data/10X/{name}"
    adata_rna = sc.read_h5ad(f"{base_dir}/adata_RNA.h5ad")
    adata_adt = sc.read_h5ad(f"{base_dir}/adata_ADT.h5ad")

    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()

    if name == "Human_Lymph_Node_A1":
        label_path = f"{base_dir}/label.npy"
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"A1 labels not found: {label_path}")
        label = np.load(label_path)
    elif name == "Human_Lymph_Node_D1":
        label_path = f"{base_dir}/D1_annotation_labels.csv"
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"D1 labels not found: {label_path}")
        label = np.loadtxt(label_path, delimiter=",", skiprows=1).astype(np.int64)
    else:
        raise ValueError(f"Unsupported lymph node dataset: {name}")

    return adata_rna, adata_adt, label


def load_mouse_brain_with_anno(name):
    """Load mouse embryo brain (E13_S1 / E15_S1) RNA + ATAC and anno.csv labels."""
    base_dir = f"./data/10X/{name}"
    adata_rna = sc.read_h5ad(f"{base_dir}/adata_RNA.h5ad")
    adata_atac = sc.read_h5ad(f"{base_dir}/adata_ATAC.h5ad")
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()

    anno_path = f"{base_dir}/anno.csv"
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"No annotation file found at {anno_path}")

    anno = np.genfromtxt(anno_path, delimiter=",", dtype=str, skip_header=1)
    if anno.ndim == 1:
        anno = anno.reshape(1, -1)
    if anno.shape[1] < 3:
        raise ValueError(f"Invalid annotation file format: {anno_path}")

    barcode_to_cluster = {row[0]: row[2] for row in anno}
    if "barcode" in adata_rna.obs.columns:
        rna_barcodes = np.asarray(adata_rna.obs["barcode"]).astype(str)
    else:
        rna_barcodes = np.asarray(adata_rna.obs_names).astype(str)

    missing = [bc for bc in rna_barcodes if bc not in barcode_to_cluster]
    if missing:
        raise ValueError(
            f"Found {len(missing)} RNA barcodes missing in anno.csv. "
            f"Example missing barcode: {missing[0]}"
        )

    cluster_labels = np.array([barcode_to_cluster[bc] for bc in rna_barcodes], dtype=str)
    _, label = np.unique(cluster_labels, return_inverse=True)
    label = label.astype(np.int64)

    return adata_rna, adata_atac, label
