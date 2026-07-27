# sage-mox
Local-Global Fusion and Alignment-Complementary Cross-Omics Integration for Spatial Domain Identification

## Requirement
- torch==2.3.0
- python==3.9.23
- numpy==1.24.4
- pandas==1.5.3
- scikit-learn==1.6.1
- scanpy==1.9.7
- scipy==1.10.1
- anndata==0.9.2
- episcanpy == 0.3.2

## Installation

Clone the repository and create a conda environment:

```bash
git clone https://github.com/zhyu-lab/sage-mox.git
cd sage-mox
conda create -n sagemox python=3.9 -y
conda activate sagemox
```

Install the required packages:

```bash
pip install torch==2.3.0 numpy==1.24.4 scipy==1.10.1 scikit-learn==1.6.1
pip install scanpy==1.9.7 anndata==0.9.2 episcanpy==0.3.2 tqdm termcolor
```

## Data Preparation

The example dataset should be organized as follows:

```text
data/
└── 10X/
    └── Human_Lymph_Node_A1/
        ├── adata_RNA.h5ad
        ├── adata_ADT.h5ad
        └── label.npy
```

The RNA AnnData object should contain spatial coordinates in `adata.obsm["spatial"]`.

## Usage

Run SAGE-MoX from the root directory:

```bash
python main.py --name Human_Lymph_Node_A1 --device cuda:0
```

The A1 dataset is used by default, so it can also be run using:

```bash
python main.py
```


## Output and Evaluation

SAGE-MoX constructs a consensus affinity matrix and applies spectral clustering to obtain spatial domains. ARI, NMI, AMI, V-measure, and homogeneity are automatically calculated and printed in the terminal.
