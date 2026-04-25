import sys
import gc

import anndata
import cassiopeia as cas
import networkx as nx
import numpy as np
import pandas as pd
import pickle as pic
import scanpy as sc
import scipy
import squidpy as sq

from tqdm import tqdm

SAMPLES = [
        "slide_pilot3/B94_04", "slide_pilot3/B94_20", "slide_pilot3/B94_23", "slide_pilot3/B95_24", "slide_pilot3/B99_22",
        "slide_pilot3/C12_01", "slide_pilot3/C12_02", "slide_pilot3/C12_03", "slide_pilot3/C12_05", "slide_pilot3/C12_09", "slide_pilot3/C12_10", "slide_pilot3/C12_12", "slide_pilot3/C12_04",
        "slide_pilot4/C27_11",
        "slide_pilot4/M09_01", "slide_pilot4/M09_04", "slide_pilot4/M09_07", "slide_pilot4/M09_08",
        "slide_pilot5/C36_09", "slide_pilot5/C36_10", "slide_pilot5/C36_11", "slide_pilot5/Curio_09", "slide_pilot5/Curio_10", 
        "slide_pilot5/M09_11", "slide_pilot5/M09_13", "slide_pilot5/M09_14", "slide_pilot5/M09_17", "slide_pilot5/M09_18", "slide_pilot5/M11_05",
        "Layer1/C44_06", "Layer1/C44_08",
        "Layer1/M18_01", "Layer1/M18_02", "Layer1/M18_04",
        "Layer2/M11_06", "Layer2/M18_06",
        "Layer3/M11_08", "Layer3/M18_17",
        "Layer4/M18_13", "Layer4/M11_14", 
        "Layer1/Curio_001", "Layer2/Curio_002", "Layer3/Curio_003", "Layer4/Curio_004"
]

hotspot_clusters = pd.read_csv("/Users/matthewjones/projects/kptc_spatial/data/hotspot_modules_consensus.tsv", sep=' ', index_col=0)
to_merge = []

HOMEDIR="/path/to/KPSpatial-release/data"

for sample in tqdm(SAMPLES):
    print(f"Processing {sample}...")

    adata_raw = sc.read_h5ad(f'{HOMEDIR}/{sample}_RNA/adata_merged.filtered.h5ad')
    adata_raw.var_names = adata_raw.var_names.astype(str)
    adata_raw.var_names_make_unique()
    
    target_sum = np.median(np.array(adata_raw.X.sum(axis=1)))
    sc.pp.normalize_total(adata_raw, target_sum=target_sum)

    # filter genes with no expression
    sc.pp.filter_genes(adata_raw, min_cells=10)

    sc.pp.log1p(adata_raw)
    sc.pp.scale(adata_raw)

    program_names = [mod for mod in hotspot_clusters['Community_Module'].unique() if type(mod) == str]

    for program_name, group in hotspot_clusters.groupby('Community_Module'):
        
        if type(program_name) != str:
            continue

        group['Gene'] = group.index.values
        unique_modules = group['Module'].unique()

        genes = group['Gene'].value_counts()
        module_genes = np.intersect1d(adata_raw.var_names, genes[genes > int(0.25 * len(unique_modules))].index.values)

        if len(module_genes) < 1:
            continue

        sc.tl.score_genes(adata_raw, module_genes, ctrl_size=100, n_bins=30, score_name=program_name, use_raw=False)
    
    new_df = adata_raw.obs[program_names]
    new_df['Sample'] = sample.split("/")[1]
    to_merge.append(new_df)

    gc.collect()

community_module_summary = pd.concat(to_merge)
community_module_summary.to_csv("/Users/matthewjones/projects/kptc_spatial/data/hotspot_modules_consensus.scores.v2.tsv", sep=' ')
