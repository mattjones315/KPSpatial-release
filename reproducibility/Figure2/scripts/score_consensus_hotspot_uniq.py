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

SAMPLES = ['M18_01', 'M18_02', 'M18_04', 'Curio_001',
           'C44_06', 'C44_08', 'M11_06', 'M18_06', 'Curio_002',
           'M11_08', 'M18_17', 'Curio_003',
           'M11_14', 'M18_13', 'Curio_004',
           'M09_11', 'M09_13', 'M09_14',
           'M09_17', 'M09_18', 'M11_05',
           'Curio_09', 'Curio_10', 'C36_09', 'C36_10', 'C36_11', 'C27_11',
           # 'C27_26',
           'M09_01', 'M09_04', 'M09_07', 'M09_08',
            'B94_04', 'B94_20', 'B94_20', 'B95_24',
            'B99_22', 'C12_01', 'C12_02', 'C12_03',
            'C12_04', 'C12_05', 'C12_09', 'C12_10', 'C12_12'
]

DATA_DIRECTORY = '/orcd/data/ki/001/lab/jones/mgjones/kptc_spatial/KPSpatial_Data/'

hotspot_modules_uniq = pd.read_csv(f"/orcd/data/ki/001/lab/jones/mgjones/kptc_spatial/KPSpatial-release/reproducibility/Figure2/data/hotspot_modules_consensus_uniq.tsv", sep='\t', index_col=0)
to_merge = []

for library_id in tqdm(SAMPLES):
    print(f"Processing {library_id}...")

    adata_raw = sc.read_h5ad(f'{DATA_DIRECTORY}/slideseq/expression/{library_id}_adata.h5ad')
    adata_raw.var_names = adata_raw.var_names.astype(str)
    adata_raw.var_names_make_unique()
    
    adata_raw.layers["counts"] = adata_raw.X.copy() # preserve counts
    adata_raw.layers['logged'] = adata_raw.X.copy()

    target_sum = np.median(np.array(adata_raw.X.sum(axis=1)))
    sc.pp.normalize_total(adata_raw, target_sum=target_sum)
    sc.pp.log1p(adata_raw, layer='logged')
    sc.pp.scale(adata_raw)

    program_names = [mod for mod in hotspot_modules_uniq['Community_Module'].unique() if type(mod) == str]

    for program_name, group in hotspot_modules_uniq.groupby('Community_Module'):
        
        if type(program_name) != str:
            continue

        module_genes = group['gene'].values

        if len(module_genes) < 1:
            continue

        sc.tl.score_genes(adata_raw, module_genes, ctrl_size=100, n_bins=30, score_name=program_name, use_raw=False)
        adata_raw.obs[program_name] = (adata_raw.obs[program_name] - adata_raw.obs[program_name].mean()) / adata_raw.obs[program_name].std()
    
    new_df = adata_raw.obs[program_names]
    new_df['Sample'] = library_id
    to_merge.append(new_df)

    gc.collect()

community_module_summary = pd.concat(to_merge)
community_module_summary.to_csv(f"{DATA_DIRECTORY}/hotspot_modules_consensus_uniq.scores.tsv", sep='\t')
