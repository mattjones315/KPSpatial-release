"""
A script to compute neighborhood abundances for slideseq data.
"""
import sys

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

sys.path.append("/path/to/KPSpatial-release/")
from utilities import spatial_utilities

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

RADIUS=28
HOMEDIR="/path/to/KPSpatial/data"

hotspot_scores = pd.read_csv("{HOMEDIR}/hotspot_modules_consensus.scores.v2.tsv", sep='\t', index_col = 0)
program_names = [m for m in hotspot_scores.columns if 'Module' in m]

neighborhood_columms = [f'neighborhood_{mod}' for mod in program_names]


to_merge = []
for sample in SAMPLES:

    print(F">> Processing {sample}...")

    DATASET = sample.split("/")[0]
    SAMPLE=sample.split("/")[1]

    adata = sc.read_h5ad(f'{HOMEDIR}/{DATASET}/{SAMPLE}_RNA/adata_merged.filtered.destvi.segmented_manual.h5ad')
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    hotspot_scores_sample = hotspot_scores[hotspot_scores['Sample'] == SAMPLE].drop_duplicates()

    adata.obs['sample'] = SAMPLE
    overlapping = np.intersect1d(adata.obs_names, hotspot_scores_sample.index.values)
    adata = adata[overlapping,:]
    adata.obs[program_names] = hotspot_scores_sample.loc[adata.obs_names, program_names]

    assignments = np.array(program_names)[adata.obs[program_names].apply(lambda x: np.argmax(x), axis=1)]
    adata.obs['program_assignments'] = assignments

    # compute spatial graph
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial", radius=RADIUS, delaunay=False)
    adata.uns['spatial_neighbors']['params']['method'] = 'umap'

    spatial_graph = nx.from_numpy_array(adata.obsp['spatial_connectivities'])
    node_map = dict(zip(range(adata.obsp['spatial_connectivities'].shape[0]), adata.obs_names))
    spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

    adata.obs[neighborhood_columms] = np.nan
    for cell in tqdm(adata.obs_names):

        neighborhood = [node for (_, node) in nx.bfs_edges(spatial_graph, cell, depth_limit=1)]

        if len(neighborhood) < 2:
            adata.obs.loc[cell, neighborhood_columms] = adata.obs.loc[cell, program_names].values

        else:
            neighborhood_composition = spatial_utilities.quantify_neighborhood(cell, adata, program_names, neighborhood_graph=spatial_graph, number_of_hops=1)
            neighborhood_composition.columns = neighborhood_columms
        
            adata.obs.loc[cell, neighborhood_composition.columns] = neighborhood_composition.values

    to_merge.append(adata.obs[neighborhood_columms + ['tumor_id', 'sample']])

neighborhood_composition = pd.concat(to_merge)
neighborhood_composition.to_csv(f"{HOMEDIR}/data/neighborhood_composition.tsv", sep='\t')
