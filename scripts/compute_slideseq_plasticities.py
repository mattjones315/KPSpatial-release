"""
A script to compute single-cell plasticitiy scores for Slide-seq data.
"""
import sys

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
from utilities import spatial_utilities, tree_utilities

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
HOMEDIR="/path/to/KPSpatial-data/"

hotspot_scores = pd.read_csv(f"{HOMEDIR}/data/hotspot_modules_consensus.scores.v2.tsv", sep='\t', index_col = 0)
program_names = [m for m in hotspot_scores.columns if 'Module' in m]

# utility functions
def calculate_l2(cell, neighbors, X):
    
    cell_vec = X.loc[cell].values
    
    l2 = []
    for n in neighbors:
        l2.append(scipy.spatial.distance.minkowski(cell_vec, X.loc[n].values, p=2))
        
    return np.mean(l2)

def min_max_normalize(data):

    _ma, _mi = data.max(), data.min()
    return (data - _mi) / (_ma - _mi)

    
to_merge = []
for sample in SAMPLES:

    print(F">> Processing {sample}...")

    DATASET = sample.split("/")[0]
    SAMPLE=sample.split("/")[1]

    tree = pic.load(open(f'{HOMEDIR}/{DATASET}/{SAMPLE}_TS/trees/reconstructions/imputed/ambiguous/nj_hybrid.pkl', 'rb'))
    state_to_indel = pic.load(open(f'{HOMEDIR}/{DATASET}/{SAMPLE}_TS/trees/character_matrix/character_matrix.ambiguous.imputed_states.pkl', 'rb'))

    # clean up and collapse edges
    tree = tree_utilities.cleanup_character_matrix_and_collapse(tree)

    adata = sc.read_h5ad(f'{HOMEDIR}/{DATASET}/{SAMPLE}_RNA/adata_merged.filtered.h5ad')
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()

    adata_segmented = sc.read_h5ad(f'{HOMEDIR}/{DATASET}/{SAMPLE}_RNA/adata_merged.filtered.destvi.segmented_manual.h5ad')

    hotspot_scores_sample = hotspot_scores[hotspot_scores['Sample'] == SAMPLE]

    adata.obs['tumor_id'] = adata_segmented.obs.loc[adata.obs_names, 'tumor_id']
    adata.obs['tumor_boundary'] = adata_segmented.obs.loc[adata.obs_names, 'tumor_boundary']
    adata.obs[program_names] = hotspot_scores_sample.loc[adata.obs_names, program_names]

    assignments = np.array(program_names)[adata.obs[program_names].apply(lambda x: np.argmax(x), axis=1)]
    adata.obs['program_assignments'] = assignments

    print(f"Read in tree with {len(tree.leaves)} leaves ({round(len(tree.leaves) / adata.shape[0] * 100, 3)}% of all spots).")

    # compute spatial graph
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial", radius=28, delaunay=False)
    adata.uns['spatial_neighbors']['params']['method'] = 'umap'

    spatial_graph = nx.from_numpy_array(adata.obsp['spatial_connectivities'])
    node_map = dict(zip(range(adata.obsp['spatial_connectivities'].shape[0]), adata.obs_names))
    spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

    hs = adata.obs[program_names]
    hs = hs.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

    cell_to_plasticity = {}
    for l in tqdm(tree.leaves):

        parent = tree.parent(l)
        children = [c for c in tree.leaves_in_subtree(parent) if c != l]
        cell_to_plasticity[l] = calculate_l2(l, children, hs)

    adata.obs['L2_plasticity'] = np.nan
    adata.obs['L2_plasticity'] = adata.obs.index.map(cell_to_plasticity)
    adata.obs['L2_plasticity'] = min_max_normalize(adata.obs['L2_plasticity'])
    adata.obs['L2_plasticity'] = np.clip(adata.obs['L2_plasticity'],  np.nanpercentile(adata.obs['L2_plasticity'], 5), np.nanpercentile(adata.obs['L2_plasticity'], 95))

    # add meta data to tree
    tree.cell_meta = pd.DataFrame(index=tree.leaves)
    tree.cell_meta['tumor_id'] = adata.obs.loc[tree.leaves, 'tumor_id']
    tree.cell_meta['plasticity'] = adata.obs.loc[tree.leaves, 'L2_plasticity']

    neighborhood_columns = [f'neighborhood_{mod}' for mod in program_names]
    
    tree.cell_meta[neighborhood_columns] = np.nan
    for leaf in tqdm(tree.character_matrix.index):

        neighborhood = [node for (_, node) in nx.bfs_edges(spatial_graph, leaf, depth_limit=1)]

        if len(neighborhood) < 2:
            tree.cell_meta.loc[leaf, neighborhood_columns] = adata.obs.loc[leaf, program_names]

        else:
            neighborhood_composition = spatial_utilities.quantify_neighborhood(leaf, adata, program_names, neighborhood_graph=spatial_graph, number_of_hops=1)
            neighborhood_composition.columns = neighborhood_columns
        
            tree.cell_meta.loc[leaf, neighborhood_composition.columns] = neighborhood_composition.values
    
    summary_df = tree.cell_meta[['plasticity', 'tumor_id'] + neighborhood_columns]
    summary_df['Sample'] = SAMPLE
    summary_df.index = [f'{SAMPLE}.{x}' for x in summary_df.index.values]
    to_merge.append(summary_df)

plasticity_df = pd.concat(to_merge)
plasticity_df.to_csv(f"{HOMEDIR}/data/plasticity_neighborhood_slideseq.tsv", sep='\t')
