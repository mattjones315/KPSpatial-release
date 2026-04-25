"""
A script that will quantify plasticity based on phylogenies inferred from 
Slide-tags data.
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
from utilities import tree_utilities

####### Functions
def min_max_normalize(data):

    _ma, _mi = data.max(), data.min()
    return (data - _mi) / (_ma - _mi)

def calculate_l2(cell, neighbors, X, meta=None, p=2):
    
    cell_vec = X.loc[cell].values
    
    l2 = []
    for n in neighbors:
        if (meta is not None) and (meta.loc[n] == meta.loc[cell]):
            l2.append(0)
        else:
            l2.append(scipy.spatial.distance.minkowski(cell_vec, X.loc[n].values, p=p))
        
    return np.mean(l2)

def calculate_single_cell_plasticity(tree, adata):

    tree.cell_meta = adata.obs.loc[tree.leaves]
    parsimony = cas.tl.score_small_parsimony(tree, meta_item="cell_type")

    # compute plasticities for each node in the tree
    for node in tqdm(tree.depth_first_traverse_nodes(), total=len(tree.nodes), desc='Computing single-cell plasticity'):
        effective_plasticity = cas.tl.score_small_parsimony(
            tree, meta_item="cell_type", root=node
        )
        size_of_subtree = len(tree.leaves_in_subtree(node))
        tree.set_attribute(
            node, "effective_plasticity", effective_plasticity / size_of_subtree
        )

    tree.cell_meta["scPlasticity"] = np.nan
    for leaf in tree.leaves:
        plasticities = []
        parent = tree.parent(leaf)
        while True:
            if parent == tree.root:
                break
            plasticities.append(tree.get_attribute(parent, "effective_plasticity"))
            parent = tree.parent(parent)

        if len(plasticities) > 0:
            tree.cell_meta.loc[leaf, "scPlasticity"] = np.mean(plasticities)

    return tree

def calculate_single_cell_l2_plasticity(tree, adata):

    cell_to_l2_plasticity = {}
    for l in tqdm(tree.leaves):

        parent = tree.parent(l)
        l2 = np.nan
        if not tree.is_root(parent):
            children = [c for c in tree.leaves_in_subtree(parent) if c != l]

            # l2 = calculate_l2(l, children, latent, None, p=2)
            l2 = calculate_l2(l, children, latent, adata.obs['cell_type'], p=2)
            # l2 = calculate_l2(l, children, X)
        
        cell_to_l2_plasticity[l] = l2

    tree.cell_meta['scPlasticity_L2'] = tree.cell_meta.index.map(cell_to_l2_plasticity)
    tree.cell_meta['scPlasticity_L2'] = min_max_normalize(tree.cell_meta['scPlasticity_L2'])
    tree.cell_meta['scPlasticity_L2'] = np.clip(tree.cell_meta['scPlasticity_L2'], np.nanpercentile(tree.cell_meta['scPlasticity_L2'], 1), np.nanpercentile(tree.cell_meta['scPlasticity_L2'], 99))
    return tree

def calculate_spatial_plasticity(adata, tumor_states):
    
    spatial_graph = nx.from_numpy_array(adata.obsp['spatial_connectivities'])
    node_map = dict(zip(range(adata.obsp['spatial_connectivities'].shape[0]), adata.obs_names))
    spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

    spatial_plasticity = pd.DataFrame(index=adata.obs_names)
    spatial_plasticity['plasticity'] = np.nan
    for cell in adata.obs_names:

        if adata.obs.loc[cell, 'cell_type'] not in tumor_states:
            continue
        
        neighborhood = [node for (_, node) in nx.bfs_edges(spatial_graph, cell, depth_limit=1)]
        neighborhood = [n for n in neighborhood if adata.obs.loc[n, 'cell_type'] in tumor_states]
        
        if len(neighborhood) < 2:
            spatial_plasticity.loc[cell, 'plasticity'] = np.nan
        else:
            spatial_plasticity.loc[cell, 'plasticity'] = len([n for n in neighborhood if adata.obs.loc[n, 'cell_type'] != adata.obs.loc[cell, 'cell_type']]) / len(neighborhood)
    
    spatial_plasticity['plasticity'] = min_max_normalize( spatial_plasticity['plasticity'])
    spatial_plasticity['plasticity'] = np.clip(spatial_plasticity['plasticity'], np.nanpercentile(spatial_plasticity['plasticity'], 1), np.nanpercentile(spatial_plasticity['plasticity'], 99))

    return spatial_plasticity

HOMEDIR = "/path/to/KPSpatial/data/"
tumor_states = ['AT1-like',
              'AT2-like',
              'Ciliated cell',
              'Club cell',
              'Early EMT',
              'Early gastric',
              'Endoderm-like',
              'Gastric-like',
              'High-plasticity cell state',
              'Late gastric',
              'Neuronal-like',
              'Pre-EMT']

adata_all = sc.read_h5ad(f"{HOMEDIR}/adata_merged.scanvi_slidetags.all.assigned.h5ad")


SAMPLES=[('Layer1', 'Puck1'),
         ('Layer2', 'Puck1'),
         ('Layer3', 'Puck1'),
         ('Layer3', 'Puck2'),
         ('Layer4', 'Puck1')]

to_merge = []
for (layer, puck) in SAMPLES:
    print(f"Processing {layer}, {puck}")

    adata = sc.read_h5ad(f'{HOMEDIR}/{layer}/adata_slidetags.{layer}.{puck}.segmented.h5ad')
    adata_assigned = sc.read_h5ad(f'{HOMEDIR}/{layer}/adata_slidetags.{layer}.{puck}.h5ad')
    
    adata.obs['cell_type'] = adata_assigned.obs.loc[adata.obs_names, 'cell_type']

    tree = pic.load(open(f'{HOMEDIR}/{layer}/slidetags_trees/reconstructions/imputed/{puck}/hybrid.pkl', 'rb'))
    state_to_indel = pic.load(open(f'{HOMEDIR}/{layer}/slidetags_trees/character_matrix/{puck}/character_matrix.imputed_states.pkl', 'rb'))

    tree.cell_meta = adata.obs.loc[tree.leaves]
    non_tumor_cells = np.setdiff1d(tree.leaves, tree.cell_meta[tree.cell_meta['cell_type'].isin(tumor_states)].index.values)
    tree.remove_leaves_and_prune_lineages(non_tumor_cells)

    adata.obs['tumor'] = 'False'
    adata.obs.loc[tree.leaves, 'tumor'] = 'True'

    adata = adata[np.intersect1d(adata_all.obs_names, adata.obs_names)]
    tree.remove_leaves_and_prune_lineages(np.setdiff1d(tree.leaves, adata.obs_names))

    print(f"Read in tree with {len(tree.leaves)} leaves ({round(len(tree.leaves) / adata.shape[0] * 100, 3)}% of all spots).")

    character_matrix = tree.character_matrix.copy()
    priors = tree.priors

    allele_table = tree_utilities.character_matrix_to_allele_table(character_matrix, state_to_indel, keep_ambiguous=False)
    adata.raw = adata

    latent = pd.DataFrame(adata_all.obsm['X_scANVI'], index=adata_all.obs_names)
    latent = latent.loc[np.intersect1d(adata.obs_names, adata_all.obs_names)]

    # compute single-cell plasticity
    tree = calculate_single_cell_plasticity(tree, adata)
    adata.obs["scPlasticity"] = np.nan
    adata.obs.loc[tree.leaves, "scPlasticity"] = tree.cell_meta["scPlasticity"]

    # compute single-cell L2 plasticity
    tree = calculate_single_cell_l2_plasticity(tree, adata)
    adata.obs['scPlasticity_L2'] = np.nan
    adata.obs.loc[tree.leaves, 'scPlasticity_L2'] = tree.cell_meta['scPlasticity_L2']

    # compute spatial plasticity
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial", n_neighs=15, delaunay=False)
    adata.uns['spatial_neighbors']['params']['method'] = 'umap'
    spatial_graph = nx.from_numpy_array(adata.obsp['spatial_connectivities'])
    node_map = dict(zip(range(adata.obsp['spatial_connectivities'].shape[0]), adata.obs_names))
    spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

    spatial_plasticity = calculate_spatial_plasticity(adata, tumor_states)
    adata.obs["spPlasticity"] = np.nan
    adata.obs.loc[spatial_plasticity.index, "spPlasticity"] = spatial_plasticity['plasticity']

    # compute distance to boundary
    boundary_cells = adata[adata.obs['tumor_id'] == 'non-tumor'].obs_names
    nn_dist = []
    for l in tqdm(tree.leaves):    
            
        nn_dist.append(np.min((np.sum(np.abs(adata[boundary_cells,:].obsm['spatial'] - adata[l,:].obsm['spatial']), axis=1))))

    plasticity_df = adata.obs[['scPlasticity', 'scPlasticity_L2', 'spPlasticity']]
    plasticity_df['dist_to_boundary'] = np.nan
    plasticity_df.loc[tree.leaves, 'dist_to_boundary'] = nn_dist
    to_merge.append(plasticity_df)

final_plasticity = pd.concat(to_merge)
final_plasticity.to_csv(f'{HOMEDIR}/data/slidetags_plasticity.tsv', sep='\t')