"""
Compute significance of clustering results from infercnv.
"""
import sys

import cassiopeia as cas
import numpy as np
import pandas as pd
import pickle as pic
import random
import scanpy as sc
from tqdm.auto import tqdm

sys.path.append("/Users/matthewjones/projects/kptc_spatial/KPTracer-spatial/")
from utilities import tree_utilities


### Utility functions
def find_nearest_neighbor(tree, leaf):

    node = tree.parent(leaf)
    while len(tree.leaves_in_subtree(node)) < 2:
        node = tree.parent(node)

    min_distance = np.inf
    neighbors = []
    for l in tree.leaves_in_subtree(node):
        
        if l == leaf:
            continue

        distance = tree.get_distance(l, leaf)
        assert int(distance) == distance
        distance = int(distance)

        if distance < min_distance:
            neighbors = [l]
            min_distance = distance
        elif distance == min_distance:
            neighbors.append(l)
    return neighbors, min_distance

def get_all_nearest_neighbors(tree):

    neighbor_dict = {}
    for l in tqdm(tree.leaves):

        neighbor_dict[l] = find_nearest_neighbor(tree, l)

    return neighbor_dict
        

def assess_nn_purity(tree, clusters, nn_dict=None):

    if nn_dict is None:
        nn_dict = get_all_nearest_neighbors(tree)
    
    purity = 0
    for l in tree.leaves:

        cluster = clusters[l]

        neighbors = find_nearest_neighbor(tree, l)
        _purity = 0
        for n in neighbors[0]:
            _purity += int(clusters[n] == cluster)

        purity += (_purity / len(neighbors[0]))

    return purity / len(tree.leaves)

def shuffle_dict(d):
    temp = list(d.values())
    random.shuffle(temp)
    
    # reassigning to keys
    res = dict(zip(d, temp))
    return res

def bootstrap_nn_purity(tree, clusters, nn_dict=None, B=100):

    if nn_dict is None:
        nn_dict = get_all_nearest_neighbors(tree)

    nn_score = assess_nn_purity(tree, clusters, nn_dict)

    significance = 0
    
    for _ in tqdm(range(B)):
        _clusters = shuffle_dict(clusters)
        _score = assess_nn_purity(tree, _clusters, nn_dict)
        if _score > nn_score:
            significance += 1

    return nn_score, significance/B

def get_subtree(_tree, tumor):

    query_cells = list(adata[adata.obs['tumor_id'] == tumor,:].obs_names)
    overlapping_cells = np.intersect1d(_tree.leaves, query_cells)
    to_remove = np.setdiff1d(_tree.leaves, overlapping_cells)
    
    subtree = _tree.copy()
    subtree.remove_leaves_and_prune_lineages(to_remove)

    return subtree

### Start script
SAMPLES=[('Layer1', 'Puck1'),
         ('Layer2', 'Puck1'),
         ('Layer3', 'Puck1'),
         ('Layer3', 'Puck2'),
         ('Layer4', 'Puck1')]

to_concat = []
for (layer, puck) in SAMPLES:

    print(f"Processing {layer}/{puck}...")
    # read in adata
    adata = sc.read_h5ad(f'/Users/matthewjones/projects/kptc_spatial/{layer}/adata_slidetags.{layer}.{puck}.segmented.annotated.h5ad')

    # read in tree
    tree = pic.load(open(f'/Users/matthewjones/projects/kptc_spatial/{layer}/slidetags_trees/reconstructions/imputed/{puck}/hybrid.pkl', 'rb'))
    tree = tree_utilities.cleanup_character_matrix_and_collapse(tree)

    infercnv_dir =  f'/Users/matthewjones/projects/kptc_spatial/{layer}/slidetags_infercnv/{puck}/output'
    infercnv_groupings = pd.read_csv(f'{infercnv_dir}/infercnv.observation_groupings.txt', sep=' ')
    infercnv_groupings['cell_group_name'] = infercnv_groupings.apply(lambda x: f'{x["Dendrogram Group"]}', axis=1)

    tree.cell_meta = pd.DataFrame(index=tree.leaves)
    tree.cell_meta['infercnv_clusters'] = None
    tree.cell_meta['tumor_id'] = adata.obs.loc[tree.leaves, 'tumor_id']

    overlapping = np.intersect1d(tree.leaves, infercnv_groupings.index)
    tree.cell_meta.loc[overlapping, 'infercnv_clusters'] = infercnv_groupings.loc[overlapping, 'cell_group_name'].astype(str)

    # start pipeline
    for tumor in tree.cell_meta['tumor_id'].unique():
        
        if tumor == 'non-tumor':
            continue

        print(f'   Processing {tumor}...')
        subtree = get_subtree(tree, tumor)

        nn_dict = get_all_nearest_neighbors(subtree)
        clusters = subtree.cell_meta['infercnv_clusters'].to_dict()
        nn_score, significance = bootstrap_nn_purity(subtree, subtree.cell_meta['infercnv_clusters'].to_dict(), nn_dict, B=1000)

        accur_df = pd.DataFrame([[layer, puck, tumor, nn_score, significance]], columns = ['Layer', 'Puck', 'Tumor', 'NN_score', 'Significance'])
        to_concat.append(accur_df)
    
    final_df = pd.concat(to_concat)
    final_df.to_csv("/Users/matthewjones/projects/kptc_spatial/data/slidetags_infercnv_nn_score.tsv", sep='\t', index=False)
