import sys

import cassiopeia as cas
import numpy as np
import pandas as pd
import pickle as pic
import scanpy as sc
import seaborn as sns
import squidpy as sq

from tqdm import tqdm

sys.path.append("/Users/matthewjones/projects/kptc_spatial/KPSpatial-release/")

from utilities import phylodynamics

SLIDETAGS_SAMPLES = [('Layer1', 'Puck1'),
                   ('Layer2', 'Puck1'),
                   ('Layer3', 'Puck1'),
                   ('Layer3', 'Puck2'),
                   ('Layer4', 'Puck1')]

DATA_DIRECTORY="/Users/matthewjones/projects/kptc_spatial/"

tumor_states = ['AT1-like',
              'AT2-like',
              'Early EMT',
              'Early gastric',
              'Endoderm-like',
              'Gastric-like',
              'High-plasticity cell state',
              'Late gastric',
              'Neuronal-like',
              'Pre-EMT']

trees = {}
adata_to_merge = []
for i, sample in zip(range(len(SLIDETAGS_SAMPLES)), SLIDETAGS_SAMPLES):
    print(sample)    
    LAYER, PUCK = sample

    _adata = sc.read_h5ad(f'{DATA_DIRECTORY}/{LAYER}/adata_slidetags.{LAYER}.{PUCK}.segmented.h5ad')
    _adata_assigned = sc.read_h5ad(f'{DATA_DIRECTORY}/{LAYER}/adata_slidetags.{LAYER}.{PUCK}.h5ad')
    _adata.obs['cell_type'] = _adata_assigned.obs.loc[_adata.obs_names, 'cell_type']
    
    _adata.obs['library_id'] = f'{LAYER}_{PUCK}'

    tree = pic.load(open(f'{DATA_DIRECTORY}/{LAYER}/slidetags_trees/reconstructions/imputed/{PUCK}/hybrid.pkl', 'rb'))
    state_to_indel = pic.load(open(f'{DATA_DIRECTORY}/{LAYER}/slidetags_trees/character_matrix/{PUCK}/character_matrix.imputed_states.pkl', 'rb'))
    
    tree.cell_meta = _adata.obs.loc[tree.leaves]
    non_tumor_cells = np.setdiff1d(tree.leaves, tree.cell_meta[tree.cell_meta['cell_type'].isin(tumor_states)].index.values)
    tree.remove_leaves_and_prune_lineages(non_tumor_cells)

    _adata.obs['tumor'] = 'False'
    _adata.obs.loc[tree.leaves, 'tumor'] = 'True'

    adata_to_merge.append(_adata)
    trees[f'{LAYER}_{PUCK}'] = (tree, state_to_indel)

adata_combined = sc.concat(adata_to_merge)

adata_combined.layers['counts'] = adata_combined.X.copy()
adata_combined.layers['logged'] = adata_combined.X.copy()

# scale_factor = np.median(np.array(adata_combined.X.sum(axis=1)))
scale_factor = 1e6
sc.pp.normalize_total(adata_combined, target_sum=scale_factor, layer='logged')

sc.pp.log1p(adata_combined, layer='logged')

fitness_trees = {}

for library_id in trees.keys():

    print(library_id)

    LAYER, PUCK = library_id.split("_")
    tree_path = f'{DATA_DIRECTORY}/{LAYER}/slidetags_trees/reconstructions/imputed/{PUCK}/hybrid.pkl'
    adata_path = f'{DATA_DIRECTORY}/{LAYER}/adata_slidetags.{LAYER}.{PUCK}.segmented.h5ad'
    
    tree = phylodynamics.score_fitness(tree_path, adata_path, None, 'lbi', 'tumor_id', None, True)

    _adata = adata_combined[adata_combined.obs['library_id'] == library_id,:]
    
    tree.cell_meta[['cell_type', 'tumor_id']] = _adata.obs.loc[tree.leaves, ['cell_type', 'tumor_id']]
    non_tumor_cells = np.setdiff1d(tree.leaves, tree.cell_meta[tree.cell_meta['cell_type'].isin(tumor_states)].index.values)
    tree.remove_leaves_and_prune_lineages(non_tumor_cells)

    fitness_trees[library_id] = tree

fitness_dfs = []

for library_id in fitness_trees.keys():

    tree = fitness_trees[library_id]
    fitness_dfs.append(tree.cell_meta['fitness'])

fitness_df = pd.DataFrame(pd.concat(fitness_dfs))
fitness_df.columns = ['fitness']
fitness_df.to_csv('./data/slidetags_fitness.tsv', sep='\t')