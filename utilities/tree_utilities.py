"""
General tools for handling trees.
"""

import numpy as np
import pandas as pd
import cassiopeia as cas
import networkx as nx
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm


def edit_frac(tree):
    return np.float64(np.apply_over_axes(np.sum,tree.character_matrix != 0,(0,1))/
                      np.apply_over_axes(np.sum,~np.isnan(tree.character_matrix),(0,1)))

def cut_tree_at_time(tree, node, time, nodes):
    if tree.nodes[node]["time"] > time:
        nodes.append(node)
    else:
        for child in tree.successors(node):
            cut_tree_at_time(tree, child, time, nodes)
    return nodes

def barcode_tree(tree, barcode, time):
    nodes = cut_tree_at_time(tree.get_tree_topology(),tree.root,time,[])
    groups = []
    barcode_iter = 1
    for node in nodes:
        cells = tree.leaves_in_subtree(node)
        groups.append(pd.DataFrame({'id': cells, barcode: str(barcode_iter)}, index=range(len(cells))))
        barcode_iter += 1
    groups = pd.concat(groups).set_index('id')
    return groups

def character_matrix_to_allele_table(character_matrix, state_to_indel=None, keep_ambiguous=True):
    allele_table = pd.DataFrame(columns=['cellBC', 'intBC', 'allele', 'r1', 'UMI'])
    def disambiguate_allele(char, allele):
        if type(allele) == tuple:
            if keep_ambiguous:
                all_alleles = [state_to_indel[int(char)][a] if a != 0 else "None" for a in allele]
                return all_alleles
            else:
                allele = allele[0]
        if allele == 0:
            return ['None']
        if state_to_indel:
            return [state_to_indel[int(char)][allele]]
        return [allele]

    for cell in tqdm(character_matrix.index):
        alleles = character_matrix.loc[cell].values
        non_missing_iid = np.where(alleles != -1)[0]
        intbcs = []
        all_alleles = []
        for iid in non_missing_iid:
            _alleles = disambiguate_allele(iid, alleles[iid])
            intbcs += [f'intbc-{iid}' for _ in range(len(_alleles))]
            all_alleles += _alleles
        cellbcs = [cell] * len(intbcs)
        umis = [1]*len(intbcs)
        new_rows = pd.DataFrame([cellbcs, intbcs, all_alleles, all_alleles, umis], index = allele_table.columns).T
        allele_table = pd.concat([allele_table, new_rows])
    return allele_table


def get_meta_colormap(tree,meta_data,missing = 0):
    values = [missing]
    for item in meta_data:
        values.extend(tree.cell_meta[item].unique().tolist())
    values = [missing] + list(set(values))
    pal = sns.color_palette('hls', n_colors=len(values)-1)
    pal.insert(0, "#FFFFFF")
    cmap = ListedColormap(pal)
    value_map = {values[i]: i for i in range(len(values))}
    value_map[missing] = 0
    return cmap, value_map

def make_character_matrix(clone_allele_table ,edits):
    character_matrix = pd.melt(clone_allele_table[['cellBC','intBC', 'r1', 'r2', 'r3']],id_vars = ["cellBC","intBC"],var_name = "site",value_name = "allele")
    sites = character_matrix.groupby(["intBC","site"]).size().reset_index(drop = False).drop(0,axis = 1)
    sites["site_id"] = "r" + (sites.index + 1).astype(str)
    character_matrix = pd.merge(character_matrix,edits,how = "left",left_on = "allele",right_on = "allele")
    character_matrix = pd.merge(character_matrix,sites,how = "left",left_on = ["intBC","site"],right_on = ["intBC","site"])
    character_matrix = character_matrix[~character_matrix["id"].isna()]
    character_matrix = character_matrix.pivot(index = "cellBC",columns = "site_id",values = "id")
    character_matrix = character_matrix.fillna(-1).astype(int).loc[:,sites.site_id]
    return character_matrix, sites

def dfs_order_leaves(tree):
    leaves = []
    nx_tree = tree.get_tree_topology()
    for node in nx.dfs_preorder_nodes(nx_tree, source=tree.root):
        if node in tree.leaves:
            leaves.append(node)
    return leaves

def get_leaf_coordinates(tree,attribute_key = "spatial"):
    coordinates = []
    for leaf in tree.leaves:
        coordinates.append(tree.get_attribute(leaf,attribute_key))
    if len(coordinates[0]) == 2:
        coordinates = pd.DataFrame(coordinates, columns=['x', 'y'],index = tree.leaves)
    elif len(coordinates[0]) == 3:
        coordinates = pd.DataFrame(coordinates, columns=['x', 'y', 'z'],index = tree.leaves)
    return coordinates

def dropout_cassettes(tree, missing_rate = .3, missing_state = -1, cassette_size=3):
    character_matrix = tree.character_matrix
    missing = np.random.choice([0,1],p=[1-missing_rate,missing_rate],
                    size=(character_matrix.shape[0],int(character_matrix.shape[1]/cassette_size)))
    missing = np.repeat(missing,3,axis=1)
    character_matrix[missing == 1] = missing_state
    tree.character_matrix = character_matrix
    tree.set_character_states_at_leaves()

def cleanup_character_matrix_and_collapse(tree, downsample=None):
    # clean up for visualization
    character_matrix_cleaned = tree.character_matrix.copy()
    tree.reconstruct_ancestral_characters()
    n_characters_cleaned = 0
    for cell in character_matrix_cleaned.index.values:
        states = character_matrix_cleaned.loc[cell].values.copy()
        new_states = states
        for character, state in zip(range(character_matrix_cleaned.shape[0]), states):
            if cas.mixins.is_ambiguous_state(state):
                parent = tree.parent(cell)
                parent_state = tree.get_character_states(parent)[character]
                # find first parent state that is not ambiguous
                while cas.mixins.is_ambiguous_state(parent_state) and parent != tree.root:
                    parent = tree.parent(parent)
                    parent_state = tree.get_character_states(parent)[character]
                # if parent state is uncut, take cut ambig state
                if parent_state == 0:
                    if len(np.unique(cas.mixins.unravel_ambiguous_states(state))) > 1:
                        new_states[character] = [s for s in cas.mixins.unravel_ambiguous_states(state) if s != 0][0]
                    else:
                        new_states[character] = state
                else:
                    new_states[character] = parent_state
                n_characters_cleaned += 1
        character_matrix_cleaned.loc[cell,:]  = new_states
    
    print(f'Cleaned up {n_characters_cleaned} ambiguous characters.')

    tree_cleaned = tree.copy()
    tree_cleaned.set_character_states_at_leaves(character_matrix_cleaned)

    # downsample if needed
    if downsample:
        kii = np.random.choice(tree_cleaned.leaves, size=min(downsample,len(tree_cleaned.leaves)) , replace=False)
        to_drop = np.setdiff1d(tree_cleaned.leaves, kii)
        tree_cleaned.remove_leaves_and_prune_lineages(to_drop)
    
    tree_cleaned.collapse_mutationless_edges(infer_ancestral_characters=True)
    tree_cleaned.collapse_unifurcations()
    return tree_cleaned
