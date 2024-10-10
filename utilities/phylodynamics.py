"""
A set of functions for phylodynamic analyses.
"""
import sys

from typing import Union

import cassiopeia as cas
import numpy as np
import pandas as pd
import pickle as pic
import scanpy as sc
import typer

app = typer.Typer()


@app.command()
def score_fitness(
    tree_path: str = typer.Argument(
        ..., help="Path to reconstructed tree."
    ),
    adata_path: str = typer.Option(None, help="Path to anndata."),
    output_directory: str = typer.Option(None, help="Path to output directory"),
    fitness_algorithm: str = "lbi",
    grouping_var = typer.Option(None, help='Group by variable.'),
    focus_group = typer.Option(None, help="Specific group variable to focus on."),
    infer_branch_lengths: bool = typer.Option(False, help="Run maximum likelihood branch length estimator.")
) -> Union[None, cas.data.CassiopeiaTree]:
    
    def _fitness_wrapper(_tree, _infer_branch_lengths):

        # infer branch lengths
        _tree.reconstruct_ancestral_characters()
        _tree.set_character_states(_tree.root, [0] * _tree.n_character)
        print('infer branch lengths: ', _infer_branch_lengths)

        if _infer_branch_lengths is True:
            print('inferring branch lengths')
            branch_length_estimator = cas.tools.branch_length_estimator.IIDExponentialMLE(minimum_branch_length=0.01)
            branch_length_estimator.estimate_branch_lengths(_tree) 
        else:
            
            for edge in _tree.depth_first_traverse_edges():
                branch_length = len(_tree.get_mutations_along_edge(edge[0], edge[1]))
                _tree.set_branch_length(edge[0], edge[1], min(1, branch_length))
                
        # score fitness
        if fitness_algorithm == 'lbi':
            fitness_estimator = cas.tools.fitness_estimator.LBIJungle()
        else:
            raise Exception("Fitness algorithm not recognized. "
                            "Choose one of: 'lbi'")
        fitness_estimator.estimate_fitness(_tree)

        # renormalize
        fitnesses = np.array([_tree.get_attribute(cell, 'fitness') for cell in _tree.leaves])
        fitnesses /= np.max(fitnesses)
        return pd.DataFrame(fitnesses, index=_tree.leaves, columns=['fitness'])


    tree = pic.load(open(tree_path, 'rb'))

    if adata_path:
        adata = sc.read_h5ad(adata_path)

        overlapping_cells = np.intersect1d(adata.obs_names, tree.leaves)
        adata = adata[overlapping_cells,:]    

    if grouping_var:
        
        if focus_group:
            print(f"Subsetting down to {focus_group}...")
            adata = adata[adata.obs[grouping_var] == focus_group]

        to_merge = []
        print(f"Scoring fitness by {grouping_var}...")
        for sample_name, group in adata.obs.groupby(grouping_var):

            if sample_name == 'non-tumor':
                continue
            print(f"Scoring fitness in {sample_name}...")

            query_cells = group.index.values
            subtree = tree.copy()

            subtree.remove_leaves_and_prune_lineages(np.setdiff1d(query_cells, tree.leaves))
            subtree.collapse_mutationless_edges(infer_ancestral_characters=True)
            _fitnesses = _fitness_wrapper(subtree, infer_branch_lengths)
            to_merge.append(_fitnesses)
        
        fitnesses = pd.concat(to_merge)
    else:
        fitnesses = _fitness_wrapper(tree, infer_branch_lengths)
    

    tree.cell_meta = pd.DataFrame(index=tree.leaves)
    tree.cell_meta['fitness'] = np.nan
    tree.cell_meta.loc[fitnesses.index.values, 'fitness'] = fitnesses['fitness'].values

    if output_directory is not None:
        output_stem = tree_path.split("/")[-1].split('.pkl')[0]
        output_pickle = f'{output_directory}/{output_stem}.fitness.pkl'

        if focus_group:
            output_pickle = f'{output_directory}/{output_stem}.{focus_group}.fitness.pkl'

        if not infer_branch_lengths:
            # output_pickle = f'{output_directory}/{output_stem}.fitness.naive_bl.pkl'
            output_pickle = f'{output_pickle.split(".fitness.pkl")[0]}.{focus_group}.fitness.naive_bl.pkl'
            
        print(f"Writing output to {output_pickle}")
        pic.dump(tree, open(output_pickle, "wb"))

    else:
        return tree


if __name__ == "__main__":
    app()
