"""
A set of scripts to perform lineage reconstruction.
"""
import sys

import ast
from functools import partial
from joblib import delayed
from multiprocessing import shared_memory
from typing import Optional, Tuple, Union
import warnings

import anndata
import cassiopeia as cas
import networkx as nx
import ngs_tools as ngs
import numpy as np
import numba
import pickle as pic
import pandas as pd
import scipy
import squidpy as sq
import typer
import tqdm

app = typer.Typer()

from cassiopeia.mixins import CassiopeiaTreeWarning
sys.path.append("/path/to/KPSpatial-release/utilities")
import target_site_utilities


@app.command()
def neighbor_joining(
    character_matrix_path: str = typer.Argument(
        ..., help="Path to character matrix."
    ),
    priors_map: str = typer.Option(None, help="Path to priors."),
    output_directory: str = typer.Option(None, help="Path to output directory"),
    distance_function: str = typer.Option(
        "cluster_dissimilarity_weighted_hamming_distance_min_linkage", help="Dissimilarity function."
    ),
    threads: int = typer.Option(
        1, help="Number of threads to use for solver."
    ),
    max_missingness: float = typer.Option(
        1.0, help="Maximum amount of missingness for a spot."
    ),
) -> Union[None, cas.data.CassiopeiaTree]:

    dissimilarity_function = None
    if distance_function == "weighted_hamming_dissimilarity":
        dissimilarity_function = (
            cas.solver.dissimilarity_functions.weighted_hamming_distance
        )
    elif distance_function == "cluster_dissimilarity_weighted_hamming_distance_min_linkage":
        dissimilarity_function = (
            cas.solver.dissimilarity_functions.cluster_dissimilarity_weighted_hamming_distance_min_linkage
        )
    else:
        raise Exception("Distance function must be one of "
            "'weighted_hamming_dissimilarity', "
            "'cluster_dissimilarity_weighted_hamming_distance_min_linkage'."
        )

    character_matrix = pd.read_csv(character_matrix_path, sep="\t", index_col=0)
    for col in character_matrix:
        character_matrix[col] = character_matrix[col].apply(
            lambda x: ast.literal_eval(str(x))
        )

    character_matrix = character_matrix[
        (
            (character_matrix == -1).sum(axis=1)
            / character_matrix.shape[1]
        )
        < max_missingness
    ]

    priors = (
        pic.load(open(priors_map, "rb")) if priors_map is not None else None
    )

    tree = cas.data.CassiopeiaTree(
        character_matrix=character_matrix, priors=priors
    )

    tree.compute_dissimilarity_map(threads=threads)

    solver = cas.solver.NeighborJoiningSolver(
        add_root=True, fast=True, dissimilarity_function=dissimilarity_function
    )
    solver.solve(tree)

    if output_directory is not None:
        output_pkl = f"{output_directory}/neighbor_joining.pkl"
        output_newick = f"{output_directory}/neighbor_joining.nwk"

        pic.dump(tree, open(output_pkl, "wb"))

        with open(output_newick, "w") as f:
            f.write(tree.get_newick())

    return tree


@app.command()
def cassiopeia_greedy(
    character_matrix_path: str = typer.Argument(
        ..., help="Path to character matrix."
    ),
    priors_map: str = typer.Option(None, help="Path to priors."),
    output_directory: str = typer.Option(None, help="Path to output directory"),
    max_missingness: float = typer.Option(
        1.0, help="Maximum amount of missingness for a spot."
    ),
) -> None:
    """Runs Cassiopeia-Greedy."""

    character_matrix = pd.read_csv(character_matrix_path, sep="\t", index_col=0)
    for col in character_matrix:
        character_matrix[col] = character_matrix[col].apply(
            lambda x: ast.literal_eval(str(x))
        )

    character_matrix = character_matrix[
        (
            (character_matrix == -1).sum(axis=1)
            / character_matrix.shape[1]
        )
        < max_missingness
    ]

    priors = (
        pic.load(open(priors_map, "rb")) if priors_map is not None else None
    )

    tree = cas.data.CassiopeiaTree(
        character_matrix=character_matrix, priors=priors
    )

    solver = cas.solver.VanillaGreedySolver()
    solver.solve(tree)
    # tree.collapse_mutationless_edges(infer_ancestral_characters=True)

    if output_directory is not None:
        output_pkl = f"{output_directory}/greedy.pkl"
        output_newick = f"{output_directory}/greedy.nwk"

        pic.dump(tree, open(output_pkl, "wb"))

        with open(output_newick, "w") as f:
            f.write(tree.get_newick())

    return tree


@app.command()
def cassiopeia_hybrid(
    character_matrix_path: str = typer.Argument(
        ..., help="Path to character matrix."
    ),
    priors_map: str = typer.Option(None, help="Path to priors."),
    output_directory: str = typer.Option(None, help="Path to output directory"),
    maximum_potential_graph_layer_size: int = typer.Option(
        10000, help="Maximum potential graph layer size during ILPSolver."
    ),
    convergence_time: int = typer.Option(
        12600, help="Time (s) for convergence of ILPSolver."
    ),
    maximum_potential_graph_lca_distance: Optional[int] = typer.Option(
        None, help="Maximum LCA distance for adding nodes in potential graph."
    ),
    threads: int = typer.Option(1, help="Number of threads to use."),
    lca_cutoff: int = typer.Option(
        None, help="LCA cutoff for determining transition to ILPSolver."
    ),
    cell_cutoff: int = typer.Option(
        200,
        help="Number of cells to use for transitioning to ILPSolver (used if lca_cutoff is not specified).",
    ),
    log_path: str = typer.Option(
        "./hybrid_solver.log", help="Stub to write out logs during ILPSolver."
    ),
    seed: int = typer.Option(None, help="Random seed."),
) -> None:
    """Runs Cassiopeia-Greedy."""

    character_matrix = pd.read_csv(character_matrix_path, sep="\t", index_col=0)
    for col in character_matrix:
        character_matrix[col] = character_matrix[col].apply(
            lambda x: ast.literal_eval(str(x))
        )

    priors = (
        pic.load(open(priors_map, "rb")) if priors_map is not None else None
    )

    weighted = priors is not None

    tree = cas.data.CassiopeiaTree(
        character_matrix=character_matrix, priors=priors
    )

    greedy_solver = cas.solver.VanillaGreedySolver()
    ilp_solver = cas.solver.ILPSolver(
        convergence_time_limit=convergence_time,
        maximum_potential_graph_layer_size=maximum_potential_graph_layer_size,
        weighted=weighted,
	seed=seed,
    )

    if lca_cutoff:
        hybrid_solver = cas.solver.HybridSolver(
            top_solver=greedy_solver,
            bottom_solver=ilp_solver,
            threads=threads,
            lca_cutoff=lca_cutoff,
        )
    else:
        hybrid_solver = cas.solver.HybridSolver(
            top_solver=greedy_solver,
            bottom_solver=ilp_solver,
            threads=threads,
            cell_cutoff=cell_cutoff,
        )

    hybrid_solver.solve(tree, logfile=log_path)
    tree.collapse_mutationless_edges(infer_ancestral_characters=True)

    if output_directory is not None:
        output_pkl = f"{output_directory}/hybrid.pkl"
        output_newick = f"{output_directory}/hybrid.nwk"

        pic.dump(tree, open(output_pkl, "wb"))

        with open(output_newick, "w") as f:
            f.write(tree.get_newick())

    return tree

@app.command()
def cassiopeia_hybrid_neighbor_joining(
    character_matrix_path: str = typer.Argument(
        ..., help="Path to character matrix."
    ),
    priors_map: str = typer.Option(None, help="Path to priors."),
    output_directory: str = typer.Option(None, help="Path to output directory"),
    threads: int = typer.Option(1, help="Number of threads to use."),
    lca_cutoff: int = typer.Option(
        None, help="LCA cutoff for determining transition to bottom solver."
    ),
    cell_cutoff: int = typer.Option(
        500,
        help="Number of cells to use for transitioning to bottom solver (used if lca_cutoff is not specified).",
    ),
    distance_function: str = typer.Option(
        "cluster_dissimilarity_weighted_hamming_distance_min_linkage", help="Dissimilarity function."
    ),
    log_path: str = typer.Option(
        "./nj_hybrid_solver.log", help="Stub to write out logs during NJ Solver."
    ),
    seed: int = typer.Option(None, help="Random seed."),
    max_missingness: float = typer.Option(
        1.0, help="Maximum amount of missingness for a spot."
    ),
) -> None:
    """Runs Cassiopeia-Hybrid with Greedy over NJ."""

    if distance_function == "weighted_hamming_dissimilarity":
        dissimilarity_function = (
            cas.solver.dissimilarity_functions.weighted_hamming_distance
        )
    elif distance_function == "cluster_dissimilarity_weighted_hamming_distance_min_linkage":
        dissimilarity_function = (
            cas.solver.dissimilarity_functions.cluster_dissimilarity_weighted_hamming_distance_min_linkage
        )
    else:
        raise Exception("Distance function must be one of "
            "'weighted_hamming_dissimilarity', "
            "'cluster_dissimilarity_weighted_hamming_distance_min_linkage'."
        )

    character_matrix = pd.read_csv(character_matrix_path, sep="\t", index_col=0)
    for col in character_matrix:
        character_matrix[col] = character_matrix[col].apply(
            lambda x: ast.literal_eval(str(x))
        )
    character_matrix = character_matrix[
        (
            (character_matrix == -1).sum(axis=1)
            / character_matrix.shape[1]
        )
        < max_missingness
    ]
    
    priors = (
        pic.load(open(priors_map, "rb")) if priors_map is not None else None
    )

    weighted = priors is not None

    tree = cas.data.CassiopeiaTree(
        character_matrix=character_matrix, priors=priors
    )

    greedy_solver = cas.solver.VanillaGreedySolver()
    nj_solver = cas.solver.NeighborJoiningSolver(add_root=True, fast=True, dissimilarity_function=dissimilarity_function)

    if lca_cutoff:
        hybrid_solver = cas.solver.HybridSolver(
            top_solver=greedy_solver,
            bottom_solver=nj_solver,
            threads=threads,
            lca_cutoff=lca_cutoff,
        )
    else:
        hybrid_solver = cas.solver.HybridSolver(
            top_solver=greedy_solver,
            bottom_solver=nj_solver,
            threads=threads,
            cell_cutoff=cell_cutoff,
        )

    hybrid_solver.solve(tree, logfile=log_path)
    # tree.collapse_mutationless_edges(infer_ancestral_characters=True)

    if output_directory is not None:
        output_pkl = f"{output_directory}/nj_hybrid.pkl"
        output_newick = f"{output_directory}/nj_hybrid.nwk"

        pic.dump(tree, open(output_pkl, "wb"))

        with open(output_newick, "w") as f:
            f.write(tree.get_newick())

    return tree


@app.command()
def create_character_matrix(
    allele_table_path: str = typer.Argument(..., help="Path to allele table."),
    anndata_path: str = typer.Argument(..., help="Path to Anndata."),
    allele_priors_path: Optional[str] = typer.Option(
        None, help="Path to allele priors."
    ),
    resolve: bool = typer.Option(
        False, help="Resolve to most abundant character state"
    ),
    collapse: bool = typer.Option(
        False, help="Collapse to unique states per spot."
    ),
    max_missing: float = typer.Option(
        0.5, help="Maximum proportion of missing states per cell."
    ),
    minimum_percent_uncut: float = typer.Option(
        0.2, help="Minimum proportion of uncut states allowed."
    ),
    minimum_spot_umi_support: int = typer.Option(
        2, help="Minimum number of UMIs necessary for a spot"
    ),
    minimum_intbc_umi_support: int = typer.Option(
        1, help="Minimum number of UMIs necessary for a given intBC."
    ),
    output_path: Optional[str] = typer.Option(
        None, help="Where to write character matrix."
    ),
    allele_rep_threshold: Optional[float] = typer.Option(
        1.0, help="Maximum allele representation threshold per intBC."
    ),
    impute: bool = typer.Option(
        False, help="Impute missing states based on spatial proximity."
    ),
    neighborhood_size: int = typer.Option(
        10, help="Spatial neighborhood size for imputation."
    ),
    neighborhood_radius: float = typer.Option(
        30.0, help='Spatial neighborood radius for imputation.'
    ),
    imputation_hops: int = typer.Option(
        2, help='Number of hops for neighborhood imputation.'
    ),
    imputation_concordance: float = typer.Option(
        0.7, help="Minimum agreement for imputation.",
    ),
    num_imputation_iterations: int = typer.Option(
        1, help="Number of iterations for imputation."
    ),
) -> Union[None, Tuple[pd.DataFrame, dict]]:
    """Creates a character matrix from an allele table."""

    allele_priors = pd.read_csv(allele_priors_path, sep="\t", index_col=0)
    allele_table = pd.read_csv(allele_table_path, sep="\t")

    allele_table = allele_table[
        ~allele_table[["r1", "r2", "r3"]].isna().any(axis=1)
    ]
    if resolve:
        allele_table = allele_table.sort_values(
            ["UMI", "readCount"], ascending=False
        ).drop_duplicates(["cellBC", "intBC"])

    adata = anndata.read(anndata_path)
    print(
        f">> Filtering out spots with fewer than {minimum_percent_uncut}% "
        "uncut sites and {minimum_spot_umi_support} UMIs."
    )
    overlapping_cells = np.intersect1d(
        adata[
            (adata.obs["PercentUncut"] <= minimum_percent_uncut)
            & (adata.obs["TS-UMI"] >= minimum_spot_umi_support)
        ].obs_names,
        allele_table["cellBC"].values,
    )

    print(
        f">> Filtering out intBCs with fewer than {minimum_intbc_umi_support}"
        " UMIs."
    )
    pre_filtered_cells = np.intersect1d(adata.obs_names, allele_table["cellBC"].unique())
    overlapping_cells = np.intersect1d(
        adata.obs_names,
        allele_table[allele_table["UMI"] >= minimum_intbc_umi_support]["cellBC"].unique(),
    )
    allele_table = allele_table[allele_table["cellBC"].isin(overlapping_cells)]

    (
        character_matrix,
        prior_probs,
        indel_to_charstate,
    ) = cas.pp.convert_alleletable_to_character_matrix(
        allele_table,
        mutation_priors=allele_priors,
        collapse_duplicates=collapse,
        allele_rep_thresh=allele_rep_threshold,
    )

    if impute:
        # add in cells that did not have high-confidence target sites
        missing_cells = np.setdiff1d(pre_filtered_cells, character_matrix.index)
        print(f">> Adding back in {len(missing_cells)} cells for imputation.")
        missing_cell_character_matrix = pd.DataFrame(
            np.zeros(
                (len(missing_cells), character_matrix.shape[1]), dtype=int
            ),
            index=missing_cells,
            columns=character_matrix.columns,
        )
        missing_cell_character_matrix.to_numpy().fill(-1)
        character_matrix = pd.concat(
            [character_matrix, missing_cell_character_matrix]
        )

        if neighborhood_radius:
            sq.gr.spatial_neighbors(
                adata,
                coord_type="generic",
                spatial_key="spatial",
                radius=neighborhood_radius,
            )
        else:
            sq.gr.spatial_neighbors(
                adata,
                coord_type="generic",
                spatial_key="spatial",
                n_neighs=neighborhood_size,
            )

        spatial_graph = nx.from_numpy_array(
            adata.obsp["spatial_connectivities"]
        )
        node_map = dict(
            zip(
                range(adata.obsp["spatial_connectivities"].shape[0]),
                adata.obs_names,
            )
        )
        spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

        prev_character_matrix_imputed = character_matrix.copy()
        missing_indices = np.where(character_matrix == -1)

        for _round in range(num_imputation_iterations):
            character_matrix_imputed = prev_character_matrix_imputed.copy()
            missing_indices = np.where(prev_character_matrix_imputed == -1)

            for i, j in tqdm.tqdm(
                zip(missing_indices[0], missing_indices[1]),
                total=len(missing_indices[0]),
            ):
                (
                    imputed_value,
                    prop_votes,
                ) = target_site_utilities.impute_single_state(
                    prev_character_matrix_imputed.index.values[i],
                    j,
                    prev_character_matrix_imputed,
                    neighborhood_graph=spatial_graph,
                    number_of_hops=1,
                )
                if (
                    prop_votes >= imputation_concordance
                    and imputed_value != -1
                    and imputed_value != 0
                ):
                    character_matrix_imputed.iloc[i, j] = imputed_value

            prev_character_matrix_imputed = character_matrix_imputed.copy()

            print(f">> Imputation round {_round+1}:")
            print(
                f"Character matrix has {character_matrix_imputed.drop_duplicates().shape[0]} unique states."
            )
            print(
                f"Character matrix has {round(((character_matrix_imputed == -1).sum(axis=1)).sum() / (character_matrix_imputed.shape[1] * character_matrix_imputed.shape[0]),2)}% missing data"
            )
            print(
                f"Found {character_matrix_imputed[character_matrix_imputed.apply(lambda x: np.all(x == -1), axis=1)].shape[0]} examples with all missing data."
            )
            print("\n")
    else:
        character_matrix_imputed = character_matrix.copy()

    # apply final missingness filter
    final_character_matrix = character_matrix_imputed[
        (
            (character_matrix_imputed == -1).sum(axis=1)
            / character_matrix_imputed.shape[1]
        )
        < max_missing
    ]

    if output_path:
        priors_output = ".".join(output_path.split(".")[:-1]) + "_priors.pkl"
        states_output = '.'.join(output_path.split(".")[:-1]) + "_states.pkl"

        final_character_matrix.to_csv(output_path, sep="\t")
        pic.dump(prior_probs, open(priors_output, "wb"))
        pic.dump(indel_to_charstate, open(states_output, 'wb'))
    else:
        return final_character_matrix, prior_probs, indel_to_charstate


if __name__ == "__main__":
    app()
