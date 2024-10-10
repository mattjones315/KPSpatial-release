"""
A set of utilities for manipulating target site information.
"""
import sys
from typing import Optional

import anndata
import cassiopeia as cas
import numpy as np
import networkx as nx
import pandas as pd

sys.path.append(
    "/path/to/KPSpatial-release/utilities/"
)
import spatial_utilities


def create_target_site_meta(allele_table: pd.DataFrame) -> pd.DataFrame:
    """Creates meta data from target site allele table.

    Args:
        allele_table: Allele table table processed with Cassiopeia.

    Returns:
        A pandas DataFrame with target site meta data.
    """

    def compute_uncut_fraction(allele_group: pd.DataFrame) -> float:
        """Helper function for computing uncut fractions of spots.

        Args:
            allele_group: DataFrame of alleles in an allele table.

        Returns:
            A fraction of uncut sites.
        """
        allele_group = allele_group.dropna()
        uncut = [x.count("None") for x in allele_group]
        sum_uncuts = np.sum(uncut)
        uncut_fraction = sum_uncuts / (len(allele_group) * 3)
        return uncut_fraction

    allele_table_copy = allele_table.copy()

    allele_table_copy["intbc-allele"] = allele_table_copy.apply(
        lambda x: x.intBC + "-" + str(x.allele), axis=1
    )

    cellbc_ts_meta = allele_table_copy.groupby("cellBC").agg(
        {
            "UMI": "sum",
            "intBC": "nunique",
            "readCount": "sum",
            "intbc-allele": "nunique",
            "allele": compute_uncut_fraction,
        }
    )
    cellbc_ts_meta.columns = [
        "TS-UMI",
        "N_intBC",
        "TS-ReadCount",
        "N_unique_alleles",
        "PercentUncut",
    ]

    return cellbc_ts_meta


def impute_states(
    cell: str,
    character_matrix: pd.DataFrame,
    adata: anndata.AnnData,
    number_of_hops: int = 1,
    max_neighbor_distance: float = np.inf,
) -> np.array:
    """Imputes missing character states for a cell.

    Args:
        cell: Cell barcode
        character_matrix: Character matrix of all character states
        adata: Anndata object with spatial nearest neighbors
        number_of_hops: Number of hops to make during imputation.
        max_neighbor_distance: Maximum distance to neighbor to be used for
            imputation.
    Returns:
        A character state array.
    """

    neighborhood_graph = spatial_utilities.get_spatial_neighborhood_graph(
        cell, adata, number_of_hops
    )

    character_states = character_matrix.loc[cell]
    new_character_states = character_states.copy()
    for character in np.where(character_states == -1)[0]:
        votes = []
        for _, node in nx.bfs_edges(neighborhood_graph, cell):
            if node not in character_matrix.index:
                continue

            distance = nx.shortest_path_length(
                neighborhood_graph, cell, node, weight="distance"
            )
            state = character_matrix.loc[node][character]
            if distance <= max_neighbor_distance and state != -1:
                votes.append(state)

        if len(votes) > 0:
            values, counts = np.unique(votes, return_counts=True)
            new_character_states[character] = values[np.argmax(counts)]

    return new_character_states


def impute_single_state(
    cell: str,
    character: int,
    character_matrix: pd.DataFrame,
    adata: Optional[anndata.AnnData] = None,
    number_of_hops: int = 1,
    neighborhood_graph: Optional[nx.DiGraph] = None,
    max_neighbor_distance: float = np.inf,
) -> np.array:
    """Imputes missing character state for a cell at a defined position.

    Args:
        cell: Cell barcode
        character: Which character to impute.
        character_matrix: Character matrix of all character states
        adata: Anndata object with spatial nearest neighbors
        number_of_hops: Number of hops to make during imputation.
        max_neighbor_distance: Maximum distance to neighbor to be used for
            imputation.
    Returns:
        A character state array.
    """

    if neighborhood_graph is None:
        if adata is None:
            raise Exception("Must pass in adata if neighborhood graph not specified.")
        neighborhood_graph = spatial_utilities.get_spatial_neighborhood_graph(
            cell, adata, number_of_hops
        )

    votes = []
    for _, node in nx.bfs_edges(neighborhood_graph, cell, depth_limit=number_of_hops):
        if node not in character_matrix.index:
            continue

        distance = nx.shortest_path_length(
            neighborhood_graph, cell, node, weight="distance"
        )
        state = character_matrix.loc[node].iloc[character]
        # state = character_matrix.loc[node, character]
        if distance <= max_neighbor_distance and state != -1:
            if type(state) == tuple:
                for _state in state:
                    votes.append(_state)
            else:
                votes.append(state)

    if len(votes) > 0:
        values, counts = np.unique(votes, return_counts=True)
        return values[np.argmax(counts)], np.max(counts) / np.sum(counts), np.max(counts)

    return -1, 0, 0
