"""
A script for assessing imputation accuracy on semi-synthetic (held out) data
from real spatial transcriptomics datasets.
"""

import os
import sys

from typing import List, Tuple, Optional

import anndata
import cassiopeia as cas
import itertools
from joblib import delayed
import ngs_tools
import networkx as nx
import numpy as np
import pandas as pd
import pickle as pic
from scipy.spatial import cKDTree
import squidpy as sq
from tqdm import tqdm
import typer

sys.path.append(
    "/path/to/KPSpatial-release/utilities/"
)
import target_site_utilities

app = typer.Typer()


def gather_files(
    home_directory,
    sample_file_list,
):

    samples = []
    with open(sample_file_list, "r") as f:

        for line in f:

            sample_name = line.strip()
            adata_file = (
                f"{home_directory}/{sample_name}_RNA/adata_merged.filtered.destvi.segmented_manual.h5ad"
            )
            allele_table_file = f'{home_directory}/{sample_name}_TS/umi_2/{sample_name.split("/")[1]}.call_lineages.txt'

            samples.append(
                (sample_name.split("/")[1], allele_table_file, adata_file)
            )

    return samples


def split_into_batches(
    sample_list: List[Tuple[str, str, str]], number_of_threads: int
) -> List[List[Tuple[int, float]]]:
    """Splits tasks into batches.

    Using the specified number of threads, create approximately evenly sized
    batches of reconstructions to score.

    Args:
        sample_list: List of samples to process.
        number_of_threads: Number of threads to utilize.

    Returns:
        A list of batches of reconstructions.
    """

    k, m = divmod(len(sample_list), number_of_threads)
    batches = [
        sample_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(number_of_threads)
    ]
    return batches


def impute(
    adata,
    character_matrix,
    num_imputation_iterations=1,
    imputation_concordance=0.8,
    min_votes=5,
    randomize=False,
):

    spatial_graph = nx.from_numpy_array(adata.obsp["spatial_connectivities"])
    barcodes = adata.obs_names
    if randomize:
        barcodes = np.random.choice(barcodes, size=len(barcodes), replace=False)

    node_map = dict(
        zip(
            range(adata.obsp["spatial_connectivities"].shape[0]),
            barcodes,
        )
    )
    spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

    prev_character_matrix_imputed = character_matrix.copy()
    missing_indices = np.where(character_matrix == -1)

    for _round in range(num_imputation_iterations):
        character_matrix_imputed = prev_character_matrix_imputed.copy()
        missing_indices = np.where(prev_character_matrix_imputed == -1)

        for i, j in zip(missing_indices[0], missing_indices[1]):

            (
                imputed_value,
                prop_votes,
                n_votes,
            ) = target_site_utilities.impute_single_state(
                prev_character_matrix_imputed.index.values[i],
                j,
                prev_character_matrix_imputed,
                neighborhood_graph=spatial_graph,
                number_of_hops=1,
            )
            if (
                prop_votes >= imputation_concordance
                and n_votes >= min_votes
                and imputed_value != -1
                and imputed_value != 0
            ):
                character_matrix_imputed.iloc[i, j] = imputed_value

        prev_character_matrix_imputed = character_matrix_imputed.copy()

    return character_matrix_imputed


def assess_accuracy_of_sample(
    character_matrix,
    priors_map,
    indel_to_charstate,
    adata,
    sample_name,
    mask_proportion,
    number_of_replicates,
    number_of_iterations_array,
    imputation_concordance=0.8,
    randomize: bool = False,
):
    imputation_log = pd.DataFrame(
        columns=[
            "SampleName",
            "Replicate",
            "GroundTruthState",
            "ImputedState",
            "GroundTruthAllele",
            "ImputedAllele",
            "MarkedCorrect",
            "Prior",
            "AlleleFreq",
            "VoteConcordance",
            "NumIterations",
        ]
    )
    prop_votes = np.nan
    for number_of_iterations in number_of_iterations_array:
        for replicate in range(number_of_replicates):
            present_indices = np.argwhere(
                np.isnan(
                    character_matrix[character_matrix == -1].values.astype(
                        float
                    )
                )
            )
            masked_cells = np.random.choice(
                np.arange(len(present_indices)),
                int(mask_proportion * len(present_indices)),
            )

            cm_masked = character_matrix.copy()
            for m_index in masked_cells:
                masked_cell = present_indices[m_index]
                cm_masked.iloc[masked_cell[0], masked_cell[1]] = -1

            imputed_character_matrix = impute(
                adata,
                cm_masked,
                num_imputation_iterations=number_of_iterations,
                imputation_concordance=imputation_concordance,
                randomize=randomize,
            )

            accur = 0
            imputed = 0
            prop_missing = 0
            for m_index in masked_cells:
                masked_cell = present_indices[m_index]
                true_value = character_matrix.iloc[
                    masked_cell[0], masked_cell[1]
                ]
                imputed_value = imputed_character_matrix.iloc[
                    masked_cell[0], masked_cell[1]
                ]

                correct = "False"
                if imputed_value != -1:
                    imputed += 1
                else:
                    prop_missing += 1

                if imputed_value == -1:
                    continue

                if (
                    type(true_value) == tuple and imputed_value in true_value
                ) or (type(true_value) == int and true_value == imputed_value):
                    correct = "True"
                    accur += 1

                if imputed_value == 0:
                    prior = np.nan
                else:
                    prior = priors_map[masked_cell[1]][imputed_value]

                if imputed_value == 0:
                    imputed_allele = "None"
                else:
                    imputed_allele = indel_to_charstate[masked_cell[1]][
                        imputed_value
                    ]

                true_allele = []
                if type(true_value) == tuple:
                    for state in true_value:
                        if state == 0:
                            true_allele.append("None")
                        else:
                            true_allele.append(
                                indel_to_charstate[masked_cell[1]][state]
                            )
                else:
                    if true_value == 0:
                        true_allele.append("None")
                    else:
                        true_allele.append(
                            indel_to_charstate[masked_cell[1]][true_value]
                        )

                allele_frequency = len(character_matrix[character_matrix.iloc[:,masked_cell[1]] == true_value]) / len(character_matrix)
                new_rows = pd.DataFrame(
                    [
                        [
                            sample_name,
                            replicate,
                            true_value,
                            imputed_value,
                            true_allele,
                            imputed_allele,
                            correct,
                            prior,
                            allele_frequency,
                            prop_votes,
                            number_of_iterations,
                        ]
                    ],
                    columns=imputation_log.columns,
                )
                imputation_log = pd.concat([imputation_log, new_rows])

    return imputation_log


def assess_accuracy_in_batch(
    sample_files: List[Tuple[str, str, str]],
    allele_priors: Optional[pd.DataFrame] = None,
    masking_proportion: float = 0.1,
    number_of_replicates: int = 5,
    number_of_iterations: List[int] = [1, 2, 3],
    radius: float = 30,
    imputation_concordance: float = 0.8,
    min_votes: int = 5,
    min_ts_umi: int = 5,
    min_intbc_umi_support: int = 3,
    randomize: bool = False
):
    summaries = []
    for sample_name, allele_table_file, anndata_file in sample_files:
        adata = anndata.read_h5ad(anndata_file)
        allele_table = pd.read_csv(allele_table_file, sep="\t")

        adata = adata[adata.obs['tumor_id'] != 'non-tumor'] # filter out non-tumor cells

        allele_table = allele_table[
            ~allele_table[["r1", "r2", "r3"]].isna().any(axis=1)
        ]

        # remove cells from allele table that don't appear in Anndata
        overlapping_cells = np.intersect1d(
            adata[
                (adata.obs["PercentUncut"] <= 0.8)
                & (adata.obs["TS-UMI"] >= min_ts_umi)
            ].obs_names,
            allele_table["cellBC"].values,
        )

        allele_table = allele_table[
            allele_table["cellBC"].isin(overlapping_cells)
        ]

        overlapping_cells = np.intersect1d(
            adata.obs_names,
            allele_table[allele_table["UMI"] >= min_intbc_umi_support]["cellBC"].unique(),
        )

        (
            character_matrix,
            priors_map,
            indel_to_charstate,
        ) = cas.pp.convert_alleletable_to_character_matrix(
            allele_table,
            mutation_priors=allele_priors,
            collapse_duplicates=False,
        )

        sq.gr.spatial_neighbors(
            adata, coord_type="generic", spatial_key="spatial", radius=radius
        )

        summary_dataframe = assess_accuracy_of_sample(
            character_matrix,
            priors_map,
            indel_to_charstate,
            adata,
            sample_name,
            masking_proportion,
            number_of_replicates,
            number_of_iterations,
            imputation_concordance=imputation_concordance,
            randomize=randomize,
        )
        summaries.append(summary_dataframe)

    return pd.concat(summaries)


@app.command()
def evaluate_imputation_accuracy(
    home_directory: str = typer.Argument(
        ...,
        help="Home directory path. Should be the root directory for the samples listed in the sample list file argument.",
    ),
    sample_list_file: str = typer.Argument(
        ..., help="Text file containing paths to samples to process."
    ),
    output_file: str = typer.Argument(..., help="Where to write output."),
    allele_priors_file: str = typer.Option(
        None, help="File path to allele priors."
    ),
    masking_proportion: float = typer.Option(
        0.1, help="Proportion of observed states to mask out."
    ),
    number_of_replicates: int = typer.Option(
        5, help="Number of replicates to simulate."
    ),
    radius: float = typer.Option(28, help="Neighborhood radius for imputation."),
    max_iterations: int = typer.Option(
        5, help="Number of imputation iterations to look at."
    ),
    threads: int = typer.Option(1, help="Number of threads to use."),
    imputation_concordance: float = typer.Option(0.8, help='Minimum agreement for imputation.'),
    min_votes: int = typer.Option(5, help="Minimum number of votes in agreement."),
    min_ts_umi: int = typer.Option(5, help="Minimum TS UMI support per spot."),
    min_intbc_umi_support: int = typer.Option(2, help='Minimum TS UMI support per intBC'),
    randomize: bool = typer.Option(False, help="Compute random baseline."),
):
    imputation_log = pd.DataFrame(
        columns=[
            "SampleName",
            "Replicate",
            "GroundTruthState",
            "ImputedState",
            "GroundTruthAllele",
            "ImputedAllele",
            "MarkedCorrect",
            "Prior",
            "AlleleFreq",
            "VoteConcordance",
            "NumHops",
        ]
    )

    # gather all test examples
    sample_files = gather_files(home_directory, sample_list_file)
    batches = split_into_batches(sample_files, threads)

    number_of_iterations = list(range(1, max_iterations + 1))
    allele_priors = None

    if allele_priors_file:
        allele_priors = pd.read_csv(allele_priors_file, sep="\t", index_col=0)

    print("Setting off batches of size:")
    for batch_i, batch in zip(range(len(batches)), batches):
        print(f"Batch-{batch_i}: {len(batch)}")

    for new_rows in ngs_tools.utils.ParallelWithProgress(
        n_jobs=threads,
        total=len(batches),
        desc="Processing batches",
    )(
        delayed(assess_accuracy_in_batch)(
            batch,
            allele_priors,
            masking_proportion,
            number_of_replicates,
            number_of_iterations,
            radius,
            imputation_concordance,
            min_votes,
            min_ts_umi,
            min_intbc_umi_support,
            randomize
        )
        for batch in batches
    ):
        imputation_log = pd.concat([imputation_log, new_rows])

    imputation_log.to_csv(output_file, sep="\t")


if __name__ == "__main__":
    app()
