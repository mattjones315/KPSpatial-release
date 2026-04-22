"""
A script for simulating lineage data in spatial context for benchmarking
imputation.
"""
import sys

import cassiopeia as cas
from joblib import delayed
import matplotlib.pyplot as plt
import ngs_tools
import numpy as np
import pandas as pd
import pickle as pic
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import typer

sys.path.append("/Users/mgjones/projects/kptc_spatial/KPTracer-spatial/utilities/")
from SpatialLeafSubsampler import SpatialLeafSubsampler
import tree_utilities

app = typer.Typer()


def spline_qdist(full_distribution, N):
    xeval = np.arange(len(full_distribution)) / len(full_distribution)
    spl = CubicSpline(np.arange(len(full_distribution)) / len(full_distribution), full_distribution)
    xnew = np.linspace(0, 1.0, num=N)
    return dict(zip(range(1, N+1), spl(xnew) / np.sum(spl(xnew))))


def simulate_tree(
        tree_simulator, lineage_simulator, spatial_simulator, dropout_proportion, sampling_rate
):
     
    tree = None
    while tree is None:
        try:
            tree = tree_simulator.simulate_tree()
        except:
            pass

    lineage_simulator(tree.get_mean_depth_of_tree()).overlay_data(tree)
    spatial_simulator.overlay_data(tree)
    tree = cas.sim.UniformLeafSubsampler(ratio = sampling_rate).subsample_leaves(tree)

    # overlay missing data process
    missing_tree = tree.copy()
    tree_utilities.dropout_cassettes(missing_tree, dropout_proportion)

    return tree, missing_tree


@app.command()
def simulate_trees(
    output_directory: str = typer.Argument(..., help="Location of output directory."),
    indel_distribution: str = typer.Argument(..., help="Path to computed indel distribution."),
    number_of_trees: int = typer.Option(
        10, help="Number of trees to simulate."
    ),
    mutation_proportion: float = typer.Option(
        0.4, help="Percentage of mutated states."
    ),
    dropout_proportion: float = typer.Option(
        0.0, help="Percentage of stochastic dropout to introduce."
    ),
    number_of_states: int = typer.Option(
        100, help="Number of states to simulate."
    ),
    number_of_cassettes: int = typer.Option(
        39, help="Number of cassettes to simulate."
    ),
    size_of_cassette: int = typer.Option(
        1, help="Size of cassette."
    ),
    number_of_cells: int = typer.Option(
        2000, help="Number of cells to simulate."
    ),
    sampling_rate: float = typer.Option(
        0.25, help="Sampling rate on tree."
    ),
    seed: int = typer.Option(
        None, help="Random seed."
    ),
    threads: int = typer.Option(1, help="Number of threads to use.")
):
    
    np.random.seed(seed)

    get_mutation_rate= lambda depth: -np.log2(1-mutation_proportion)/depth

    full_state_priors = pic.load(open(indel_distribution, "rb"))
    state_priors = spline_qdist(list(full_state_priors.values()), number_of_states)

    # set up simulators
    tree_simulator = cas.sim.BirthDeathFitnessSimulator(
        birth_waiting_distribution = lambda scale: np.random.lognormal(mean = np.log(scale),sigma = .5),
        initial_birth_scale = 1,
        death_waiting_distribution = lambda: np.random.uniform(0,4),
        mutation_distribution = lambda: 1,
        fitness_distribution = lambda: np.random.normal(0, .25),
        fitness_base = 1,
        num_extant = number_of_cells
    )

    lt_simulator = lambda depth: cas.simulator.Cas9LineageTracingDataSimulator(
        number_of_cassettes = number_of_cassettes,
        size_of_cassette = size_of_cassette,
        mutation_rate = get_mutation_rate(depth),
        number_of_states=number_of_states,
        state_priors=state_priors,
        heritable_silencing_rate=0.0,
        stochastic_silencing_rate=0.0,
        collapse_sites_on_cassette=True
    )

    spatial_simulator = cas.sim.ClonalSpatialDataSimulator(shape=(1,1,1))

    if threads > 1:

        trees, missing_trees = [], []
        for tree, missing_tree in ngs_tools.utils.ParallelWithProgress(
            n_jobs=threads,
            total=len(range(number_of_trees)),
            desc="Simulating trees",
            )(delayed(simulate_tree)(tree_simulator, lt_simulator, spatial_simulator, dropout_proportion, sampling_rate) for _ in range(number_of_trees)):

                trees.append(tree)
                missing_trees.append(missing_tree)

            
        for iteration, tree in zip(range(number_of_trees), trees):
        
            pic.dump(tree, open(f"{output_directory}/simulated_tree_{iteration}.pkl", 'wb'))

        for iteration, missing_tree in zip(range(number_of_trees), missing_trees):
            pic.dump(missing_tree, open(f"{output_directory}/simulated_tree_{iteration}.missing.pkl", 'wb'))


    else:

        for iteration in tqdm(range(number_of_trees), desc="Simulating trees"):
            # simulate
            tree, missing_tree = simulate_tree(tree_simulator, lt_simulator, spatial_simulator, dropout_proportion, sampling_rate)
            
            pic.dump(tree, open(f"{output_directory}/simulated_tree_{iteration}.pkl", 'wb'))
            pic.dump(missing_tree, open(f"{output_directory}/simulated_tree_{iteration}.missing.pkl", 'wb'))

if __name__ == "__main__":
    app()
