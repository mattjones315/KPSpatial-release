#!/bin/bash

## simple bash script to simulate ground truth trees for spatial imputation benchmarking
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate kp-spatial


HOMEDIR="/Users/mgjones/projects/kptc_spatial/spatial_theory/imputation_simulations/data"
SCRIPT="/Users/mgjones/projects/kptc_spatial/scripts/simulate_spatial_benchmarking.py"

INDEL_DISTRIBUTION="/Users/mgjones/projects//cassiopeia_benchmarking/full_indel_dist.pkl"

DROPOUTS=(0.1 0.25 0.5 0.6 0.7 0.9)
NUMBER_OF_TREES=10
MUTATION_PROPORTION=0.5
NUMBER_OF_STATES=100
SIZE_OF_CASSETTE=1
NUMBER_OF_CELLS=5000
SAMPLING_RATE=0.4
NUMBER_OF_CASSETTES=39

for DROPOUT in ${DROPOUTS[@]};
do
    echo ">> Simulating ${DROPOUT}..."
    python $SCRIPT ${HOMEDIR}/dropout_${DROPOUT} $INDEL_DISTRIBUTION \
        --number-of-trees ${NUMBER_OF_TREES} \
        --mutation-proportion ${MUTATION_PROPORTION} \
        --dropout-proportion ${DROPOUT} \
        --number-of-states ${NUMBER_OF_STATES} \
        --size-of-cassette ${SIZE_OF_CASSETTE} \
        --number-of-cells ${NUMBER_OF_CELLS} \
        --sampling-rate ${SAMPLING_RATE} \
        --number-of-cassettes ${NUMBER_OF_CASSETTES}

done