#!/bin/bash
# Set some random seeds that will be the same for all experiments
SEEDS=(17181920)

# Setup directories
mkdir logs
mkdir experiments

# # Convert json files to conllu for training
# # Currently only creates head_final, but you can
# # experiment with other graph setups by expanding this section
for DATASET in darmstadt_unis mpqa multibooked_ca multibooked_eu norec opener_es opener_en; do
    python3 convert_to_conllu.py --json_dir ../../data/"$DATASET" --out_dir sentiment_graphs/"$DATASET" --setup head_final
done;

# Download word vectors
if [ -d embeddings ]; then
    echo "Using downloaded word embeddings"
else
    mkdir embeddings
    cd embeddings
    wget http://vectors.nlpl.eu/repository/20/58.zip
    wget http://vectors.nlpl.eu/repository/20/32.zip
    wget http://vectors.nlpl.eu/repository/20/34.zip
    wget http://vectors.nlpl.eu/repository/20/18.zip
    wget http://vectors.nlpl.eu/repository/20/68.zip
cd ..
fi

# Iterate over datsets
for DATASET in darmstadt_unis mpqa multibooked_ca multibooked_eu norec opener_es opener_en; do
    mkdir logs/$DATASET;
    mkdir experiments/$DATASET;
    # Iterate over the graph setups (head_final, head_first, head_final-inside_label, head_final-inside_label-dep_edges, head_final-inside_label-dep_edges-dep_labels, etc)
    # Currently, just use head_final, but you can use others
    for SETUP in head_final; do
        mkdir experiments/$DATASET/$SETUP;
        echo "Running $DATASET - $SETUP"
        SEED=${SEEDS[0]}
        OUTDIR=experiments/$DATASET/$SETUP;
        mkdir experiments/$DATASET/$SETUP;
        # If a model is already trained, don't retrain
        if [ -f "$OUTDIR"/test.conllu.pred ]; then
            echo "$DATASET-$SETUP already trained"
        else
            mkdir logs/$DATASET/$SETUP;
            LOGFILE=logs/$DATASET/$SETUP/log.txt
            bash ./sentgraph.sh  $DATASET $SETUP $SEED > $LOGFILE
        fi
    done;
done;
