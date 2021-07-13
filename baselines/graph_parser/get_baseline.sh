#!/usr/bin/bash
# Set some random seeds that will be the same for all experiments
SEEDS=(17181920)

# Setup directories
mkdir logs
mkdir experiments

# Download word vectors
mkdir vectors
cd vectors
#wget http://vectors.nlpl.eu/repository/20/58.zip
#wget http://vectors.nlpl.eu/repository/20/32.zip
#wget http://vectors.nlpl.eu/repository/20/34.zip
#wget http://vectors.nlpl.eu/repository/20/18.zip
#wget http://vectors.nlpl.eu/repository/20/68.zip
cd ..

# Iterate over datsets
#for DATASET in darmstadt_unis mpqa multibooked_ca multibooked_eu norec opener_es opener_en; do
for DATASET in multibooked_eu; do
    mkdir logs/$DATASET;
    mkdir experiments/$DATASET;
    # Iterate over the graph setups (head_final, head_first)
    # Currently, just use head_final, but you can use others
    for SETUP in head_final; do
        mkdir experiments/$DATASET/$SETUP;
        echo "Running $DATASET - $SETUP"
        i=$(($RUN - 1))
        SEED=${SEEDS[i]}
        OUTDIR=experiments/$DATASET/$SETUP/$RUN;
        mkdir experiments/$DATASET/$SETUP/$RUN;
        # If a model is already trained, don't retrain
        if [ -f "$OUTDIR"/test.conllu.pred ]; then
            echo "$DATASET-$SETUP-$RUN already trained"
        else
            mkdir logs/$DATASET/$SETUP;
            bash ./sentgraph.sh  $DATASET $SETUP $SEED
        fi
    done;
done;
