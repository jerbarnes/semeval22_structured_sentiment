#!/usr/bin/bash

PREDICTION_DATA=$1
MODELDIR=$2
EMBEDDINGS=$3

MODEL="$MODELDIR"/best_model.save

# Example usage: bash ./inference.sh sentiment_graphs/multibooked_eu/head_final/dev.conllu experiments/multibooked_eu/head_final vectors/32.zip

python3 ./src/main.py --config configs/sgraph.cfg --predict_file $PREDICTION_DATA --external $EMBEDDINGS --load $MODEL --dir $MODELDIR
