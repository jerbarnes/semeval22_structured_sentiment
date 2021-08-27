#!/bin/bash

EMBEDDINGDIR="../graph_parser/embeddings"

python3 convert_to_bio.py
python3 convert_to_rels.py

# Iterate over datsets
for DATASET in darmstadt_unis mpqa multibooked_ca multibooked_eu norec opener_es opener_en; do
    if [ $DATASET == norec ]; then
        EXTERNAL=$EMBEDDINGDIR/58.zip
    elif [ $DATASET == multibooked_eu ]; then
        EXTERNAL=$EMBEDDINGDIR/32.zip
    elif [ $DATASET == multibooked_ca ]; then
        EXTERNAL=$EMBEDDINGDIR/34.zip
    elif [ $DATASET == mpqa ]; then
        EXTERNAL=$EMBEDDINGDIR/18.zip
    elif [ $DATASET == darmstadt_unis ]; then
        EXTERNAL=$EMBEDDINGDIR/18.zip
    elif [ $DATASET == opener_en ]; then
        EXTERNAL=$EMBEDDINGDIR/18.zip
    elif [ $DATASET == opener_es ]; then
        EXTERNAL=$EMBEDDINGDIR/68.zip
    else
        echo "NO EMBEDDINGS SUPPLIED FOR THIS DATASET"
        echo "EXITING TRAINING PROCEDURE"
        exit
    fi

    # Train extraction models
    for ANNOTATION in sources targets expressions; do
        python3 extraction_module.py -data "$DATASET" -emb "$EXTERNAL" -ann "$ANNOTATION"
    done;

    # Train relation prediction model
    python3 relation_prediction_module.py -data "$DATASET" -emb "$EXTERNAL"

done;
