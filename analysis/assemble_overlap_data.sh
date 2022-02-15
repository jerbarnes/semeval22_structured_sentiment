#!/bin/sh
GOLD_BASE=../data
SUB_BASE=../submissions

rm -f assembled_overlap.json

for team in $SUB_BASE/*
do
    TEAM=$(basename "$team")
    echo Team: $TEAM
    for monoornot in $SUB_BASE/"$TEAM"/*
    do
        if ! [ -d "$monoornot" ]
        then
            continue
        fi
        CATEGORY=$(basename "$monoornot")
        if [ "$CATEGORY" = "__MAXOSX" ]
        then
            continue
        fi
        for dataset in $SUB_BASE/"$TEAM"/"$CATEGORY"/*
        do
            DATASET=$(basename "$dataset")
            python3 overlap_analysis.py "$GOLD_BASE/$DATASET/test_labeled.json" "$dataset/predictions.json" >>assembled_overlap.json
        done
    done
done
