#!/bin/bash

DATASET=$1;
SETUP=$2;
SEED=$3;


# EXTERNAL EMBEDDINGS
############################
echo "###########################"
echo EXTERNAL EMBEDDINGS
echo "###########################"

if [ $DATASET == norec ]; then
    EXTERNAL=vectors/58.zip
elif [ $DATASET == multibooked_eu ]; then
    EXTERNAL=vectors/32.zip
elif [ $DATASET == multibooked_ca ]; then
    EXTERNAL=vectors/34.zip
elif [ $DATASET == mpqa ]; then
    EXTERNAL=vectors/18.zip
elif [ $DATASET == darmstadt_unis ]; then
    EXTERNAL=vectors/18.zip
elif [ $DATASET == opener_en ]; then
    EXTERNAL=vectors/18.zip
elif [ $DATASET == opener_es ]; then
    EXTERNAL=vectors/68.zip
else
    echo "NO EMBEDDINGS SUPPLIED FOR THIS DATASET"
    echo "EXITING TRAINING PROCEDURE"
    exit
fi

echo using external vectors: $EXTERNAL
echo

# INPUT FILES
############################
echo "###########################"
echo INPUT FILES
echo "###########################"

TRAIN=sentiment_graphs/$DATASET/$SETUP/train.conllu
DEV=sentiment_graphs/$DATASET/$SETUP/dev.conllu
TEST=sentiment_graphs/$DATASET/$SETUP/test.conllu

echo train data: $TRAIN
echo dev data: $DEV
echo test data: $TEST
echo



# OUTPUT DIR
############################
echo "###########################"
echo OUTPUT DIR
echo "###########################"

DIR=experiments/$DATASET/$SETUP/$RUN
echo saving experiment to $DIR

pwd
rm -rf $DIR
mkdir $DIR

python3 ./src/main.py --config configs/sgraph.cfg --train $TRAIN --val $DEV --predict_file $TEST --dir $DIR --external $EXTERNAL --seed $SEED


# The models can be quite big and eat up a lot of space
# If you want to delete the models themselves, uncomment the next lines

# rm $DIR/best_model.save
# rm $DIR/last_epoch.save
