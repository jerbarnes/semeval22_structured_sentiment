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
    EXTERNAL=embeddings/58.zip
elif [ $DATASET == multibooked_eu ]; then
    EXTERNAL=embeddings/32.zip
elif [ $DATASET == multibooked_ca ]; then
    EXTERNAL=embeddings/34.zip
elif [ $DATASET == mpqa ]; then
    EXTERNAL=embeddings/18.zip
elif [ $DATASET == darmstadt_unis ]; then
    EXTERNAL=embeddings/18.zip
elif [ $DATASET == opener_en ]; then
    EXTERNAL=embeddings/18.zip
elif [ $DATASET == opener_es ]; then
    EXTERNAL=embeddings/68.zip
else
    echo "NO EMBEDDINGS SUPPLIED FOR THIS DATASET"
    echo "EXITING TRAINING PROCEDURE"
    exit
fi

echo using external embeddings: $EXTERNAL
echo

# INPUT FILES
############################
echo "###########################"
echo INPUT FILES
echo "###########################"

TRAIN=sentiment_graphs/$DATASET/$SETUP/train.conllu

# During the development phase, you will not have access to gold development labels, but in the evaluation phase you can uncomment the line below and make the appropriate changes to use the gold development labels
# DEV=sentiment_graphs/$DATASET/$SETUP/dev.conllu

echo train data: $TRAIN
# echo dev data: $DEV
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


python3 ./src/main.py --config configs/sgraph.cfg --train $TRAIN --enable_train_eval true --disable_val_eval true --dir $DIR --external $EXTERNAL --seed $SEED

# When you receive the gold development labels, you can use them in training
# python3 ./src/main.py --config configs/sgraph.cfg --train $TRAIN --val $DEV --predict_file $DEV --dir $DIR --external $EXTERNAL --seed $SEED


# The models can be quite big and eat up a lot of space
# If you want to delete the models themselves, uncomment the next lines

# rm $DIR/best_model.save
# rm $DIR/last_epoch.save
