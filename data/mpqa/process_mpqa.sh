#!/bin/bash


###############################################################################
# COLLECT DATA
###############################################################################

# First download the mpqa 2.0 data from http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0 and change the following path to point to the tar file
mpqa_tar_file="./mpqa_2_0_database.tar.gz"
tar -xvf $mpqa_tar_file


###############################################################################
# PROCESS DATA
###############################################################################

# Process mpqa data
python3 process_mpqa.py
