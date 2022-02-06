#!/bin/bash

# location of the predictions.json file
PREDFILE=$1

# domain analysis
python3 domain_analysis.py ../data/norec/test_labeled.json $PREDFILE metadata.json

# negation analysis
python3 neg_scope_analysis.py ../data/norec/test_labeled.json $PREDFILE negation_test.json
