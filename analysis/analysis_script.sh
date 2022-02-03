#!/bin/bash

# domain analysis
python3 domain_analysis.py ../data/norec/test_labeled.json predictions.json metadata.json

# negation analysis
python3 neg_scope_analysis.py ../data/norec/test_labeled.json ../data/norec/test.json negation_test.json 
