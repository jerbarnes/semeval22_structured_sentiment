#!/bin/bash
if [ ! -f metrics.py ]; then
    curl https://raw.githubusercontent.com/sarnthil/emotion-stimulus-detection/main/scripts/eval/metrics.py > metrics.py
fi
if [ ! -f count_errors.py ]; then
    curl https://raw.githubusercontent.com/sarnthil/emotion-stimulus-detection/main/scripts/eval/count_errors.py > count_errors.py
fi
exit

# location of the predictions.json file
PREDFILE=$1

# domain analysis
python3 domain_analysis.py ../data/norec/test_labeled.json $PREDFILE metadata.json

# negation analysis
python3 neg_scope_analysis.py ../data/norec/test_labeled.json $PREDFILE negation_test.json

# overlap analysis
python3 overlap_analysis.py ../data/norec/test_labeled.json $PREDFILE
