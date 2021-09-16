#!/bin/bash


###############################################################################
# COLLECT DATA
###############################################################################


# First download the darmstadt data from the following url (https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2448) and put the zip file in this directory.


###############################################################################
# PROCESS DATA
###############################################################################

# Process darmstadt data
unzip DarmstadtServiceReviewCorpus.zip
cd DarmstadtServiceReviewCorpus
unzip universities
grep -rl "&" universities/basedata | xargs sed -i '' -e 's/&/and/g'
cd ..
python3 process_darmstadt.py
rm -rf DarmstadtServiceReviewCorpus
