## Structured Sentiment as Dependency Graph Parsing baseline

This subrepository contains the code and datasets described in following paper: [Structured Sentiment Analysis as Dependency Graph Parsing]().

Rather than treating the sentiment graph prediction task as sequence-labeling, we can treat it as a bilexical dependency graph prediction task, although some decisions must be made. We provide two versions (a) *head-first* and (b) *head-final*, shown below:

![bilexical](./figures/bilexical.png)


## Requirements

1. python3
2. pytorch
3. matplotlib
4. sklearn
5. gensim
6. numpy
7. h5py
8. transformers
9. tqdm


## Experimental results

To reproduce the results, first you will need to download the word vectors used:

```
mkdir vectors
cd vectors
wget http://vectors.nlpl.eu/repository/20/58.zip
wget http://vectors.nlpl.eu/repository/20/32.zip
wget http://vectors.nlpl.eu/repository/20/34.zip
wget http://vectors.nlpl.eu/repository/20/18.zip
cd ..
```

You will similarly need to extract mBERT token representations for all datasets.
```
./do_bert.sh
```

Finally, you can run the SLURM scripts to reproduce the experimental results.

```
./scripts/run_SLURM_all_BERT.sh
./scripts/run_SLURM_no_BERT.sh
```

