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

## Get the baseline models

In order to train the baseline models, we provide scripts which will download pretrained embeddings and train monolingual models:

```
bash ./get_baseline.sh
```

The models will be saved as pytorch models in /experiments under the name best_model.save. You can then use the inference.sh script to use the models to predict on unseen data:

```
bash ./inference.sh sentiment_graphs/multibooked_eu/head_final/dev.conllu experiments/multibooked_eu/head_final/ embeddings/32.zip
```

The predictions (dev.conllu.pred, dev.conllu.json) will be written to the same directory where the model is found. The json files contain the predictions converted to the appropriate submission format, while the conllu.pred files show the actual predictions.
