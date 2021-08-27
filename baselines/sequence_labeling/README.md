## Pipeline sequence labeling + relation classification
Here we provide starter code for a model that first learns to extract the sub-elements (holders, targets, expressions) using sequence labelers, and then tries to classify whether or not they have a relationship.


## Requirements

1. python3
2. pytorch
3. torchtext
4. sklearn
5. gensim



## Get the baselines

1. Download embeddings if they are not already located in the graph_parser/embedding directory (see first lines of get_baseline.sh in graph_parser baseline) and set the EMBEDDINGDIR variable in get_baselines.sh

2. Run the following script to train baseline models on all datasets
```
bash ./get_baselines
```

3. Perform inference on any dataset, the resulting json file will be stored as "saved_models/relation_prediction/DATASET/prediction.json"
```
python3 inference.py -data opener_en -file dev.json
```
