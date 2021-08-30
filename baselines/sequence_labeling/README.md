## Pipeline sequence labeling + relation classification
Here we provide starter code for a model that first learns to extract the sub-elements (holders, targets, expressions) using sequence labelers, and then tries to classify whether or not they have a relationship.

Specifically, we first train three separate BiLSTM models to extract holders, targets, and expressions, respectively (extraction_module.py). We then train a relation prediction model (relation_prediction_model.py), which uses a BiLSTM + max pooling to create contextualized representations of 1) the full text, 2) the first element (either a holder or target) and 3) the sentiment expression.

These three representations are then concatenated and passed to a linear layer followed by a sigmoid function. The training consists of predicting whether two elements have a relationship or not, converting the problem in binary classification.

During inference (inference.py), we first predict all sub-elements and then decide if they have a relationship (prediction > 0.5). Finally, the predictions are converted to the json format used in the shared task.


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
