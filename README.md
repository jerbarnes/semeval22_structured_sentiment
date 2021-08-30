## SemEval-2022 Shared Task 10: Structured Sentiment Analysis

This Github repository hosts the data and baseline models for the [SemEval-2022 shared task 10](https://competitions.codalab.org/competitions/33556) on structured sentiment. In this repository you will find the datasets, baselines, and other useful information on the shared task.

**Table of contents:**

1. [Problem description](#problem-description)
2. [Subtasks](#subtasks)
   1. [Monolingual](#monolingual)
      1. [Data](#data)
   2. [Cross-lingual](#cross-lingual)
3. [Evaluation](#evaluation)
4. [Data format](#data-format)
5. [Resources](#resources)
6. [Submission via Codalab](#submission-via-codalab)
7. [Baselines](#baselines)
8. [Important dates](#important-dates)
9. [Frequently Asked Questions](#frequently-asked-questions)
10. [Task organizers](#task-organizers)

## Problem description

The task is to predict all structured sentiment graphs in a text (see the examples below). We can formalize this as finding all the opinion tuples *O* = *O*<sub>i</sub>,...,*O*<sub>n</sub> in a text. Each opinion *O*<sub>i</sub> is a tuple *(h, t, e, p)*

where *h* is a **holder** who expresses a **polarity** *p* towards a **target** *t* through a **sentiment expression** *e*, implicitly defining the relationships between the elements of a sentiment graph.

The two examples below (first in English, then in Basque) give a visual representation of these *sentiment graphs*.

![multilingual example](./figures/multi_sent_graph.png)

Participants can then either approach this as a sequence-labelling task, or as a graph prediction task.

## Subtasks
### Monolingual
This track assumes that you train and test on the same language. Participants will need to submit results for seven datasets in five languages.

 The datasets can be found in the [data](./data) directory.

#### Data

| Dataset | Language | # sents | # holders | # targets | # expr. |
| --------| -------- | ------- | --------- | --------- | ------- |
| [NoReC_fine](https://aclanthology.org/2020.lrec-1.618/) | Norwegian | 11437 | 1128|8923 |11115 |
| [MultiBooked_eu](https://aclanthology.org/L18-1104/) | Basque |1521 |296 |1775 |2328 |
| [MultiBooked_ca](https://aclanthology.org/L18-1104/) | Catalan |1678 |235 |2336 |2756 |
| [OpeNER_es](http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/4891) | Spanish |2057 |255 |3980 |4388 |
| [OpeNER_en](http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/4891) | English |2494 |413 |3850 |4150 |
| [MPQA](http://mpqa.cs.pitt.edu/) | English | | | | |
| [Darmstadt_unis](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2448) | English | 2803 | 86 | 1119 | 1119 |

### Cross-lingual
This track will explore how well models can generalize across languages. The test data will be the `MultiBooked Datasets (Catalan and Basque)` and the `OpeNER Spanish` dataset. For training, you can use any of the other datasets, as well as any other resource that does not come directly from the test datasets.

## Evaluation

The two subtasks will be evaluated separately. In both tasks, the evaluation will be based on [Sentiment Graph F<sub>1</sub>](https://arxiv.org/abs/2105.14504).


This metric defines true positive as an *exact match* at
graph-level, *weighting the overlap* in predicted and
gold spans for each element, averaged across all
three spans.

For *precision* we weight the number
of correctly predicted tokens divided by the total
number of predicted tokens (for *recall*, we divide
instead by the number of gold tokens), allowing
for empty holders and targets which exist in the gold standard.


 The leaderboard for each dataset, as well as the average of all 7. The winning submission will be the one that has the highest average Sentiment Graph F<sub>1</sub>.


## Data format

We provide the data in json lines format.

Each line is an annotated sentence, represented as a dictionary with the following keys and values:

* `'sent_id'`: unique sentence identifiers

* `'text'`: raw text version of the previously tokenized sentence

* `opinions'`: list of all opinions (dictionaries) in the sentence

Additionally, each opinion in a sentence is a dictionary with the following keys and values:

* `'Source'`: a list of text and character offsets for the opinion holder

* `'Target'`: a list of text and character offsets for the opinion target

* `'Polar_expression'`: a list of text and character offsets for the opinion expression

* `'Polarity'`: sentiment label ('negative', 'positive', 'neutral')

* `'Intensity'`: sentiment intensity ('average', 'strong', 'weak')


```
{
    "sent_id": "../opener/en/kaf/hotel/english00164_c6d60bf75b0de8d72b7e1c575e04e314-6",

    "text": "Even though the price is decent for Paris , I would not recommend this hotel .",

    "opinions": [
                 {
                    "Source": [["I"], ["44:45"]],
                    "Target": [["this hotel"], ["66:76"]],
                    "Polar_expression": [["would not recommend"], ["46:65"]],
                    "Polarity": "negative",
                    "Intensity": "average"
                  },
                 {
                    "Source": [[], []],
                    "Target": [["the price"], ["12:21"]],
                    "Polar_expression": [["decent"], ["25:31"]],
                    "Polarity": "positive",
                    "Intensity": "average"}
                ]
}
```

You can import the data by using the json library in python:

```
>>> import json
>>> with open("data/norec/train.json") as infile:
            norec_train = json.load(infile)
```

## Resources:
The task organizers provide training data, but participants are free to use other resources (word embeddings, pretrained models, sentiment lexicons, translation lexicons, translation datasets, etc). We do ask that participants document and cite their resources well.

We also provide some links to what we believe could be helpful resources:

1. [pretrained word embeddings](http://vectors.nlpl.eu/repository/)
2. [pretrained language models](https://huggingface.co/models)
3. [translation data](https://opus.nlpl.eu/)
4. [sentiment resources](https://github.com/jerbarnes/sentiment_resources)
5. [syntactic parsers](https://stanfordnlp.github.io/stanza/)


## Submission via Codalab
Submissions will be handled through our [codalab competition website](https://competitions.codalab.org/competitions/33556).

## Baselines

The task organizers provide two baselines: one that takes a sequence-labelling approach and a second that converts the problem to a dependency graph parsing task. You can find both of them in [baselines](./baselines).

## Important dates

   - Training data ready: September 3, 2021
   - Evaluation data ready: December 3, 2021
   - Evaluation start: January 10, 2022
   - Evaluation end: by January 31, 2022
   - Paper submissions due: roughly February 23, 2022
   - Notification to authors: March 31, 2022


## Frequently asked questions

Q: How do I participate?

A: Sign up at our [codalab website](https://competitions.codalab.org/competitions/33556), download the [data](./data), train the [baselines](./baselines) and submit the results to the codalab website.


## Task organizers

* Corresponding organizers
    * [Jeremy Barnes](https://jerbarnes.github.io/): contact for info on task, participation, etc. (<jeremycb@ifi.uio.no>)
    * [Andrey Kutuzov](https://www.mn.uio.no/ifi/english/people/aca/andreku/index.html): <andreku@ifi.uio.no>
* Organizers
    * Jan Buchman
    * Laura Ana Maria Oberländer
    * Enrica Troiano
    * Rodrigo Agerri
    * Lilja Øvrelid
    * Erik Velldal
    * Stephan Oepen



