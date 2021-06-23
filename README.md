## Structured Sentiment

This Github repository hosts the data and baseline models for the SemEval-2022 shared task on structured sentiment.

## Problem description

The task is to extract all of the opinion tuples **O** = *O*<sub>i</sub>,...,*O*<sub>n</sub> in a text. Each opinion *O*<sub>i</sub> is a tuple *(h, t, e, p)*

where *h* is a **holder** who expresses a **polarity** *p* towards a **target** *t* through a **sentiment expression** *e*, implicitly defining the relationships between these elements.

The two examples below (first in English, then in Basque) show how we define *sentiment graphs*.

![multilingual example](./figures/multi_sent_graph.png)

You can then either treat this as a sequence-labelling task, or as a graph prediction task.

## Track 1: Monolingual structured sentiment
The first track is

## Track 2: Cross-lingual structured sentiment
This track will explore how well


## Data and formatting

We provide the data in json format.

Each sentence has a dictionary with the following keys and values:

* 'sent_id': unique sentence identifiers

* 'text': raw text

* 'opinions': list of all opinions (dictionaries) in the sentence

Additionally, each opinion in a sentence is a dictionary with the following keys and values:

* 'Source': a list of text and character offsets for the opinion holder

* 'Target': a list of text and character offsets for the opinion target

* 'Polar_expression': a list of text and character offsets for the opinion expression

* 'Polarity': sentiment label ('Negative', 'Positive')

* 'Intensity': sentiment intensity ('Standard', 'Strong', 'Slight')


```
{
    "sent_id": "../opener/en/kaf/hotel/english00164_c6d60bf75b0de8d72b7e1c575e04e314-6",

    "text": "Even though the price is decent for Paris , I would not recommend this hotel .",

    "opinions": [
                 {
                    "Source": [["I"], ["44:45"]],
                    "Target": [["this hotel"], ["66:76"]],
                    "Polar_expression": [["would not recommend"], ["46:65"]],
                    "Polarity": "Negative",
                    "Intensity": "Standard"
                  },
                 {
                    "Source": [[], []],
                    "Target": [["the price"], ["12:21"]],
                    "Polar_expression": [["decent"], ["25:31"]],
                    "Polarity": "Positive",
                    "Intensity": "Standard"}
                ]
}
```

You can import the data by using the json library in python:

```
>>> import json
>>> with open("data/norec/train.json") as infile:
            norec_train = json.load(infile)
```


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


