## Structured Sentiment

This Github repository hosts the data and baseline models for the SemEval-2022 shared task on structured sentiment.

## Problem description

The task is to extract all of the opinion tuples **O** = *O*<sub>i</sub>,...,*O*<sub>n</sub> in a text. Each opinion *O*<sub>i</sub> is a tuple *(h, t, e, p)*

where *h* is a **holder** who expresses a **polarity** *p* towards a **target** *t* through a **sentiment expression** *e*, implicitly defining the relationships between these elements.

The two examples below (first in English, then in Basque) show how we define *sentiment graphs*.

![multilingual example](./figures/multi_sent_graph.png)

You can then either treat this as a sequence-labelling task, or as a graph prediction task.

## Monolingual Track
The first track is

## Cross-lingual Track

## Data

We provide the data in json format.


## Task organizers

* Corresponding organizers
    * [Jeremy Barnes](jeremycb@ifi.uio.no)
    * [Andrey Kutuzov](andreku@ifi.uio.no)
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


