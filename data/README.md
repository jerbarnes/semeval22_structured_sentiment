## Requirements

1. lxml==4.3.2
2. tqdm=4.56.0
3. stanza==1.1.1

## Step 1:

Go to the [MPQA 2.0](http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/) website, agree to the license and download the corpus. Put the zipped archive in /mpqa. Finally, run the extraction script.

```
bash process_mpqa.sh
```


Go to the [Darmstadt Service Review Corpus](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2448) website, agree to the license and download the corpus. Put the zipped archive in /darmstadt_unis and finally, run the extraction script.

```
bash process_darmstadt.sh
```



## Subtask 1: Monolingual structured sentiment
This track assumes that we train and test on the same languages. For this we will use the following datasets:

1. norec (Norwegian professional reviews in multiple domains)
2. multibooked_ca (Catalan hotel reviews)
3. multibooked_eu (Basque hotel reviews)
4. opener_en (English hotel reviews)
5. opener_es (Spanish hotel reviews)
6. darmstadt_unis (English online university reviews)
7. MPQA

## Subtask 2: Cross-lingual structured sentiment
This track will instead train only on a high-resource language (English) and test on several languages.

For training, you can use any of the other datasets, as well as any other resource that does not contain sentiment annotations in the target language.

Test:
1. opener_es
2. multibooked_ca
3. multibooked_eu

That means that the cross-lingual models should be able to adapt quickly to new languages.


## Data and formatting
We provide the data in json lines format.

Each line is an annotated sentence, represented as a dictionary with the following keys and values:

* 'sent_id': unique sentence identifiers

* 'text': raw text

* 'opinions': list of all opinions (dictionaries) in the sentence

Additionally, each opinion in a sentence is a dictionary with the following keys and values:

* 'Source': a list of text and character offsets for the opinion holder

* 'Target': a list of text and character offsets for the opinion target

* 'Polar_expression': a list of text and character offsets for the opinion expression

* 'Polarity': sentiment label ('negative', 'positive', 'neutral')

* 'Intensity': sentiment intensity ('average', 'strong', 'weak')


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


