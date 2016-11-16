# nchah/machine-learning-scripts/nltk

## Setup

This library is called the Natural Language Toolkit (NLTK) and imported as follows.

```
$ python
import nltk
```

NLTK comes with some additional datasets and models and can be downloaded as follows.

```
nltk.download()  # This brings up a separate GUI to manage the downloads
```

## Basics

### Loading Text

Text data can come in many forms.
Some common examples are plaintext files that can be edited with most basic text editors (.txt), text data encoded in JSON, XML, HTML format (.json, .xml, .html), or text in tabular data (.csv, .tsv).
Regardless of the source data type, the textual data needs to be loaded into the Python environment with the appropriate commands first.

For the most, in Python 2.x we can use the open(), or codecs.open() in the case of opening with a unicode encoding.

```
import codecs

with codecs.open('path/to/file', encoding='utf8') as f:
    text = f.read()  # or .readlines() etc...
```

### Tokenization

Once a corpus of texts has been loaded into Python, tokenization is often a next step.

```
from nltk import word_tokenize

text = ""

```


### Sentence Segmentation


### "Pre-processing" 

Or, what to do with stop words, punctuation, and inconsistent letter case.



## Applications

### Word Counts and Distributions












