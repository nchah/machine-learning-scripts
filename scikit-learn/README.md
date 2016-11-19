# nchah/machine-learning-scripts/scikit-learn

## Setup

Import the Scikit-learn library as follows.

```
# python
import sklearn
```

## Basics

This section assumes that the text has already been loaded into the Python environment.

## Applications

### Vectorization

Sklearn is ideal for vectorizing texts.

```
from sklearn.feature_extraction.text import CountVectorizer

# Convert the corpus into a list where each item is a document of type:string
# then create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Create a corpus of type:list
corpus = [doc1, doc2]

X = vectorizer.fit_transform(corpus)

```







