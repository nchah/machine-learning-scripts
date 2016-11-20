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

Sklearn is ideal for vectorizing texts, an important step in many machine learning applications.

```
from sklearn.feature_extraction.text import CountVectorizer

# Convert the corpus into a list where each item is a document of type:string
# then create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Create a corpus of type:list
corpus = [doc1, doc2]

X = vectorizer.fit_transform(corpus)
```

### TF-IDF

Term-frequency-inverse document frequency is another widely used statistic.

```
from sklearn.feature_extraction.text import TfidfVectorizer

doc1 = "Pack my box with five dozen liquor jugs. Jackdaws love my big sphinx of quartz. The five boxing wizards jump quickly."
doc2 = "The quick brown fox jumps over the lazy dog. How vexingly quick daft zebras jump. Bright vixens jump; dozy fowl quack."

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform([doc1, doc2])

# Get the terms and the respective scores. Access by indexing: terms[0], idf[0]
terms = vectorizer.get_feature_names()
# -> [u'big', u'box', u'boxing', u'bright', u'brown', u'daft', u'dog', u'dozen', u'dozy', u'five', u'fowl', u'fox', u'how', u'jackdaws', u'jugs', u'jump', u'jumps', u'lazy', u'liquor', u'love', u'my', u'of', u'over', u'pack', u'quack', u'quartz', u'quick', u'quickly', u'sphinx', u'the', u'vexingly', u'vixens', u'with', u'wizards', u'zebras']

idf = vectorizer.idf_
# -> array([ 1.40546511,  1.40546511,  1.40546511,  1.40546511,  1.40546511,
        1.40546511,  1.40546511,  1.40546511,  1.40546511,  1.40546511,
        1.40546511,  1.40546511,  1.40546511,  1.40546511,  1.40546511,
        1.        ,  1.40546511,  1.40546511,  1.40546511,  1.40546511,
        1.40546511,  1.40546511,  1.40546511,  1.40546511,  1.40546511,
        1.40546511,  1.40546511,  1.40546511,  1.40546511,  1.        ,
        1.40546511,  1.40546511,  1.40546511,  1.40546511,  1.40546511])
```

### Similarity Measures

#### Cosine Similarity

Using the already calculated TF-IDF matrix to calculate the cosine similarity.

```
(tfidf * tfidf.T).A
# -> array([[ 1.        ,  0.08607287],
            [ 0.08607287,  1.        ]])
```

#### Jaccard Coefficient

This doesn't make use of sklearn but is worth mentioning as a similarity measure.

```
import nltk
from __future__ import division  # if using Python 2.x

# Usual pre-processing and setup steps
doc1 = word_tokenize(doc1)
doc2 = word_tokenize(doc2)
punctuation = nltk.corpus.stopwords.words('english')
doc1 = [word.lower() for word in doc1 if word not in punctuation]
doc2 = [word.lower() for word in doc2 if word not in punctuation]


# Jaccard coefficient = intersection_count/union_count
intersection = [word for word in doc1 if word in doc2]
union = doc1 + doc2
jaccard = len(intersection)/len(union)
# -> 0.047619047619047616
```

## Machine Learning Steps

### Vectorizing

This was covered in the above sections. To repeat:
```
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

doc1 = "Pack my box with five dozen liquor jugs. Jackdaws love my big sphinx of quartz. The five boxing wizards jump quickly."
doc2 = "The quick brown fox jumps over the lazy dog. How vexingly quick daft zebras jump. Bright vixens jump; dozy fowl quack."

corpus = [doc1, doc2]

X = vectorizer.fit_transform(corpus)
```

### Creating Training & Test Samples


### Fitting and Evaluating Models


#### Cross-validation and Accuracy




## Further Machine Learning Models

### Naive Bayes Classifier


### K-Nearest Neighbor (K-NN Classifier)


### Support Vector Machines


### Clustering





