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

## Supervised Machine Learning 

### Common Steps

#### Vectorizing

This was covered in the above sections. To repeat the core steps:
```
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(corpus)
```

#### Creating Training & Test Samples

The data needs to be divided into different sets for training the machine learning model, and then testing the model's accuracy.
```
# Assume the following variables where each item's index match across variable:
# text = a list of text values
# labels = a list of labels corresponding to each item in text

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

vectorizer = CountVectorizer(stop_words='english')  # Option for stopwords
x = vectorizer.fit_transform(text)

# Create the training, test samples with test_size 30%
# Remember "x" is the vectorized "text" variable
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.3)

# Check the sample sizes to verify
print len(y_train), len(y_test)
```

#### Fitting and Evaluating Models

Logistic Regression and Naive Bayes models will be shown here.
```
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

# Evaluating the model that has now been fit
from sklearn import metrics

y_hat = lr.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_hat)
accuracy = metrics.accuracy_score(y_test, y_hat)

print confusion_matrix
print accuracy
```

This is the Multinomial Naive Bayes model. Same process can be done for the Bernoulii Naive Bayes classifier.
```
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

nb = MultinomialNB()
nb.fit(x_train, y_train)

y_hat = nb.predict(x_test)
confusion_matrix = metrics.confusion_matrix(y_test, y_hat)
accuracy = metrics.accuracy_score(y_test, y_hat)

print confusion_matrix
print accuracy
```

With the Bernoulli Naive Bayes, the threshold for how many times a term occurs in a document can be set, as well as a minimum document frequency, and an upper limit on number of features.
```
vectorizer = CountVectorizer(stop_words='english', min_df=10, max_features=1000)
x = vectorizer.fit_transform(text)
x_train, x_test, y_train, y_test = train_test_split(x, labels, random_state=0)

nb = BernoulliNB(binarize=2)  # A term must occur at least 2 times in a document
nb.fit(x_train, y_train)

y_hat = nb.predict(x_test)

print metrics.accuracy_score(y_test, y_hat)
```

In addition to the accuracy measure, it's good to check the most significant features according to the model.
```
# Get how the feature classes are ordered in the model, if not known
print nb.classes_

prob_class1 = nb.feature_log_prob_[0,:].tolist()
prob_class2 = nb.feature_log_prob_[1,:].tolist()
# etc...

terms = vectorizer.get_feature_names()

import math
from __future__ import division

# Further complex functions from class notes, TODO: re-write
probs = [(terms[i], math.exp(prob_class1[i]), math.exp(prob_class2[i]),
          math.exp(prob_class1[i])/math.exp(prob_class2[i]),
          math.exp(prob_class2[i])/math.exp(prob_class1[i]))
          for i,_ in enumerate(prob_class1)]

most_important_features_1 = sorted(probs, key=lambda tup: tup[3], reverse=True)[0:10]
# etc...

for t,_,_,fratio,_ in most_important_features_1:
    print '%-20s%-0.3f' %(t, fratio)
```

#### Cross-validation and Accuracy

This section covers the K-folds and Stratified Shuffle Split cross-validation measures.

Accuracy and F1 scores can also be calculated.

```
# Usual boilerplate for setting up the model
vectorizer = CountVectorizer(stop_words='english', min_df=10, max_features=1000)
x = vectorizer.fit_transform(text)
nb = MultinomialNB()
```

K-folds method:
```
from sklearn.cross_validation import KFold,StratifiedShuffleSplit
import numpy as np

k10 = KFold(x.shape[0], 10, random_state=0)
# convert classes into an array of integers.
z, y = np.unique(y, return_inverse=True)

acc = []
f1 = []
for k, (train, test) in enumerate(k10):
    Mnb.fit(x[train], y[train])
    yhat = Mnb.predict(x[test])
    t_acc = metrics.accuracy_score(y[test],yhat)
    t_f1 = metrics.f1_score(y[test],yhat)
    acc.append(t_acc)
    f1.append(t_f1)

print "The percent correctly predicted is %0.2f%%" % (float(np.mean(acc))*100)
print "The F1 score is %0.3f." % float(np.mean(f1))
```

Stratified Shuffle Split:
```
from sklearn.cross_validation import StratifiedShuffleSplit

ssp = StratifiedShuffleSplit(y, 10, test_size=0.3, random_state=0)

acc = []
f1 = []
for train_index, test_index in ssp:
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Mnb.fit(x_train,y_train)
    yhat = Mnb.predict(x_test)
    t_acc = metrics.accuracy_score(y_test,yhat)
    t_f1 = metrics.f1_score(y_test,yhat)
    acc.append(t_acc)
    f1.append(t_f1)  

print "The percent correctly predicted is %0.2f%%" % (float(np.mean(acc))*100)
print "The F1 score is %0.3f." % float(np.mean(f1))
```

## Further Machine Learning Models

### Naive Bayes Classifier

Naives Bayes models (Multinomial, Bernoulli) were covered earlier.

### K-Nearest Neighbor (K-NN Classifier)


### Support Vector Machines



## Unsupervised Machine Learning

### Clustering

#### Flat Clustering

Some steps are similar to the supervised machine learning methods.

```
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', max_features=200)
x = tfidf.fit_transform(corpus)
```

```
from sklearn.cluster import KMeans, MiniBatchMeans

km = KMeans(n_clusters=3)
km.fit(x)

# Getting the predicted clusters
km.predict(x)
# OR
km.labels_  # with the _ underscore

# The predicted data can then be inserted into a pandas DataFrame for ex:
import pandas as pd

newdata = pd.DataFrame({'classes' : km.predict(x),'text' : corpus})
newdata.head(10)
```

Using silhouette scores to determine the quality of the resulting clusters.
```
from sklearn.metrics import silhouette_score

silhouette_score(x, km.labels_, metric='euclidean')

# Looping over various K clusters to see the best K
for K in [2,3,4,5,6,7]:
    km = KMeans(n_clusters=K)
    km.fit(x)
    print silhouette_score(x, km.labels_,metric='euclidean')
```

A simple loop to get the words in each cluster:
```
centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names()
for i in range(3):
    print "Cluster %d:" % (i+1) 
    for ind in centroids[i, :10]:
        print ' %s' % terms[ind] 
```

#### Hierarchical Clustering





### Latent Dirichlet Allocation (LDA)
















