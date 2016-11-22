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

sentence = "The quick brown fox jumps over the lazy dog."
sentence_tokens = word_tokenize(sentence)

print(sentence_tokens)
# -> ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'] 

```
This turns the text (type: string) into a list where every element is what the tokenizer detects to be a standalone word.


### Sentence Segmentation

Similar to the tokenization step, sentence segmentation is applied to a body of text to break it up into a list of sentences.

```
from nltk import sent_tokenize

text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
text_sentences = sent_tokenize(text)

print(text_sentences)
# -> ['Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.', 'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.', 'Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.', 'Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'] 
```

Once a corpus of text is divided into individual sentences, you can apply word tokenization to obtain each individual element.
Since this is Python, use list comprehensions for more idiomatic programming.
This creates a list of lists.

```
sentence_tokens = [word_tokenize(sent) for sent in sent_tokenize(text)] 

print(sentence_tokens)
# -> [['Lorem', 'ipsum', 'dolor', 'sit', 'amet', ',', 'consectetur', 'adipiscing', 'elit', ',', 'sed', 'do', 'eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore', 'magna', 'aliqua', '.'], ['Ut', 'enim', 'ad', 'minim', 'veniam', ',', 'quis', 'nostrud', 'exercitation', 'ullamco', 'laboris', 'nisi', 'ut', 'aliquip', 'ex', 'ea', 'commodo', 'consequat', '.'], ['Duis', 'aute', 'irure', 'dolor', 'in', 'reprehenderit', 'in', 'voluptate', 'velit', 'esse', 'cillum', 'dolore', 'eu', 'fugiat', 'nulla', 'pariatur', '.'], ['Excepteur', 'sint', 'occaecat', 'cupidatat', 'non', 'proident', ',', 'sunt', 'in', 'culpa', 'qui', 'officia', 'deserunt', 'mollit', 'anim', 'id', 'est', 'laborum', '.']]
```

There is no requirement to use the list data structure, The same word_tokenize() and sent_tokenize() functions can be applied to the appropriate text data.


### Pre-processing Stages

Or, what to do with stop words, punctuation, and inconsistent letter case.


#### Lower Case

Use Python's built-in function to turn text into an all lower-case version if this is needed for further analysis.

```
sentence = "The quick brown fox jumps over the lazy dog."
sentence_tokens = word_tokenize(sentence)

sentence_tokens = [word.lower() for word in sentence_tokens]
print(sentence_tokens)
# -> ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
```

Converting all text into lower case may be a necessary step when checking for matches against a lexicon, finding substrings where the text can have inconsistent spelling, and other use cases.

#### Stop Words

How to handle stopwords is another integral step in Natural Language Processing.
NLTK comes with its own stopwords list and can be accessed as follows.

```
from nltk.corpus import stopwords

# Specify the language
stopwords = stopwords.words('english')

print(stopwords)
# -> [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn', u'weren', u'won', u'wouldn']
```

The specific words that make up a stopwords list may vary for different methodologies, research studies, and use cases.

You could also use a custom stopwords list or add further items to the existing Python list.

```
more_stopwords = ['this', 'that', 'stopword', 'etc...']
# Add another list
stopwords += more_stopwords

# Append
stopwords.append('another word')

# Remove
stopwords.remove('etc...')
```

#### Punctuation

Punctuation can be handled in many ways as well. One method is to use the built-in String module.

```
import string
punctuation = set(string.punctuation)  # OR list(string.punctuation)

print(punctuation)
# -> set(['!', '#', '"', '%', '$', "'", '&', ')', '(', '+', '*', '-', ',', '/', '.', ';', ':', '=', '<', '?', '>', '@', '[', ']', '\\', '_', '^', '`', '{', '}', '|', '~'])
```

## Applications

### Type-Token Ratio (TTR)

You can get the Types in a text by applying the set() function on a text.
The Types are the set of *different* words in a text.

Higher values in the TTR usually indicate greater word variation (lexical complexity), and as a ratio the range of values range from [0, 1].

```
sentence = 'hello hello world world'
sentence = word_tokenize(sentence)
types = set(sentence)

print types
# -> set(['world', 'hello'])
```

Computing the Type-Token Ratio then can be done as follows with some simple arithmetic implemented in Python.
The proper division function must be used depending on the Python version.

```
text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'
text = word_tokenize(text)
types = set(text)

from __future__ import division  # if using Python 2.x
text_TTR = len(types)/len(text)

print(text_TTR)
# -> 0.8571428571428571
```

Variations on the TTR may use lemmas or word families to be more precise in how the Types are calculated.
The TTR calculation could also be done for every chunk of X words in a corpus and averaged over those values.

### Frequency Distributions

Applying a Frequency Distribution on a text will indicate the most frequent words along with their frequencies.

```
import nltk

text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'
text = word_tokenize(text)
text = [word for word in text if word not in punctuation]

text_FD = nltk.FreqDist(text)

# Get the Top 10 or get all if no param set
text_FD.most_common(10)
# -> [('in', 3), ('dolore', 2), ('ut', 2), ('dolor', 2), ('ad', 1), ('irure', 1), ('ea', 1), ('officia', 1), ('sunt', 1), ('elit', 1)]

# Or get the frequency of a word in the list of tuples
text_FD['Lorem']
# -> 1
```

There's a further step to use the matplotlib libraries to create a visualization if the library is not responsive.
```
# If running on OS X and the command below doesn't bring up a GUI
import matplotlib
matplotlib.use('TkAgg')
# Or follow the instructions in http://stackoverflow.com/questions/29433824/unable-to-import-matplotlib-pyplot-as-plt-in-virtualenv

text_FD.plot(20, cumulative=False)
```

### N-grams and Collocations

Collocations of 2 words by default can be accessed with a simple method under the NLTK Text class.

```
from nltk import word_tokenize
import string
punctuation = set(string.punctuation)

text = "The quick brown fox jumps over the lazy dog. The brown fox jumps over the dog. The quick brown fox jumps over the lazy fox. The quick brown fox jumps over the quick dog."
text = word_tokenize(text)

text = nltk.Text(text)
print text.collocations()
# -> brown fox; fox jumps; quick brown
```

Or by using the nrgams() method.
```
import nltk

ngrams = nltk.ngrams(text, 4)  # where # is #-gram

for ng in ngrams:
    print ng
# -> ('The', 'quick', 'brown', 'fox')
('quick', 'brown', 'fox', 'jumps')
('brown', 'fox', 'jumps', 'over')
('fox', 'jumps', 'over', 'the')
...
```

For deeper applications using collocations, there are additional functions.

This is another approach for bigrams that also includes scores for each ngram according to a measure.
```
from nltk.collocations import *

# For obtaining stats on bigrams
bigram_stats = nltk.collocations.BigramAssocMeasures()

# Initializing a bigram finder
bigram_finder = BigramCollocationFinder.from_words(text)

bigram_finder.nbest(bigram_stats.chi_sq, 5)
# -> [('jumps', 'over'), ('over', 'the'), ('brown', 'fox'), ('fox', 'jumps'), ('dog', '.')]

# Getting the score for a specific bigram
bigram_finder.score_ngram(bigram_stats.chi_sq, 'jumps', 'over')
# -> 38.0
```

A similar approach for trigrams.

```
# For obtaining stats on trigrams
trigram_stats = nltk.collocations.TrigramAssocMeasures()

# Initializing a trigram finder
trigram_finder = TrigramCollocationFinder.from_words(text, window_size=3)

# Get trigrams, top 5 ranked according to different rankings
# Chi-square, likelihood ratio, PMI score, etc...
trigram_finder.nbest(trigram_stats.chi_sq, 5)
# -> [('jumps', 'over', 'the'), ('brown', 'fox', 'jumps'), ('fox', 'jumps', 'over'), ('The', 'quick', 'brown'), ('over', 'the', 'lazy')]

trigram_finder.nbest(trigram_stats.likelihood_ratio, 5)
trigram_finder.nbest(trigram_stats.pmi, 5)

# Applying a filter to get collocations occurring at least X times
trigram_finder.apply_freq_filter(5)
```


### Part-of-speech (POS) Tagging

The NLTK POS tagger first needs to be downloaded (maxent_treebank_pos_tagger).

```
from nltk.tag import pos_tag

sentence = "The quick brown fox jumps over the lazy dog."
sentence = word_tokenize(sentence)

pos = nltk.pos_tag(sentence)
print pos
# -> [('The', 'DT'), ('quick', 'NN'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'NNS'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'NN'), ('dog', 'NN'), ('.', '.')]
```

The output is a list of tuples, where each tuple is the word and the POS.
Printing this out in a new string format using this data structure.
```
print " ".join([word + '_(' + tag + ')' for word, tag in pos])
# -> 'The_(DT) quick_(NN) brown_(NN) fox_(NN) jumps_(NNS) over_(IN) the_(DT) lazy_(NN) dog_(NN) ._(.)'
```


### Stemming and Lemmas

NLTK also comes with a number of different stemmers.
Each one has its own distinct behavior and can produce different stems.

Porter Stemmer:
```
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

text = "The five boxing wizards jump quickly."
text = word_tokenize(text)

stems = [stemmer.stem(word) for word in text]
print stems
# -> [u'The', u'five', u'box', u'wizard', u'jump', u'quickli', u'.']
```

Snowball Stemmer:
```
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')  # can add ignore_stopwords=True

stems = [stemmer.stem(word) for word in text]
print stems
# -> [u'the', u'five', u'box', u'wizard', u'jump', u'quick', '.']
```

Lancaster Stemmer:
```
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

stems = [stemmer.stem(word) for word in text]
print stems
# -> ['the', 'fiv', 'box', 'wizard', 'jump', 'quick', '.']
```

Using a lemmatizer instead of a stemmer is another approach.

```
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

text = "The five boxing wizards jump quickly."
text = word_tokenize(text)
pos_text = nltk.pos_tag([word.lower() for word in text])

lemmas = [lemmatizer.lemmatize(word) for word, tag in pos_text]
print lemmas
# -> ['the', 'five', 'boxing', u'wizard', 'jump', 'quickly', '.']
```


### Named Entity Recognition (NER)

NLTK implements named entity recognition as follows.
```
sentence = 'Canada and the U.S.A. are in North America.'
sentence = word_tokenize(sentence)
sentence = pos_tag(sentence)

ner = nltk.ne_chunk(sentence, binary=True)
print ner
# -> 
(S
  (NE Canada/NNP)
  and/CC
  the/DT
  U.S.A./NNP
  are/VBP
  in/IN
  (NE North/NNP America/NNP)
  ./.)
```

Applying the technique to a larger corpus.
```
text = 'word word word Canada. word United States. word word North America.'
text = sent_tokenize(text)
text = [word_tokenize(sentence) for sentence in text]
text = [pos_tag(sentence) for sentence in text]

chunked_nes = nltk.ne_chunk_sents(text, binary=True)

entities = []
for tree in chunked_nes:
    for branch in tree:
        if hasattr(branch, 'label') and branch.label:
            if branch.label() == 'NE':
                entities.append(' '.join([c[0] for c in branch]))
print entities
# -> ['United States', 'North America']
```

### Lexicons

In a basic implementation, lexicons can be a list of words that fall under a specific category.
For example, a 'positive' words lexicon would be a list of words that indicate a 'positive' sentiment.
The lexicon can then be used to find the presence of 'positive' words in a corpus.












