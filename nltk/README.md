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


### "Pre-processing" 

Or, what to do with stop words, punctuation, and inconsistent letter case.


### Lower Case

Use Python's built-in function to turn text into an all lower-case version if this is needed for further analysis.

```
sentence = "The quick brown fox jumps over the lazy dog."
sentence_tokens = word_tokenize(sentence)

sentence_tokens = [word.lower() for word in sentence_tokens]
print(sentence_tokens)
# -> ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
```

Converting all text into lower case may be a necessary step when checking for matches against a lexicon, finding substrings where the text can have inconsistent spelling, and other use cases.

### Stop Words

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
# Add
stopwords += more_stopwords

# Remove
stopwords.remove('etc...')
```



## Applications

### Word Counts and Distributions












