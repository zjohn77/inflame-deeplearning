[![Build Status](https://travis-ci.org/zjohn77/corpus4classify.svg?branch=master)](https://travis-ci.org/zjohn77/corpus4classify)
[![License](https://img.shields.io/github/license/zjohn77/corpus4classify.svg)](https://github.com/zjohn77/corpus4classify/blob/master/LICENSE.md)
[![PyPI](https://img.shields.io/pypi/v/corpus4classify.svg)](https://pypi.org/project/corpus4classify/)

## 1. Features
* several corpora for NLP and specifically text classification research
* the original source data in multiple separate text files
* python scripts that input the source data, place it in data structures, and
expose an API for access to the data.

## 2. Install
```sh
pip install artifact
```

## 3. Usage
The two corpora currently available are the 20 Newsgroups corpus and the BBC News corpus. Fetching a corpus is accomplished using the **getdata** function, which can be imported like so:

```python
from artifact import getdata
```

Once getdata is in namespace, get the BBC News corpus via:
```python
data, target = getdata('bbcnews')
print(data[:5])
```

Likewise for the 20 Newsgroups corpus:
```python
data, target = getdata('newsgrp')
print(data[:5])
```

## 4. Contribute
Any contribution is welcome. To get started:
```sh
git clone https://github.com/zjohn77/corpus4classify.git
pip install -r requirements.txt
cd test && python -m unittest
```

## Sentiment Classifier Training App
### Install
Go inside the sentiment_imdb directory and then execute the following:

`python -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

### Run
Run the trainer script from its entry point as follows:

`python pytorch_train.py`

The above should write to the configured cloud bucket the sentiment classification of a sample review.