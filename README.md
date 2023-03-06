## Table of Contents
1. [Basic Idea and Key Benefits](#1-basic-idea-and-key-benefits)
2. [Install](#2-install)
3. [Quick Start](#3-quick-start)
4. [Contribute](#4-contribute)

## 1. Basic Idea and Key Benefits:
To build world-class deep learning models, it is necessary to do two things:
* train on state-of-the-art GPUs,
* experiment--on an industrial scale--with different parameter configurations.

The bottom line is that training a model one or twice on a basic, homegrown jupyter notebook isn't going to cut it when it comes to building a really good AI model--at least not ChatGPT good. That is exactly the raison d'être of this project (inflame-deeplearning). The goal is to be able to iterate lots and lots of times, throwing fresh data, parameters, ideas, and insights at the problem--until model accuracy is really good.

The inflame-deeplearning project helps solve this problem by architecting the training code as an app. This app has essentially 3 modules:

* artifact (going with MLOps nomenclature): houses the code for handling the data and the metadata
* infra: short for infrastructure, this holds the code for provisioning cloud GPUs and scale out computations massively
* model: we put the model training and inference code here

This organization reflects the fact that all machine learning projects involve these three components. It's worth noting that there is complete separation of concerns between these 3 modules. An example of why this is important is the following common workflow: 

* Step 1: run training code locally for one or two epochs on a smaller sample of the training data. 
* Step 2: run training code on a beefier machine in the cloud, for the real number of epochs and on the full data.

The reason that this is a pretty good workflow is that we don't want to wait for everything to get uploaded to the cloud and the remote machine warmed up, only to fail due to a bug. So the local run is to check that the code works.

With this intro to inflame-deeplearning, let's dive deeper into the project architecture.

## 2. Project Architecture
The directory structure for the model module looks like the following:
```
inflame-deeplearning/model
├── __init__.py
├── aml-curated-requirements.txt
└── fowl_classifier
    ├── __init__.py
    ├── app.py
    ├── model-config.toml
    ├── score.py
    ├── train.py
    └── utils.py
```
I put aml-curated-requirements.txt at the module root level, because several of the cloud MLOps frameworks bundle up and upload the module folder (I'm specifically referring to Azure ML and Google Cloud's Vertex AI). Notice also that I named it aml-curated-requirements.txt and not generically requirements.txt; the reason is that I left out of this file cloud platform related packages such as azure-ai-ml, because it comes pre-installed in the cloud container or virtual machine.

Under the model module, I have only one model example right now, but it is easy to insert many more models into this flat structure without it being cluttered. Here, the example is classifying pictures into turkey or chicken using PyTorch, hence the name of the submodule. Inside this submodule, the entry point is app.py, and for the most part, app.py loads and calls each of the other programs: train.py, score.py.

## 3. Install
```sh
git clone https://github.com/zjohn77/inflame-deeplearning.git
cd inflame-deeplearning
conda env create -n inflame-deeplearning --file environment.yml
```

## 3. Quick Start
The two corpora currently available are the 20 Newsgroups corpus and the BBC News corpus. Fetching a corpus is accomplished using the **getdata** function, which can be imported like so:

```python
from artifact import getdata
```

## 4. Contribute
Any contribution is welcome. To get started:
```sh
cd tests && python -m unittest
```