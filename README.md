# 🔥 The inflame-deeplearning Trainer App

## Table of Contents
1. [Basic Idea and Key Benefits](#1-basic-idea-and-key-benefits)
2. [Code Walkthrough](#2-code-walkthrough)
3. [Install and Get Started](#3-install-and-get-started)
4. [Contribute](#4-contribute)

## 1. Basic Idea and Key Benefits:
To build world-class deep learning models, it is necessary to do two things:

* Train on state-of-the-art GPUs.
* Experiment and iterate lots and lots of times, by throwing fresh data, parameters, ideas, and insights at the problem--rinse, repeat, on an industrial scale, until the model accuracy is really really good. 

That is the essence of what this project (inflame-deeplearning) is trying to facilitate. The bottom line is that this is to enable you to train models on GPUs in the cloud and also to extend the training code into an experimentation factory for effortlessly trying different variations of your model until the sauce is just right. At its core, the inflame-deeplearning project is model training code re-architected as an app. Let's walk through what you need to know to get started using, customizing, and extending it for your own deep learning project. 

## 2. Code Walkthrough
### 📝 Some Conventions to Know
* Functions are categorized as either helper functions or APIs. Python's single underscore protected notation is used here to tag a function as a helper function.  
* `__init__.py` exposes the public APIs of the module it is in so that we can conveniently shorten imports for deeply submoduled functionality like this example:
```python
from .dataset import split_datafiles
```
in `artifact/__init__.py` enables code outside artifact to import split_datafiles without having to remember the breadcrumbs leading to where this function sits, like this:
```python
from artifact import split_datafiles
```
* The entry point of a module is designed to be readily apparent; when there are several py files on the same level in a directory, the entry point is either `app.py` or `main.py`, in accordance with python convention.

### 🚶 A Tour of the Project
Starting from the project root, there are essentially 4 modules:

* `artifact` (going with MLOps nomenclature): houses the code for handling the data and the metadata
* `infra`: we put the code for the cloud infrastructure here 
* `model`: the model training and inference code are here
* `tests`: the unit tests go here

Let's walk through each one of these modules in turn, starting with the artifact, the structure of which looks like below:

```
inflame-deeplearning/artifact
├── __init__.py
├── cli.py
└── dataset
    ├── __init__.py
    └── random_data_split.py
```

In here, we've got a basic `dataset` module. We can fulfill many purposes with this module, such as sourcing imagery or textual data from their storage location, converting them to Arrow or Tensor representations for further transformations in PyTorch or HuggingFace. Or, this could be where our pandas exploratory data analysis goes. I've placed here a utility function for creating training, validation, and test splits directly from a folder containing the flat files that I want to split. The code with this function is in [random_data_split.py](https://github.com/zjohn77/inflame-deeplearning/blob/main/artifact/dataset/random_data_split.py), and this code should hopefully give an idea about what the `artifact` module is for.

Next, we are going to talk about the `infra` module, and this is its structure:
```
inflame-deeplearning/infra
├── __init__.py
└── azureml
    ├── __init__.py
    ├── blob_storage
    │   ├── __init__.py
    │   ├── access.py
    │   └── fs_access.py
    ├── job-config.toml
    └── training_job.py
```
The submodule `azureml` is a template that gives a taste of what the code looks like for one specific cloud AI platform--Azure ML in this case. It is meant to be extended to other platforms, which would each have its own folder here. In its end state, I envision we would have folders for: Azure ML, SageMaker, ClearML, and Vertex AI. This would let us provision resources on multiple infrastructures to take advantage of the best features of each one.

It's worth noting the separation of concerns between the `infra` module and the `model` module; in fact, this allows us to do things like run training on our local machine one day and then run it in the cloud the next, without making any changes to our training code. That is at the heart of what the inflame-deeplearning project can do--the raison d'être for it. 

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
Again, there is some stylized starter code here to represent what a working `model` module should look like, and in its end state, you'd want to extend this by adding many more models, each in its own folder with its own set of dependencies. The overarching concept is that we are training machine learning model in the same manner as developing apps--by building small, modular components and combining them using an architecture that coheres some elements and decouples others. This style of working is different from that of a jupyter notebook, and has both advantages and disadvantages to the latter. But one way of looking at it is that the two styles can complement each other.

With this intro to the code, let's get into the nuts and bolts of training PyTorch deep learning models with this project. What we are going to do is install this project locally, run the training app for one epoch to make sure it works, and then set up what I call the "glue code" to connect to an end-to-end cloud MLOps service that lets us run our code on GPUs remotely. Examples of such services: Azure ML, SageMaker, ClearML, Vertex AI.

## 3. Install and Get Started
The first step is to clone this repository into a local working directory and then change into the root of the cloned repo:
```sh
git clone https://github.com/zjohn77/inflame-deeplearning.git
cd inflame-deeplearning
```

Next, we are going to create a virtual environment using conda from `environment.yml`; ideally, we'd want our local virtual environment to mirror our cloud runtime as perfectly as possible, which was why I pinned python to 3.9 and torch to 1.12.1, as those are the exact versions on the pre-built container on Azure ML that I want to use. Your configurations will likely vary.
```sh
conda env create -n inflame-deeplearning --file environment.yml
```

There is some housekeeping to do with pointing the training code to data on your own computer, and the `model-config.toml` file enables you to switch parameters in and out easily. This file has a section for hyperparameters and a section for IO. The IO section contains the following variables: `training_input`, `training_output`, `inference_input`, and `inference_output`.

To run our example model, we are going to download and unzip this [bird images dataset](https://azuremlexamples.blob.core.windows.net/datasets/fowl_data.zip), and point the `training_input` variable to it. Then, point the `inference_input` variable to whatever test image you like. It's easy. Now, we're ready to run kick off a training run with `app.py`, and the way that this module works is that it is really just like a telephone switchboard. The actual training loop is in `train.py`; abstracting the interface from the implementation in this way allows us to easily swap out that training loop in `train.py` with another, totally unrelated, neural network architecture without worrying about breaking any communications between `app.py` and `model-config.toml`, or between the `model` module and the `infra` module. As a matter of fact, the `infra` module is aware of only one thing about the `model` module: that there is a script called `app.py` in the model module. Arguably, this project (inflame-deeplearning) wins against jupyter notebook on this point.

In your virtual environment, you'll change into or point your IDE to where `app.py` is and run:
```python
python app.py
```
It will log the loss and the accuracy metric for each epoch, and at the end it will print to console the following json (the values may be different):

```
{"label": "turkey", "probability": "0.7779676"}
```

You have trained your first model using inflame-deeplearning! And that wasn't so bad. Where to go from here? A tutorial of the scale out to cloud code in the `infra` module is work in progress. Check back soon.

## 4. Contribute
Any contribution is welcome. We use GitHub pull requests for code review, and we use [the Black formatter](https://black.readthedocs.io/en/stable/). To ensure that the unit tests pass:
```sh
python -m unittest
```