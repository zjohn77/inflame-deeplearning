"""
Includes a function to create a model with a sequence classification head from a pre-trained model.
"""
from urllib.request import urlretrieve

import pandas as pd
import torch

from trainer import LOCAL_DATA_PATH, HIDDEN_UNITS, LEARNING_RATE, CLASS_VOCAB

urlretrieve(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    LOCAL_DATA_PATH,
)


datatrain = pd.read_csv(
    LOCAL_DATA_PATH,
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
)

# change string value to numeric
datatrain.loc[datatrain["species"] == "Iris-setosa", "species"] = 0
datatrain.loc[datatrain["species"] == "Iris-versicolor", "species"] = 1
datatrain.loc[datatrain["species"] == "Iris-virginica", "species"] = 2
datatrain = datatrain.apply(pd.to_numeric)

# change dataframe to array
datatrain_array = datatrain.values

# split x and y (feature and target)
xtrain = datatrain_array[:, :4]
ytrain = datatrain_array[:, 4]

input_features = xtrain.shape[1]
num_classes = len(CLASS_VOCAB)

model = torch.nn.Sequential(
    torch.nn.Linear(input_features, HIDDEN_UNITS),
    torch.nn.Sigmoid(),
    torch.nn.Linear(HIDDEN_UNITS, num_classes),
    torch.nn.Softmax(),
)

loss_metric = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)