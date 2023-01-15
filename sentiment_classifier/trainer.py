"""
Includes a function to create a model with a sequence classification head from a pre-trained model.
"""
from urllib.request import urlretrieve

import pandas as pd
import torch
from torch.autograd import Variable

from sentiment_classifier.config import (
    HIDDEN_UNITS,
    LEARNING_RATE,
    CLASS_VOCAB,
    NUM_EPOCHS,
    MODEL_SAVE_PATH,
)


def process_data(data_save_path):
    # try:
    #     urlretrieve(
    #         "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    #         DATA_SAVE_PATH,
    #     )
    # except Exception:
    #     print(Exception)

    datatrain = pd.read_csv(
        data_save_path,
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

    return xtrain, ytrain, input_features, num_classes


def train(xtrain, ytrain, input_features, num_classes):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_features, HIDDEN_UNITS),
        torch.nn.Sigmoid(),
        torch.nn.Linear(HIDDEN_UNITS, num_classes),
        torch.nn.Softmax(),
    )

    loss_metric = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        x = Variable(torch.Tensor(xtrain).float())
        y = Variable(torch.Tensor(ytrain).long())
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_metric(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 1:
            print(
                "Epoch [{}/{}] Loss: {}".format(epoch + 1, NUM_EPOCHS, round(loss.item(), 3))
            )

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    example = torch.rand(1, 4)
    traced_script_module = torch.jit.trace(model, example)

    # Save the TorchScript model
    try:
        traced_script_module.save(MODEL_SAVE_PATH)
        return MODEL_SAVE_PATH
    except Exception:
        print(Exception)