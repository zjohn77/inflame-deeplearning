import torch


def predict_class(instances, classifier):
    instances = torch.Tensor(instances)
    output = classifier(instances)
    _, predicted = torch.max(output, 1)
    return predicted
