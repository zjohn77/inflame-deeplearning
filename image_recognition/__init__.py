from .hyperparams import *
from .model_architecture import *
from .preprocess import *

import torch

from config import DEVICE


def trainer(dataloader, model, loss_fn, learn_rate, epoch, device):
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learn_rate,
    )
    num_batches = len(dataloader)

    model.train()
    running_loss = 0.0

    for i, (X, y) in enumerate(dataloader, 0):
        batch_size = len(X)
        num_processed_samples = (i + 1) * batch_size

        # get the inputs; data is a list of [inputs, labels]
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # print every ~8000 samples
        if num_processed_samples % 8000 < batch_size:
            step_str = f"[{epoch + 1}, {i + 1:5d}/{num_batches}]"
            print(f"{step_str} loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

    return model


def tester(dataloader, model, loss_fn, device):
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, num_correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_loss_per_batch = test_loss / num_batches
    perc_correct = num_correct / num_samples * 100
    print(f"Test Error: \n Accuracy: {perc_correct :>0.1f}%, "
          f"Avg loss: {avg_loss_per_batch:>8f} \n")


def training_loop():
    trainloader, testloader, _ = preprocess()

    for _epoch in range(EPOCHS):
        print(f"Epoch {_epoch + 1}\n-------------------------------")
        _model = trainer(
            dataloader=trainloader,
            model=ConvNet,
            loss_fn=LOSS_FN,
            learn_rate=LEARN_RATE,
            epoch=LEARN_RATE,
            device=DEVICE,
        )
        tester(dataloader=testloader,
               model=_model,
               loss_fn=LOSS_FN,
               device=DEVICE)
