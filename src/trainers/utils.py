import numpy as np
import torch.nn


def train_one_epoch(train_loader,
                    model,
                    optimizer,
                    rank=0):
    """Train the KNF model for one epoch.

  Args:
    train_loader: the dataloader of the training set
    model: KNF model
    loss_fun: loss function
    optimizer: Adam
    rank: rank of the device (-1 if cpu)

  Returns:
    RMSE on the training set
    :param rank:

  """
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for x, y in train_loader:
        if rank >= 0:
            x = x.to(rank)
            y = y.to(rank)
        y_pred = model(x)
        loss = loss_fn(y_pred, y.squeeze(1))
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)


def eval_one_epoch(eval_loader, model, rank=0):
    """

    :param eval_loader:
    :param model:
    :param loss_fun:
    :param rank:
    :return:
    """

    predictions = []
    true_labels = []
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in eval_loader:
            if rank >= 0:
                x = x.to(rank)
                y = y.to(rank)
            y_pred = model(x)
            # loss = loss_fn(y_pred, y.squeeze(1))
            # total_loss += loss
            predictions.append(y_pred.detach().cpu().numpy())
            true_labels.append(y.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)
    return total_loss / len(eval_loader), \
           predictions, \
           true_labels
