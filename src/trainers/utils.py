from collections import defaultdict

import numpy as np
import torch.nn


def SMAPE(test_preds, test_tgts):
  smape = np.abs(test_preds - test_tgts) * 200 / (
      np.abs(test_preds) + np.abs(test_tgts))
  return np.mean(smape)


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
            loss = loss_fn(y_pred, y.squeeze(1))
            total_loss += loss
            predictions.append(y_pred.detach().cpu().numpy())
            true_labels.append(y.detach().cpu().numpy())
    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)
    return total_loss / len(eval_loader), \
           predictions, \
           true_labels


def train_one_knf_epoch(
        train_loader,
        model,
        optimizer,
        rank=-1):

    total_loss = defaultdict(float)
    for x, y, l in train_loader:
        if rank >= 0:
            x = x.to(rank)
            y = y.to(rank)
            l = l.to(rank)
        output = model(x, y, l)
        losses = output["loss"]
        loss = None
        for k, v in losses.items():
            if not loss:
                loss = v
            else:
                loss += v
            total_loss[k] += v.item()
        total_loss["loss"] += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for k, v in total_loss.items():
        total_loss[k] /= len(train_loader)
    return total_loss


def eval_one_knf_epoch(
        eval_loader,
        model,
        rank=0):
    predictions = []
    true_signal = []
    true_labels = []
    pred_labels = []
    total_loss = defaultdict(float)

    with torch.no_grad():
        for x, y, l in eval_loader:
            if rank >= 0:
                x = x.to(rank)
                y = y.to(rank)
                l = l.to(rank)
            output = model(x, y, l)
            lookback = output["lookback"].cpu().numpy()
            lookahead = output["lookahead"].cpu().numpy()
            cls_scores = output["cls_scores"].cpu().numpy()
            losses = output["loss"]
            loss = 0
            for k, v in losses.items():
                total_loss[k] += v.item()
                loss += v.item()
            total_loss["loss"] += loss
            true_signal.append(np.concatenate([x.cpu().numpy(), y.cpu().numpy()], axis=1))
            predictions.append(np.concatenate([lookback, lookahead], axis=-1))
            pred_labels.append(cls_scores)
            true_labels.append(l.detach().cpu().numpy())
        for k, v in total_loss.items():
            total_loss[k] /= len(eval_loader)
    predictions = np.concatenate(predictions)
    true_signal = np.concatenate(true_signal)
    sMAPE = SMAPE(predictions[:,:,-4:], np.swapaxes(true_signal[:, 1:, :], 1, 2)[:,:,-4:])
    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    return total_loss, \
           predictions, pred_labels, \
           true_labels, sMAPE