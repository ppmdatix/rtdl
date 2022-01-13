from typing import Any, Dict

import numpy as np
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import zero
import pandas as pd


device = torch.device('cpu')

progress = zero.ProgressTracker(patience=100)

def apply_model(x_num, x_cat=None, model=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(part, model, X, y, y_std, task_type="regression"):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 1024):
        prediction.append(apply_model(batch, model=model))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
    return score


def learn_that(_model, _optimizer, _loss_fn, _X, _y, y_std, _epochs, _batch_size, _relational_batch, _old_X, print_mode=False, _task_type="regression"):
    # Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
    batch_size = 256
    train_loader = zero.data.IndexLoader(len(_X['train']), batch_size, device=device)

    # Create a progress tracker for early stopping
    # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
    _progress = zero.ProgressTracker(patience=100)

    if print_mode:
        print(f'Test score before training: {evaluate("test", _model):.4f}')

    report_frequency = len(_X['train']) // _batch_size // 5
    losses = dict()
    losses['val'] = []
    losses['test'] = []
    for epoch in range(1, _epochs + 1):
        for iteration, batch_idx in enumerate(train_loader):
            _model.train()
            _optimizer.zero_grad()
            x_batch = _X['train'][batch_idx]
            y_batch = _y['train'][batch_idx]

            loss = _loss_fn(apply_model(x_batch, model=_model).squeeze(1), y_batch)
            loss.backward()

            factors = dict()

            # Modify gradients
            if _relational_batch:
                for name, param in _model.named_parameters():
                    if name == "blocks.0.linear.weight":
                        column_count = len(_old_X['train'].columns)
                        factors = torch.ones(column_count,param.grad.shape[0])
                        for i in range(column_count):
                            column = _old_X['train'].columns[i]
                            if True:  # not column in oldNames:
                                idx = _old_X['train'][iteration * batch_size:(iteration+1) * batch_size].columns[i]
                                realCount = _old_X['train'][iteration * batch_size:(iteration+1) * batch_size][idx].sum()
                                if realCount > 0:
                                    factors[i] = (batch_size / (1.0 * realCount)) * factors[i]
                                else:
                                    ()
                                    # factors[i] = float('nan') * factors[i]
                        param.grad = torch.mul(param.grad, torch.transpose(factors,0,1))

            _optimizer.step()
            if iteration % report_frequency == 0:
                batch = "batch"
                if _relational_batch:
                    batch= "relational-batch"
                if print_mode:
                    print(f'(epoch) {epoch} ({batch}) {iteration} (loss) {loss.item():.4f}')

        losses['val'].append(float(_loss_fn(apply_model(_X['val'],model=_model).squeeze(1), _y['val'])))
        losses['test'].append(float(_loss_fn(apply_model(_X['test'],model=_model).squeeze(1), _y['test'])))

        val_score  = evaluate("val", _model, _X, _y, y_std, task_type=_task_type)
        test_score = evaluate("test", _model, _X, _y, y_std, task_type=_task_type)
        _progress.update((-1 if _task_type == 'regression' else 1) * val_score)
        if _progress.fail:
            break
        if print_mode:
            print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
            if _progress.success and print_mode:
                print(' <<< BEST VALIDATION EPOCH', end='')
    return losses