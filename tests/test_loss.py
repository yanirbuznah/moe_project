from torch import nn
import numpy as np
from losses.LossWrapper import LossWrapper, LossWrapperWithWeights
import torch
import pytest


out = torch.eye(10)
target = torch.LongTensor([1, 1, 1, 1, 3, 4, 4, 1, 2, 0])
loss1 = nn.CrossEntropyLoss()
loss2 = nn.MSELoss()
loss3 = nn.L1Loss()
def check_loss(loss):
    return loss(torch.eye(10), torch.LongTensor([1, 1, 1, 1, 3, 4, 4, 1, 2, 0]))
def test_add_operator():
    operator = '+'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, (loss1(out, target) + loss2(out, target) + loss3(out, target)).item())

def test_sub_operator():
    operator = '-'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, (loss1(out, target) - loss2(out, target) - loss3(out, target)).item())

def test_mul_operator():
    operator = '*'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, (loss1(out, target) * loss2(out, target) * loss3(out, target)).item())

def test_div_operator():
    operator = '/'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, (loss1(out, target) / loss2(out, target) / loss3(out, target)).item())

def test_max_operator():
    operator = 'max'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, max(loss1(out, target).item(), loss2(out, target).item(), loss3(out, target).item()))

def test_min_operator():
    operator = 'min'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, min(loss1(out, target).item(), loss2(out, target).item(), loss3(out, target).item()))

def test_mean_operator():
    operator = 'mean'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, (loss1(out, target).item() + loss2(out, target).item() + loss3(out, target).item()) / 3)

def test_sum_operator():
    operator = 'sum'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, loss1(out, target).item() + loss2(out, target).item() + loss3(out, target).item())


def test_LossWrapperWithWeights():
    operator = '+'
    weights = [1, 2, 3]
    loss = LossWrapperWithWeights(operator, [loss1, loss2, loss3], weights)
    res = check_loss(loss).item()
    assert np.isclose(res, (loss1(out, target) + 2 * loss2(out, target) + 3 * loss3(out, target)).item())


def test_combination_loss():
    loss12 = LossWrapper('+', [loss1, loss2])
    loss123 = LossWrapper('+', [loss12, loss3])
    res = check_loss(loss123).item()
    assert np.isclose(res, (loss1(out, target) + loss2(out, target) + loss3(out, target)).item())

def test_combination_loss2():
    loss12 = LossWrapper('+', [loss1, loss2])
    loss123 = LossWrapper('*', [loss12, loss3])
    res = check_loss(loss123).item()
    assert np.isclose(res, (loss1(out, target) + loss2(out, target)) * loss3(out, target)).item()


def test_combination_loss_with_weights():
    loss12 = LossWrapperWithWeights('+', [loss1, loss2], [1, 2])
    loss123 = LossWrapperWithWeights('+', [loss12, loss3], [2, 3])
    res = check_loss(loss123).item()
    assert np.isclose(res, ((loss1(out, target) + 2 * loss2(out, target))*2 + 3 * loss3(out, target)).item())

