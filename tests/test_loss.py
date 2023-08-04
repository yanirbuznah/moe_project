import numpy as np
import torch

from losses import *

out = torch.eye(10)
target = torch.LongTensor([1, 1, 1, 1, 3, 4, 4, 1, 2, 0])
kwargs = {'out': out, 'target': target}
loss1 = CrossEntropyLoss()
loss2 = MSELoss()
loss3 = L1Loss()


def check_loss(loss):
    return loss(**kwargs)


def test_add_operator():
    operator = '+'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, (check_loss(loss1) + check_loss(loss2) + check_loss(loss3)).item())


def test_sub_operator():
    operator = '-'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res,(check_loss(loss1) - check_loss(loss2) - check_loss(loss3)).item())


def test_mul_operator():
    operator = '*'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, check_loss(loss1) * check_loss(loss2) * check_loss(loss3).item())


def test_div_operator():
    operator = '/'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, check_loss(loss1) / check_loss(loss2) / check_loss(loss3).item())


def test_max_operator():
    operator = 'max'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, max(check_loss(loss1).item(), check_loss(loss2).item(), check_loss(loss3).item()))


def test_min_operator():
    operator = 'min'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, min(check_loss(loss1).item(), check_loss(loss2).item(), check_loss(loss3).item()))


def test_mean_operator():
    operator = 'mean'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, (check_loss(loss1) + check_loss(loss2) + check_loss(loss3)) / 3.0)


def test_sum_operator():
    operator = 'sum'
    loss = LossWrapper(operator, [loss1, loss2, loss3])
    res = check_loss(loss).item()
    assert np.isclose(res, check_loss(loss1) + check_loss(loss2) + check_loss(loss3))


def test_LossWrapperWithWeights():
    operator = '+'
    weights = [1, 2, 3]
    loss = LossWrapper(operator, [loss1, loss2, loss3], weights)
    res = check_loss(loss).item()
    assert np.isclose(res, (check_loss(loss1) + 2 * check_loss(loss2) + 3 * check_loss(loss3)).item())


def test_combination_loss():
    loss12 = LossWrapper('+', [loss1, loss2])
    loss123 = LossWrapper('+', [loss12, loss3])
    res = check_loss(loss123).item()
    assert np.isclose(res, (check_loss(loss1) + check_loss(loss2) + check_loss(loss3)).item())


def test_combination_loss2():
    loss12 = LossWrapper('+', [loss1, loss2])
    loss123 = LossWrapper('*', [loss12, loss3])
    res = check_loss(loss123).item()
    assert np.isclose(res, (check_loss(loss1) + check_loss(loss2)) * check_loss(loss3).item())


def test_combination_loss_with_weights():
    loss12 = LossWrapper('+', [loss1, loss2], [1, 2])
    loss123 = LossWrapper('+', [loss12, loss3], [2, 3])
    res = check_loss(loss123).item()
    assert np.isclose(res, ((check_loss(loss1) + 2 * check_loss(loss2)) * 2 + 3 * check_loss(loss3)).item())
