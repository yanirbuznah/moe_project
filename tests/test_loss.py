import numpy as np
import torch
from torch import nn
from losses import *
from torch.autograd import Variable

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


def test_loss_backward():
    # Define your model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    # Define the loss functions
    loss_fn1 = nn.MSELoss()
    loss_fn2 = nn.L1Loss()

    # Create the model
    model = SimpleModel()

    # Generate some random input data and target
    input_data = torch.randn(5, 10)
    target = torch.randn(5, 1)

    # Wrap input and target in Variables (for gradient tracking)
    input_data = Variable(input_data, requires_grad=True)
    target = Variable(target, requires_grad=False)


    # Compute the two losses
    output = model(input_data)
    loss1 = loss_fn1(output, target) + loss_fn2(output, target)

    # Compute gradients using autograd
    loss1.backward(retain_graph=True)
    gradients1 = input_data.grad.clone()
    input_data.grad.zero_()  # Clear gradients for the next loss

    loss12 = LossWrapper('+', [MSELoss(), L1Loss()])

    loss2 = loss12(**{'out':output, 'target':target})

    loss2.backward()
    gradients2 = input_data.grad.clone()

    # Compare gradients
    gradient_difference = torch.norm(gradients1 - gradients2).item()

    # Define a threshold for comparison
    gradient_threshold = 1e-8

    # Perform the test
    assert gradient_difference < gradient_threshold

    # loss1 = torch.nn.CrossEntropyLoss()
    # loss2 = torch.nn.MSELoss()
    # inp = torch.randn(10, 10)
    # model = torch.nn.Linear(10, 10)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # model_state_dict_before = model.state_dict()
    # loss12 = LossWrapper('+', [loss1, loss2])
    # y = model(inp)
    #
    # optimizer.zero_grad()
    # loss = loss1(y, target) + loss2(y, target)
    # loss.backward()
    # optimizer.step()
    #
    # model_state_dict_after = model.state_dict()
    # model.load_state_dict(model_state_dict_before)
    #
    # y = model(inp)
    # optimizer.zero_grad()
    # loss = loss12(kwargs)
    # loss.backward()
    # optimizer.step()
    #
    # for key in model_state_dict_before.keys():
    #     assert torch.allclose(model_state_dict_before[key], model_state_dict_after[key])


