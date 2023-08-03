import torch


def get_operator(function: str):
    if function in ['+', 'add']:
        return lambda *args: sum(args)
    elif function in ['-', 'sub']:
        return lambda *args: args[0] - sum(args[1:])
    elif function in ['*', 'mul']:
        return lambda *args: torch.prod(torch.stack(args))
    elif function in ['/', 'div']:
        return lambda *args: args[0] / torch.prod(torch.stack(args[1:]))
    elif function == 'max':
        return lambda *args: torch.max(torch.stack(args))
    elif function == 'min':
        return lambda *args: torch.min(torch.stack(args))
    elif function in ['mean', 'avg']:
        return lambda *args: torch.mean(torch.stack(args))
    elif function == 'sum':
        return lambda *args: torch.sum(torch.stack(args))
    else:
        raise NotImplementedError
