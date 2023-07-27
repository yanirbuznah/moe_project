import torch


class Metric:

    def compute(self):
        pass

    def reset(self):
        pass

    def get_name(self):
        return self.__class__.__name__

    @staticmethod
    def _preprocess_args(*args):
        if isinstance(args[0], torch.Tensor):
            args = [arg.detach().cpu().numpy() for arg in args]
        return args
