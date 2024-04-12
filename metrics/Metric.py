import torch


class Metric:
    def __init__(self):
        self.stat = 0.0
        self.possible_y_pred = ['y_pred', 'output', 'y_hat', 'out']
        self.possible_y_true = ['target', 'y_true']
        self.possible_x = ['x', 'input', 'inp', 'X', 'Input', 'Inp']
        self.possible_model = ['model', 'm', 'Model', 'M']
        self.possible_routes_probs = ['router_probs', 'rp']
        self.possible_counts = ['count', 'cnt', 'c', 'Count', 'Cnt', 'C', 'Counts', 'Cnts', 'counts']
        self.possible_routes = ['routes', 'route', 'r', 'Routes', 'Route', 'R']
        self.possible_super_classes = ['super_targets', 'super_classes', 'sc', 'Super_classes', 'SC', 'Super_Classes', 'super_Classes']

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__} with {self.stat:.2f}'

    def compute(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_name(self):
        return self.__class__.__name__

    @staticmethod
    def _preprocess_args(*args):
        if isinstance(args[0], torch.Tensor):
            args = [arg.detach().cpu().numpy() for arg in args]
        return args
