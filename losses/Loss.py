class Loss:
    def __init__(self):
        self.stat = 0.0
        self.init_possible_input_options()

    def init_possible_input_options(self):
        self.possible_y_pred = ['y_pred', 'output', 'y_hat', 'out']
        self.possible_y_true = ['target', 'y_true']
        self.possible_route_probabilities = ['route_probabilities', 'route_prob', 'route_probs', 'router_probs']
        self.possible_counts = ['counts']

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__} with {self.stat:.2f}'
