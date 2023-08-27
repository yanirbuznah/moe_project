from .utils import *


class Model(nn.Module):
    def __init__(self, config: dict, output_shape, train_set=None):
        super().__init__()
        self.config = config
        model_config: dict = config.get('model')
        self.model = get_model(model_config, output_shape, train_set)
        self.optimizer = get_optimizer(self.model, model_config.get('optimizer'), model_config.get('lr'))
        self.criterion = get_loss(model_config.get('loss'))
        self.metrics = get_metrics(config.get('metrics'), output_shape)
        self.train_set = train_set

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def train_set(self):
        return self._train_set

    @train_set.setter
    def train_set(self, value):
        self._train_set = value

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.model(x).argmax(dim=1)

    def loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        output = self.forward(x)
        if isinstance(output, torch.Tensor):
            output = {'y_pred': output}
        output['target'] = y
        return self.criterion(**output)

    def evaluate(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        output = self.forward(x)
        if isinstance(output, torch.Tensor):
            output = {'output': output}
        kwargs = output.copy()
        kwargs['target'] = y
        kwargs['model'] = self.model
        kwargs['input'] = x
        self.metrics(**kwargs)
        return self.criterion(**kwargs)

    def compute_metrics(self):
        return self.metrics.compute()

    def reset_metrics(self):
        self.metrics.reset()

    def reset_parameters(self, input):
        if hasattr(self.model, 'reset_parameters'):
            self.model.reset_parameters(input)

    def get_loss_repr(self):
        return self.criterion.__repr__()


