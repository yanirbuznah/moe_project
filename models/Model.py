from .utils import *


class Model(nn.Module):
    def __init__(self, config: dict, input_shape, output_shape):
        super().__init__()
        model_config: dict = config.get('model')
        self.model = get_model(model_config.get('model'), output_shape, input_shape)
        self.optimizer = get_optimizer(self.model, model_config.get('optimizer'), model_config.get('lr'))
        self.criterion = get_loss(model_config.get('loss'))
        self.metrics = get_metrics(config.get('metrics'), output_shape)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.model(x).argmax(dim=1)

    def loss(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        output = self.forward(x)
        y_hat = output['output'] if isinstance(output, dict) else output
        return self.criterion(y_hat, y)

    def evaluate(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        output = self.forward(x)
        y_hat = output['output'] if isinstance(output, dict) else output
        results = self.metrics(y_hat.argmax(1), y)
        results['loss'] = self.criterion(y_hat, y)
        return results

    def reset_parameters(self, input):
        if hasattr(self.model, 'reset_parameters'):
            self.model.reset_parameters(input)
