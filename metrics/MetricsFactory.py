from metrics import *


class MetricsFactory:
    def __init__(self, metrics: dict, num_classes: int):
        self.metrics_list = []
        self.num_classes = num_classes
        self.generate_metrics(metrics)

    def generate_metrics(self, metrics):
        for metric in metrics:
            if metric.lower() == 'accuracy':
                self.metrics_list.append(Accuracy())
            elif metric.lower() == 'precision':
                self.metrics_list.append(Precision())
            elif metric.lower() == 'recall':
                self.metrics_list.append(Recall())
            elif metric.lower() == 'f1':
                self.metrics_list.append(F1())
            else:
                raise NotImplementedError(f"Metrics {metric} not implemented")

    def __call__(self, *args, **kwargs):
        self.update_metrics(*args, **kwargs)

    def compute(self):
        return self.compute_metrics()

    def compute_metrics(self):
        results = {}
        for metric in self.metrics_list:
            results[metric.get_name()] = metric.compute()
        return results

    def update_metrics(self, *args, **kwargs):
        for metric in self.metrics_list:
            metric(*args, **kwargs)

    def reset(self):
        for metric in self.metrics_list:
            metric.reset()
