from metrics import *
from metrics.ClassificationMetric import ClassificationMetric
from metrics.MOEMetric import MOEMetric, PValue, MOEConfusionMatrix, RouterVSRandomAcc, \
    ExpertEntropy, SuperClassEntropy, SuperClassConfusionMatrix, Consistency, Specialization, NewSpecialization, \
    DeadExperts, AccuracyDiff, SpearmanCorrelation, DeadExpertsNormalized, DeadExpertsSparsed, \
    SuperClassConfusionMatrixNormalized, MOEConfusionMatrixNormalized, MH, SuperClassMH
from models.MOE import MixtureOfExperts


class MetricsFactory:
    def __init__(self, metrics: dict, num_classes: int):
        self.metrics_list = []
        self.num_classes = num_classes
        self.generate_metrics(metrics)
        self.classification_metrics = ClassificationMetric()
        self.moe_metric = MOEMetric()

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
            elif metric.lower() == 'confusionmatrix':
                self.metrics_list.append(ConfusionMatrix())
            elif metric.lower() == 'pvalue':
                self.metrics_list.append(PValue())
            elif metric.lower() == 'moeconfusionmatrix':
                self.metrics_list.append(MOEConfusionMatrix())
                self.metrics_list.append(MOEConfusionMatrixNormalized())
            elif metric.lower() == 'routervsrandomacc':
                self.metrics_list.append(RouterVSRandomAcc())
            elif metric.lower() == 'expertentropy':
                self.metrics_list.append(ExpertEntropy())
            elif metric.lower() == 'superclassentropy':
                self.metrics_list.append(SuperClassEntropy())
            elif metric.lower() == 'superclassconfusionmatrix':
                self.metrics_list.append(SuperClassConfusionMatrix())
                self.metrics_list.append(SuperClassConfusionMatrixNormalized())
            elif metric.lower() == 'consistency':
                self.metrics_list.append(Consistency())
            elif metric.lower() == 'specialization':
                self.metrics_list.append(Specialization())
            elif metric.lower() == 'newspecialization':
                self.metrics_list.append(NewSpecialization())
            elif metric.lower() == 'deadexperts':
                self.metrics_list.append(DeadExperts())
                self.metrics_list.append(DeadExpertsNormalized())
                self.metrics_list.append(DeadExpertsSparsed())
            elif metric.lower() == 'accuracydiff':
                self.metrics_list.append(SpearmanCorrelation())
            elif metric.lower() == 'mh':
                self.metrics_list.append(MH())
            elif metric.lower() == 'superclassmh':
                self.metrics_list.append(SuperClassMH())
            # else:
            #     self.metrics_list.append(MOEMetric())
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
        if 'Consistency' in results and 'Specialization' in results:
            results['Expertise'] = results['Consistency'] * results['Specialization'].mean()
        if 'Consistency' in results and 'NewSpecialization' in results:
            results['NewExpertise'] = results['Consistency'] * results['NewSpecialization']
        return results

    def update_metrics(self, *args, **kwargs):
        self.classification_metrics(*args, **kwargs)
        if isinstance(kwargs['model'], MixtureOfExperts):
            self.moe_metric(*args, **kwargs)

    def reset(self):
        for metric in self.metrics_list:
            metric.reset()
