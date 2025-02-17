import torch

from losses import Loss


class DiversityLoss(Loss):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.full_model_parameters = kwargs.get('full_model_parameters', False)

    def __call__(self, *args, **kwargs):
        self.stat = self.calc(self.model.experts)
        return self.stat

    @staticmethod
    def calc(models, full_model_parameters=True):
        loss_orth = 0.0
        num_models = len(models)

        # Iterate over all pairs of models (i < j)
        for i in range(num_models):
            for j in range(i + 1, num_models):
                model1, model2 = models[i], models[j]

                models_parameters = zip(model1.parameters(), model2.parameters()) if full_model_parameters else \
                    zip(model1.fc.parameters(), model2.fc.parameters())

                # Iterate over the parameters of both models
                for param1, param2 in models_parameters:
                    # Check if the parameters are weight matrices (exclude biases)
                    if len(param1.shape) == 2:  # Weight matrix (not bias)
                        # Compute the Gram matrix (dot product between the weight matrices)
                        gram_matrix = torch.matmul(param1.T, param2)  # Shape: [output_dim, output_dim]

                        # Add the Frobenius norm squared of the Gram matrix to the regularization loss
                        loss_orth += torch.norm(gram_matrix, p='fro') ** 2

            return loss_orth
