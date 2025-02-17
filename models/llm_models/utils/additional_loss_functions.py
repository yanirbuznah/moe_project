import torch
from transformers import DynamicCache

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

def compute_additional_loss(model, outputs, batch):
    gain_losses = []

    for i in range(1, len(outputs.hidden_states) - 1):
        batch_new = batch.copy()
        batch_new.update(
            {
                "inputs_embeds": outputs.hidden_states[i].detach(),  # Fix: Detach to avoid modifying computation graph
                "decoder_layer_idx": i,
                "past_key_values": DynamicCache(),
                'use_cache': False,
                'random_routing': 'random_weighted'
            }
        )
        del batch_new["input_ids"]

        model.eval()
        with torch.no_grad():
            outputs_new = model(**batch_new)
        model.train()
        gain_loss = outputs.loss - outputs_new.loss
        if gain_loss > 0:
            AddAuxiliaryLoss(model.model.layers[i].mlp.gate.logits, gain_loss)
            gain_losses.append(gain_loss)

    return torch.stack(gain_losses).mean() if gain_losses else torch.tensor(-1., device=model.device)