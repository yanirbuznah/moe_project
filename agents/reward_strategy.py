import numpy as np
import torch

# from utils.general_utils import device


class RewardStrategy:
    def __init__(self, reward_type, num_of_experts, num_of_classes=10):
        self.reward_type = reward_type
        self.num_of_experts = num_of_experts
        self.num_of_classes = num_of_classes
        self.cf_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def get_reward_function(self):
        if self.reward_type == 'CrossEntropy':
            return self._get_cross_entropy_loss_reward
        elif self.reward_type == 'BestFromAll':
            return self._get_best_from_all_reward
        elif self.reward_type == 'DiffFromRandom':
            return self._get_diff_from_random_reward
        elif self.reward_type == 'CrossEntropyProbabilities':
            return self._get_ce_dot_probability_reward
        elif self.reward_type == 'CrossEntropyProbabilitiesWithMax':
            return self._acc_dot_probs
        elif self.reward_type == 'CrossEntropyProbabilitiesWithTanh':
            return self.acc_and_ce_dot_probs_with_tanh
        else:
            raise NotImplementedError

    def _get_cross_entropy_loss_reward(self, sample, action, model):
        model.eval()
        out, y = self._get_output_from_model(action, model, sample)
        reward = -self.cf_entropy(out, y)
        return reward

    def _get_output_from_model(self, action, model, sample):
        model.eval()
        with torch.no_grad():
            x, y, action = sample[0].to(model.device), sample[1].to(model.device), action.to(model.device)
            out = model.get_unsupervised_output(x, routes=action)
        return out, y

    def _get_best_from_all_reward(self, sample, action, model):
        model.eval()
        with torch.no_grad():
            x, y, action = sample[0].to(model.device), sample[1].to(model.device), action.to(model.device)
            all_outs = [model.experts[i](x) for i in range(self.num_of_experts)]
            out = model.forward_unsupervised(x, routes=action)[0]
        all_outs_cross_entropy = [self.cf_entropy(all_outs[i], y) for i in range(self.num_of_experts)]
        best_from_all_outs = torch.min(torch.stack(all_outs_cross_entropy), dim=0)[0]
        reward = best_from_all_outs - self.cf_entropy(out, y)
        return reward

    def _get_diff_from_random_reward(self, sample, action, model):
        model.eval()
        with torch.no_grad():
            x, y, action = sample[0].to(model.device), sample[1].to(model.device), action.to(model.device)
            out = model.forward_unsupervised(x, routes=action)[0]
            random_out = model.experts[torch.randint(0, self.num_of_experts, (1,))](x)
        reward = self.cf_entropy(out, y) - self.cf_entropy(random_out, y)
        return reward

    def _get_probabilities_per_expert(self, action):
        routes = action.cpu().numpy()
        probabilities = np.zeros(self.num_of_experts)
        for expert in routes:
            probabilities[expert] += 1
        probabilities = probabilities / len(routes)
        return probabilities

    def _get_ce_dot_probability_reward(self, sample, action, model, out = None, y = None):
        probs = self._get_probabilities_per_expert(action)
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        ce = self.cf_entropy(out, y)
        rewards = [-ce[i] * probs[action[i]] for i in range(len(ce))]
        rewards = torch.stack(rewards)
        return rewards

    def _acc_dot_probs(self, sample, action, model, max_prob=0.3, out = None, y = None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds,action.detach(),y)

        acc = preds == y
        rewards = [acc[i] * min(max_prob, load[action[i], y[i]]) * C[action[i], y[i]] for i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def _ce_dot_probs_tanh(self, sample, action, model, out = None, y = None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds,action.detach(),y)

        ce = 1 / self.cf_entropy(out, y)
        rewards = [self._tanh(ce,0.01) * self._tanh(load[action[i], y[i]], 5.5) * C[action[i], y[i]] for i in range(len(ce))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def _acc_dot_probs_tanh(self, sample, action, model ,out = None, y = None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds,action.detach(), y)

        acc = preds == y
        rewards = [acc[i] * self._tanh(load[action[i], y[i]], 5.5) * C[action[i], y[i]] for i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def acc_and_ce_dot_probs_with_tanh(self, sample, action, model, out = None, y = None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds,action.detach(), y)

        acc = preds == y
        ce = 1 / self.cf_entropy(out, y)
        rewards = [10 * self._tanh(ce[i],1) + acc[i] * self._tanh(load[action[i], y[i]], 5.5) * C[action[i], y[i]] for i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards
    def _acc_dot_probs_1_minus_prob(self, sample, action, model,out = None, y = None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds,action.detach(), y)
        ce = self.cf_entropy(out, y)
        acc = preds == y
        rewards = [-ce[i] * (1 - load[action[i], y[i]]) * C[action[i], y[i]] for i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards
    def _get_C_matrix(self, preds, routes, true_assignments):
        C = np.zeros((self.num_of_experts, self.num_of_classes))
        load = np.zeros((self.num_of_experts, self.num_of_classes))
        for i in range(len(preds)):
            C[routes[i], true_assignments[i]] += int(true_assignments[i] == preds[i])
            load[routes[i], true_assignments[i]] += 1

        C = C / np.maximum(load, 1) #
        load = load / load.sum(axis=0, keepdims=True)
        return load, C

    # A: Aij = class j goes to expert i
    def _get_A_matrix(self, true_assignments, routes):
        import utils.general_utils as ut
        x = ut.calc_confusion_matrix(true_assignments, routes)
        x = x / x.sum(axis=1, keepdims=True)
        return x


    def _tanh(self, x, alpha=2.5):
        return torch.tanh(alpha * x)



# -CE * max(0.5, p)
# -CE * p   (p = probability of expert)
# right_on * max(0.5, p)
# C : Cij = probability of expert i for class j (right on class j)
# A: Aij = class j goes to expert i
# rij = cij* min(Aij,t) /
# maybe use tanh and not min (so it will be more smooth)
# check whether it is better to multiply the reward by the (1-probability) of the expert, so the reward will be higher for experts with low probability
# what about connection between 2 experts? (if they are close to each other, they should be similar)

# check different alphas in tanh
# what about remove the replay buffer and just use the current batch?