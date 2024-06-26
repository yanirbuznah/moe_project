import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from metrics.MOEMetric import Specialization, Consistency, NewSpecialization
from models.MOE import MixtureOfExperts


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
        elif self.reward_type == 'AccWithTanh':
            return self._acc_dot_probs_tanh
        elif self.reward_type == 'CrossEntropyWithTanh':
            return self._ce_dot_probs_tanh
        elif self.reward_type == 'SpecializationAndConsistency':
            return self._specialization_and_consistency
        elif self.reward_type == 'Entropy':
            return self._entropy
        elif self.reward_type == 'RegretBased':
            return self._regret_based_reward
        elif self.reward_type == 'Expertise':
            return self._new_expertise_reward
        elif self.reward_type == 'SpearmanCorrelation':
            return self._spearman_correlation_reward
        else:
            return self._proposal_specialization_and_consistency

    def _get_cross_entropy_loss_reward(self, sample, action, model):
        model.eval()
        out, y = self._get_output_from_model(action, model, sample)
        reward = -self.cf_entropy(out, y)
        return reward

    def _get_output_from_model(self, action, model, sample):
        model.eval()
        with torch.no_grad():
            x, y, action = sample[0].to(model.device), sample[1].to(model.device), action.to(model.device)
            x = model.encoder(x)
            out = model.get_unsupervised_output(x, routes=action)
        return out, y

    def _get_routes_from_model(self, model, sample):
        model.eval()
        with torch.no_grad():
            x = sample[0].to(model.device)
            x = model.encoder(x)
            out = model.router(x)
        return out

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

    def _get_ce_dot_probability_reward(self, sample, action, model, out=None, y=None):
        probs = self._get_probabilities_per_expert(action)
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        ce = self.cf_entropy(out, y)
        rewards = [-ce[i] * probs[action[i]] for i in range(len(ce))]
        rewards = torch.stack(rewards)
        return rewards

    def _acc_dot_probs(self, sample, action, model, max_prob=0.3, out=None, y=None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds, action.detach(), y)

        acc = preds == y
        rewards = [acc[i] * min(max_prob, load[action[i], y[i]]) * C[action[i], y[i]] for i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def _ce_dot_probs_tanh(self, sample, action, model, out=None, y=None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds, action.detach(), y)
        acc = preds == y
        ce = self.cf_entropy(out, y)
        rewards = [-ce[i] + (acc[i] * self._tanh(load[action[i], y[i]] * C[action[i], y[i]], 5.5)) for i in
                   range(len(ce))]
        # rewards = [self._tanh(ce,0.01) * self._tanh(load[action[i], y[i]], 5.5) * C[action[i], y[i]] for i in range(len(ce))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def _acc_dot_probs_tanh(self, sample, action, model, out=None, y=None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds, action.detach(), y)

        acc = preds == y
        rewards = [acc[i] * (self._tanh(load[action[i], y[i]] * C[action[i], y[i]], 5.5) / load[action[i], y[i]]) for i
                   in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def acc_and_ce_dot_probs_with_tanh(self, sample, action, model, out=None, y=None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds, action.detach(), y)

        acc = preds == y
        ce = 1 / self.cf_entropy(out, y)
        rewards = [
            10 * self._tanh(ce[i].item(), 1) + acc[i] * self._tanh(load[action[i], y[i]], 5.5) * C[action[i], y[i]] for
            i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def _acc_dot_probs_1_minus_prob(self, sample, action, model, out=None, y=None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        load, C = self._get_C_matrix(preds, action.detach(), y)
        ce = self.cf_entropy(out, y)
        acc = preds == y

        rewards = [acc[i] * (1 - load[action[i], y[i]]) * C[action[i], y[i]] for i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def _specialization_and_consistency(self, sample, action, model, out=None, y=None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        consistency, specialization = self._get_C_matrix(preds, action.detach(), y)
        load = torch.FloatTensor(consistency.sum(axis=1)) / self.num_of_classes
        acc = preds == y
        # output_of_true_class = out[torch.arange(len(out)), y]
        # rewards = [
        # output_of_true_class[i] * consistency[action[i], y[i]] * specialization[action[i], y[i]] / load[action[i]]
        # for i in range(len(acc))]
        rewards = [acc[i] * consistency[action[i], y[i]] * specialization[action[i], y[i]] / load[action[i]] for i in
                   range(len(acc))]
        # rewards = [acc[i] * consistency[action[i], y[i]] * specialization[action[i],y[i]] * load_var for i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards
    def _get_C_matrix(self, preds, routes, true_assignments):
        specialization = np.zeros((self.num_of_experts, self.num_of_classes))
        consistency = np.zeros((self.num_of_experts, self.num_of_classes))
        for i in range(len(preds)):
            specialization[routes[i], true_assignments[i]] += int(true_assignments[i] == preds[i])
            consistency[routes[i], true_assignments[i]] += 1

        specialization = specialization / np.maximum(consistency, 1)  #
        consistency = consistency / np.maximum(consistency.sum(axis=0, keepdims=True), 1)
        return consistency, specialization



    def _entropy(self, sample, action, model, out=None, y=None):
        action_count = torch.bincount(action, minlength=self.num_of_experts)
        action_probs = action_count / action_count.sum()
        entropy = -torch.sum(action_probs * torch.log2(action_probs + 1e-8))
        return torch.zeros_like(action) + min(action_probs)

    def _proposal_specialization_and_consistency(self, sample, action, model, out=None, y=None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        consistency = self._get_consistency(preds, action.detach(), y)
        _, specialization = self._get_C_matrix(preds, action.detach(), y)
        acc = (preds == y).float()
        rewards = [acc[i] * consistency[action[i]] * specialization[action[i], y[i]] for i in range(len(acc))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def _proposal_specialization_and_consistency_linear_assignment(self, sample, action, model, out=None, y=None):
        routes = self._get_routes_from_model(model, sample)
        action = self._linear_assignment(routes).type_as(action).to(action.device)
        return self._proposal_specialization_and_consistency(sample, action, model, out, y)

    def _regret_based_reward(self, sample, action, model, out=None, y=None, k=2):
        # This is appropriate only for actor critic methods
        dist, values = self._get_routes_from_model(model, sample)
        routes = dist.probs
        topk = torch.topk(routes, k, dim=1)
        max_probs, router_preds = torch.max(routes, dim=1)
        out, y = self._get_output_from_model(router_preds, model, sample) if out is None else (out, y)
        out = nn.functional.softmax(out, dim=1)
        preds = torch.argmax(out, dim=1)

        current = max_probs * out[torch.arange(len(out)), y] # best_expert_prob * prob_of_right_class
        out_all = [out]
        for i in range(1, k):
            out_i, _ = self._get_output_from_model(topk.indices[:, i], model, sample)
            out_i = nn.functional.softmax(out_i, dim=1)
            out_all.append(out_i)
        out_all = torch.stack(out_all)
        preds_on_right = out_all[:, torch.arange(len(out)), y]
        best_pred, _ = torch.max(preds_on_right, dim=0)
        regret = best_pred - current
        return -1 * regret
        # action_count = torch.bincount(action, minlength=self.num_of_experts)
        # action_probs = action_count / action_count.sum()
        # consistency, specialization = self._get_C_matrix(preds, action.detach(), y)
        # load = torch.FloatTensor(consistency.sum(axis=1)) / self.num_of_classes
        # entropy = -torch.sum(action_probs * torch.log2(action_probs + 1e-10)) / np.log2(self.num_of_experts)
        # reward = entropy + (current - best_pred)

    def _get_consistency(self, preds, routes, true_assignments):
        consistency = np.zeros((self.num_of_experts, self.num_of_classes))
        for i in range(len(preds)):
            consistency[routes[i], true_assignments[i]] += 1
        prob_consistency = consistency / np.maximum(consistency.sum(axis=0, keepdims=True), 1)
        entropy = -np.sum(prob_consistency * np.log2(prob_consistency + 1e-10), axis=0)
        normalized_entropy = entropy / np.log2(self.num_of_experts)
        return 1 - normalized_entropy

    def _get_C_matrix(self, preds, routes, true_assignments):
        specialization = np.zeros((self.num_of_experts, self.num_of_classes))
        consistency = np.zeros((self.num_of_experts, self.num_of_classes))
        for i in range(len(preds)):
            specialization[routes[i], true_assignments[i]] += int(true_assignments[i] == preds[i])
            consistency[routes[i], true_assignments[i]] += 1

        specialization = specialization / np.maximum(consistency, 1)  #
        consistency = consistency / np.maximum(consistency.sum(axis=0, keepdims=True), 1)
        return consistency, specialization

    # A: Aij = class j goes to expert i
    def _get_A_matrix(self, true_assignments, routes):
        import utils.general_utils as ut
        x = ut.calc_confusion_matrix(true_assignments, routes)
        x = x / x.sum(axis=1, keepdims=True)
        return x

    def _tanh(self, x, alpha=2.5):
        return np.tanh(alpha * x)

    def _new_expertise_reward(self, sample, action, model, out=None, y=None):
        out, y = self._get_output_from_model(action, model, sample) if out is None else (out, y)
        preds = torch.argmax(out, dim=1)
        correct = preds == y
        specialization = NewSpecialization.compute_manual(
            gates=action, labels=y, correct=correct, num_experts=self.num_of_experts, num_classes=self.num_of_classes)
        consistency = Consistency.compute_manual(
            gates=action, labels=y, num_experts=self.num_of_experts, num_classes=self.num_of_classes)
        correct = correct.cpu().numpy()
        rewards = [correct[i] * specialization * consistency for i in range(len(correct))]
        rewards = torch.stack(rewards) if isinstance(rewards[0], torch.Tensor) else torch.FloatTensor(rewards)
        return rewards

    def _spearman_correlation_reward(self, sample, action, model, out=None, y=None, k=3):
        k = min(k, self.num_of_experts)
        batch_size = action.shape[0]
        routes = self._get_routes_from_model(model, sample)
        _, topk_routes = torch.topk(routes, k, dim=1)
        # get output from topk models
        out_all = [self._get_output_from_model(topk_routes[:, 0], model, sample)[0]]
        for i in range(1, k):
            out_i, _ = self._get_output_from_model(topk_routes[:, i], model, sample)
            out_all.append(out_i)
        # loss of the topk models
        loss_all = [self.cf_entropy(out, sample[1].to(out.device)) for out in out_all]
        loss_all = torch.stack(loss_all).T
        # get topk by loss
        _, topk_by_loss = torch.topk(loss_all, k, dim=1, largest=False)
        spearman_corr = spearmanr(topk_routes.cpu().numpy(), topk_by_loss.cpu().numpy(), axis=1)
        return torch.tensor(spearman_corr.correlation[torch.arange(0,batch_size), torch.arange(batch_size, batch_size * 2)])

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
