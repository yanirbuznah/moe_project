from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def action_assignment_strategy(strategy: str, num_experts: int):
    if strategy is None or strategy == 'none':
        return lambda x: x.argmax(axis=1)
    if strategy == 'LinearAssignment':
        return LinearAssignment()
    elif strategy == 'LinearAssignmentWithCapacity':
        return LinearAssignmentWithCapacity()
    elif strategy == 'LinearAssignmentByDiff':
        return LinearAssignmentByDiff()
    elif strategy == 'BaseLinearAssignment':
        return BaseLinearAssignment()
    elif strategy == 'BiasAssignment':
        return BiasAssignment(num_experts)
    else:
        raise ValueError(f'Unknown strategy {strategy}')



class LinearAssignment:

    @classmethod
    def __call__(cls, routes: torch.Tensor) -> torch.Tensor:
        costs = -1 * routes.T
        assignments = cls._balanced_assignment(costs)
        new_assignment = torch.zeros(len(routes))
        for expert, assignment in enumerate(assignments):
            new_assignment[assignment] = expert

        return new_assignment.type(torch.long)

    @staticmethod
    def _balanced_assignment(cost_matrix: torch.Tensor) -> List[List[int]]:
        cost_matrix = cost_matrix.cpu().numpy()
        experts_assignments = [[] for _ in range(cost_matrix.shape[0])]
        unassigned = cost_matrix.shape[1]
        while unassigned > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            cost_matrix[row_ind, col_ind] = np.inf
            for i, j in zip(row_ind, col_ind):
                experts_assignments[i].append(j)
            unassigned -= len(row_ind)

            # cost_matrix = np.delete(cost_matrix, col_ind, axis=1)
        return experts_assignments


class LinearAssignmentWithCapacity:

    def __init__(self, capacity: float = 1.2):
        self.capacity = capacity

    def __call__(self, routes: torch.Tensor) -> torch.Tensor:
        max_capacity = round(self.capacity * len(routes) / routes.shape[1])
        assignments = self._get_expert_assignments(routes, max_capacity)
        assert all([len(assignment) <= max_capacity for assignment in assignments])
        new_assignment = torch.zeros(len(routes))
        for expert, assignment in enumerate(assignments):
            new_assignment[assignment] = expert
        return new_assignment.type(torch.long).to(routes.device)

    def _get_expert_assignments(self, routes: torch.Tensor, max_capacity: int) -> list:
        unassigned = set([i for i in range(len(routes))])
        experts_assignments = [[] for _ in range(routes.shape[1])]
        while len(unassigned) > 0:
            routes_max, experts = torch.max(routes, dim=-1)
            for i, (expert, route) in enumerate(zip(experts, routes_max)):
                if i in unassigned:
                    experts_assignments[expert].append([i, route.item()])
            experts_assignments = [sorted(expert, key=lambda x: x[1], reverse=True) for expert in experts_assignments]
            unassigned = [expert[max_capacity:] for expert in experts_assignments]
            unassigned = [i[0] for expert in unassigned for i in expert]
            routes[unassigned, experts[unassigned]] = -np.inf
            unassigned = set(unassigned)
            experts_assignments = [expert[:max_capacity] for expert in experts_assignments]
        experts_assignments = [[i[0] for i in expert] for expert in experts_assignments]
        return experts_assignments


class LinearAssignmentByDiff(LinearAssignmentWithCapacity):

    def __init__(self, capacity: float = 1.2):
        super().__init__(capacity)

    def __call__(self, routes):
        max_capacity = round(self.capacity * len(routes) / routes.shape[1])
        assignments = self._get_expert_assignments(routes, max_capacity)
        return assignments.type(torch.long).to(routes.device)

    def _get_expert_assignments(self, routes: torch.Tensor, max_capacity: int) -> torch.Tensor:
        max_routes, initial_assignment = torch.max(routes, dim=-1)
        # count each expert's assignment
        counts = torch.zeros(routes.shape[1])
        for assignment in initial_assignment:
            counts[assignment] += 1
        assignment_to_remove = (counts - max_capacity)
        diff_matrix = max_routes.view(-1, 1) - routes
        diff_matrix[diff_matrix == 0] = float('inf')
        while torch.any(assignment_to_remove > 0):
            min_routes, assignment = torch.min(diff_matrix, dim=-1)
            _, indices = torch.sort(min_routes)
            sorted_assignment = assignment[indices]
            for i, expert in enumerate(sorted_assignment):
                prev_assignment = initial_assignment[indices[i]]
                if assignment_to_remove[prev_assignment] > 0:
                    assignment_to_remove[prev_assignment] -= 1
                    initial_assignment[indices[i]] = expert
                    assignment_to_remove[expert] += 1
                    diff_matrix[indices[i], expert] = float('inf')
        return initial_assignment
class BaseLinearAssignment:
    def __init__(self, capacity: float = 1.2):
        self.capacity = capacity

    def __call__(self, routes):
        if routes.shape[0] > routes.shape[1]:
            routes = routes.T
        max_capacity = round(self.capacity * routes.shape[1] / routes.shape[0])
        assignments =  self._get_experts_assignment(routes, max_capacity)
        return assignments

    def _get_experts_assignmentT(self, scores: torch.Tensor, max_capacity: int, max_iterations: int = 100):
        # score is a matrix of shape (num_workers, num_jobs)
        eps = 1e-6
        num_jobs, num_workers = scores.size()
        jobs_per_worker = max_capacity
        value = scores.clone()

        iterations = 0
        cost = scores.new_zeros(1, num_jobs).float()

        jobs_with_bids = torch.zeros(num_workers).bool()

        while not jobs_with_bids.all():
            top_values, top_index = torch.topk(value, k=jobs_per_worker + 1, dim=0)

            # Each worker bids the difference in value between a job and the k+1th job
            bid_increments = top_values[:-1, :] - top_values[-1:, :] + eps
            bids = torch.scatter(torch.zeros_like(scores), dim=1, index=top_index[:-1, :], src=bid_increments)

            if 0 < iterations < max_iterations:
                # If a worker won a job on the previous round, put in a minimal bid to retain
                # the job only if no other workers bid this round.
                bids[top_bidders, jobs_with_bids] = eps

            # Find the highest bidding worker per job
            top_bids, top_bidders = bids.max(dim=1)
            jobs_with_bids = top_bids > 0
            top_bidders = top_bidders[jobs_with_bids]

            # Make popular items more expensive
            cost += top_bids
            value = scores - cost

            if iterations < max_iterations:
                # If a worker won a job, make sure it appears in its top-k on the next round
                value[jobs_with_bids,top_bidders] = float('inf')
            else:
                value[jobs_with_bids, top_bidders] = scores[jobs_with_bids, top_bidders]
            iterations += 1

            if iterations >= max_iterations:
                break

        return top_bidders
    def _get_experts_assignment(self, scores:torch.Tensor, max_capacity:int, max_iterations:int =100):
        # score is a matrix of shape (num_workers, num_jobs)
        eps = 1e-6
        num_workers, num_jobs = scores.size()
        jobs_per_worker = max_capacity
        value = scores.clone()

        iterations = 0
        cost = scores.new_zeros(1, num_jobs).float()

        jobs_with_bids = torch.zeros(num_workers).bool()

        while not jobs_with_bids.all():
            top_values, top_index = torch.topk(value, k=jobs_per_worker + 1, dim=1)

            # Each worker bids the difference in value between a job and the k+1th job
            bid_increments = top_values[:, :-1] - top_values[:, -1:] + eps
            bids = torch.scatter(torch.zeros_like(scores), dim=1, index=top_index[:, :-1], src=bid_increments)

            if 0 < iterations < max_iterations:
                # If a worker won a job on the previous round, put in a minimal bid to retain
                # the job only if no other workers bid this round.
                bids[top_bidders, jobs_with_bids] = eps

            # Find the highest bidding worker per job
            top_bids, top_bidders = bids.max(dim=0)
            jobs_with_bids = top_bids > 0
            top_bidders = top_bidders[jobs_with_bids]

            # Make popular items more expensive
            cost += top_bids
            value = scores - cost

            if iterations < max_iterations:
                # If a worker won a job, make sure it appears in its top-k on the next round
                value[top_bidders, jobs_with_bids] = float('inf')
            else:
                value[top_bidders, jobs_with_bids] = scores[top_bidders, jobs_with_bids]
            iterations += 1

            if iterations >= max_iterations:
                break

        return top_bidders

class BiasAssignment:

    def __init__(self, num_experts: int, u: float = 0.001):
        self.num_experts = num_experts
        self.u = u
        self.bias = torch.zeros(num_experts)

    def __call__(self, routes: torch.Tensor):
        scores = torch.softmax(routes, dim=-1).cpu() + self.bias
        assignments = scores.argmax(dim=-1)

        counts = torch.bincount(assignments, minlength=self.num_experts).to(torch.float32)
        rewards = counts.mean() - counts
        self.update(rewards)
        return assignments

    def update(self, rewards: torch.Tensor):
        rewards_sign = torch.sign(rewards)
        self.bias += self.u * rewards_sign


if __name__ == '__main__':
    scores = torch.FloatTensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3]])
    print(balanced_assignment(scores))
    print(LinearAssignment()(scores))
    print(LinearAssignmentWithCapacity)