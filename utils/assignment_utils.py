from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


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
