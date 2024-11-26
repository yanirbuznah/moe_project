def convert_count_experts_indices_to_json(num_experts_per_tok, n_routed_experts, count_experts_indices):
    data = []
    for i in range(num_experts_per_tok):
        row = {"top expert": f"top_{i}"}
        for j in range(n_routed_experts):
            row[f"expert_{j}"] = self.count_experts_indices[i, j].item()
        data.append(row)
    return data