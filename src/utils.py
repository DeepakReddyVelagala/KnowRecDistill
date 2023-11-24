import torch


def calculate_precision_recall(top_indices, labels, top_n):
    relevant_items = set(labels.nonzero().view(-1).cpu().numpy())
    recommended_items = set(top_indices.cpu().numpy())
    intersection = relevant_items & recommended_items
    precision = len(intersection) / len(recommended_items)
    recall = len(intersection) / len(relevant_items) if len(relevant_items) > 0 else 0
    return precision, recall

# Credit: Topology Distillation for Recommender Systems	Authors
def sim(A, B, is_inner=False):
	if not is_inner:
		denom_A = 1 / (A ** 2).sum(1, keepdim=True).sqrt()
		denom_B = 1 / (B.T ** 2).sum(0, keepdim=True).sqrt()

		sim_mat = torch.mm(A, B.T) * denom_A * denom_B
	else:
		sim_mat = torch.mm(A, B.T)

	return sim_mat