import torch

def accuracy(y_pred, y_true):
	num_samples = y_pred.size(0)
	correct = (y_pred == y_true).sum().item()
	accuracy_score = (correct / num_samples)
	return accuracy_score

def recall(y_pred, y_true):
    true_positives = ((y_pred == 1) & (y_true == 1)).sum().item()
    false_negatives = ((y_pred == 0) & (y_true == 1)).sum().item()
    recall_score = true_positives / (true_positives + false_negatives)
    return recall_score

def precision(y_pred, y_true):
    true_positives = ((y_pred == 1) & (y_true == 1)).sum().item()
    false_positives = ((y_pred == 1) & (y_true == 0)).sum().item()
    precision_score = true_positives / (true_positives + false_positives)
    return precision_score

class LabelMetrics(object):
	def __init__(self):
		self.true_labels = torch.tensor([])
		self.pred_labels_origin = torch.tensor([])
		self.pred_labels_generated = torch.tensor([])

	def record(self, true_labels, pred_labels_origin, pred_labels_generated):
		mask = true_labels != 0 # masked out labels which is mentioned with ambiguity / not mentioned
		true_labels = (true_labels[mask] + 1) / 2
		pred_labels_origin = torch.round(pred_labels_origin[mask]).cpu()
		pred_labels_generated = torch.round(pred_labels_generated[mask]).cpu()
		self.true_labels = torch.cat((self.true_labels, true_labels))
		self.pred_labels_origin = torch.cat((self.pred_labels_origin, pred_labels_origin))
		self.pred_labels_generated = torch.cat((self.pred_labels_generated, pred_labels_generated))

	def compute_score(self, clear=True):
		accuracy_origin, accuracy_generated = self.compute_accuracy()
		recall_origin, recall_generated = self.compute_recall()
		precision_origin, precision_generated = self.compute_precision()

		result = {
			"acc_true": accuracy_origin,
			"acc_pred": accuracy_generated,
			"rec_true": recall_origin,
			"rec_pred": recall_generated,
			"pre_true": precision_origin,
			"pre_pred": precision_generated
		}

		if clear:
			self.clear()

		return result

	def clear(self):
		self.true_labels = torch.tensor([])
		self.pred_labels_origin = torch.tensor([])
		self.pred_labels_generated = torch.tensor([])

	def compute_accuracy(self):
		accuracy_origin = accuracy(self.true_labels, self.pred_labels_origin)
		accuracy_generated = accuracy(self.true_labels, self.pred_labels_generated)
		return accuracy_origin, accuracy_generated

	def compute_recall(self):
		recall_origin = recall(self.true_labels, self.pred_labels_origin)
		recall_generated = recall(self.true_labels, self.pred_labels_generated)
		return recall_origin, recall_generated

	def compute_precision(self):
		precision_origin = precision(self.true_labels, self.pred_labels_origin)
		precision_generated = precision(self.true_labels, self.pred_labels_generated)
		return precision_origin, precision_generated
    
