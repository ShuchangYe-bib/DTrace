import os
import torch
import pandas as pd
from tabulate import tabulate
from modules.evaluator import Evaluator

class Recorder(object):
	def __init__(self, model, criterion, metric_ftns, args, train_dataloader, val_dataloader,
				test_dataloader, load_model=True, best_recorder=None):
		self.args = args

		# setup GPU device if available, move model into configured device
		self.device, device_ids = self._prepare_device(args.n_gpu)
		self.model = model.to(self.device)
		if len(device_ids) > 1:
			self.model = torch.nn.DataParallel(model, device_ids=device_ids)

		self.criterion = criterion
		self.metric_ftns = metric_ftns

		self.epochs = self.args.epochs
		self.save_dir = self.args.save_dir

		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader
		self.test_dataloader = test_dataloader

		self.best_recorder = best_recorder
		self.best_epoch = None
		self.train_loss = None
		self.val_log = None
		if best_recorder is not None:
			self.best_epoch = best_recorder.pop("epoch")
			self.train_loss = best_recorder.pop("train_loss")
			self.val_log = best_recorder

		self.checkpoint_path = os.path.join(args.save_dir, 'model_best.pth')
		if load_model:
			self._load_checkpoint()

		self.record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
		if not os.path.exists(self.args.record_dir):
			os.makedirs(self.args.record_dir)

	def _load_checkpoint(self):
		print("Loading checkpoint: {} ...".format(self.checkpoint_path))
		checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
		self.model.load_state_dict(checkpoint['state_dict'])
		self.best_epoch = checkpoint["epoch"]
		self.train_loss = checkpoint["loss"]
	
	def print_log(log):
		for key, value in log.items():
			print('\t{:15s}: {}'.format(str(key), value))
	
	def _prepare_device(self, n_gpu_use):
		n_gpu = torch.cuda.device_count()
		if n_gpu_use > 0 and n_gpu == 0:
			print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
			n_gpu_use = 0
		if n_gpu_use > n_gpu:
			print(
				"Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
					n_gpu_use, n_gpu))
			n_gpu_use = n_gpu
		device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
		list_ids = list(range(n_gpu_use))
		return device, list_ids

	def record(self):
		print("Evaluating best model based on validation set (w.r.t {}):".format(self.args.monitor_metric))

		# Evaluate the model on train, validation and test dataset and record the results
		indent_level = 0
		train_evaluator = Evaluator(
			self.model, self.criterion, self.metric_ftns, self.args,
			self.train_dataloader, split="train", indent_level=indent_level, to_print=False
		)
		train_log = train_evaluator.evaluate()
		indent_level += 1

		if self.val_log is None:
			val_evaluator = Evaluator(
				self.model, self.criterion, self.metric_ftns, self.args,
				self.val_dataloader, split="val", indent_level=indent_level, to_print=False
			)
			val_log = val_evaluator.evaluate()
			indent_level += 1
			
		else:
			val_log = {k.replace("val_", ""): v for k, v in self.val_log.items()}

		test_evaluator = Evaluator(
			self.model, self.criterion, self.metric_ftns, self.args,
			self.test_dataloader, split="test", indent_level=indent_level, to_print=False
		)
		test_log = test_evaluator.evaluate()
		print()
		
		additional_log = {"epoch": self.best_epoch, "loss": self.train_loss}
		additional_log = pd.DataFrame(additional_log, index=["train"])
		result = pd.DataFrame(data=[train_log, val_log, test_log], index=["train", "val", "test"])
		result = pd.concat([additional_log, result], axis=1).round(decimals=3).fillna("--")
		print(tabulate(result, headers='keys', tablefmt='psql'))
		result.to_csv(self.record_path)


		