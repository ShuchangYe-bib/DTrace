import torch
import numpy as np
from functools import partial
import time
from modules.tokenizers import Tokenizer
from modules.dataloaders import MedicalDataLoader
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.recorder import Recorder
from modules.trainer import Trainer
from modules.metrics import compute_scores
from modules.loss import compute_loss
from models.model import Model
from config import args


def main():
	
	# fix random seeds
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(args.seed)
    
	# create tokenizer
	tokenizer = Tokenizer(args.ann_path, args.threshold, args.dataset_name)
    
	# create data loader
	train_dataloader = MedicalDataLoader(args, tokenizer, split='train', shuffle=True)
	val_dataloader = MedicalDataLoader(args, tokenizer, split='val', shuffle=False)
	test_dataloader = MedicalDataLoader(args, tokenizer, split='test', shuffle=False)
    
	# build model architecture
	model = Model(args, tokenizer)
    
	# get function handles of loss and metrics
	criterion = partial(compute_loss, loss_fn=args.loss_fn)
	metrics = compute_scores
    
	# build optimizer, learning rate scheduler
	optimizer = build_optimizer(args, model)
	lr_scheduler = build_lr_scheduler(args, optimizer)
    
	# build trainer and start to train
	trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
	trainer.train()

if __name__ == "__main__":
	main()






