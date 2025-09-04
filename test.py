import torch
import numpy as np
from functools import partial
from modules.tokenizers import Tokenizer
from modules.dataloaders import MedicalDataLoader
from modules.metrics import compute_scores
from modules.evaluator import Evaluator
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
    test_dataloader = MedicalDataLoader(args, tokenizer, split='test', shuffle=False)
    # build model architecture
    model = Model(args, tokenizer)
    # get function handles of loss and metrics
    criterion = partial(compute_loss, loss_fn=args.loss_fn)
    metrics = compute_scores
    # build trainer and start to train
    evaluator = Evaluator(model, criterion, metrics, args, test_dataloader, load_model=True, indent_level=2)
    evaluator.evaluate()


if __name__ == '__main__':
    main()