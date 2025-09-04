import os
import nltk
import torch
import textwrap
import numpy as np

from PIL import Image
from models.model import Model
from torchvision import transforms
from modules.tokenizers import Tokenizer
from modules.utils import print_report, post_process_report

from config import args


def load_checkpoint(model, model_path):
    print("Loading checkpoint: {} ...".format(model_path))
    checkpoint = torch.load(model_path, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['state_dict'])

def generate_report(model, image_path, n_channels=3):
    if n_channels == 1:
        mean = (0.5)
        std = (0.5)
    elif n_channels == 3:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise ValueError
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
    ])
    
    image = Image.open(image_path)
    if n_channels == 1:
        image = image.convert('L')
    elif n_channels == 3:
        image = image.convert('RGB')
    else:
        raise ValueError
    image = transform(image).to(args.device)
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0), mode='sample').cpu().numpy()
    report = model.tokenizer.decode_batch(output)[0]
    report = post_process_report(report)

    return report

def main():
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    # create tokenizer
    tokenizer = Tokenizer(args.ann_path, args.threshold, args.dataset_name)
    # build model architecture
    model = Model(args, tokenizer)
    model = model.to(args.device)
    load_checkpoint(model, args.model_path)
    
    generated_report = generate_report(model, args.image_path, args.image_shape[0])
    print_report(generated_report, name="generated")

if __name__ == '__main__':
    main()


