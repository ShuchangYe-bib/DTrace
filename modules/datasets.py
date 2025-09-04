import os
import json
import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, args, tokenizer, split, transform=None):
        self.args = args
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            tokenized = self.examples[i]['ids'][1:-2] # cut off leading and trailing 0
            counter = Counter(tokenized)
            report_len = len(tokenized)
            self.examples[i]['tf-idfs'] = [counter[token_idx] / report_len * tokenizer.idx2idf[token_idx] for token_idx in tokenized]
            self.examples[i]['tf-idfs'] = [0] + self.examples[i]['tf-idfs'] + [0]

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']

        image_path = example['image_path']
        images = []
        for path in image_path:
            image = Image.open(os.path.join(self.image_dir, path))
            if self.args.image_shape[0] == 1:
                image = image.convert('L')
            elif self.args.image_shape[0] == 3:
                image = image.convert('RGB')
            else:
                raise ValueError
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        image = torch.stack(images, 0)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        tf_idfs = example['tf-idfs']
        labels = example['label_vec']
        sample = (image_id, image, report_ids, report_masks, seq_length, tf_idfs, labels)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0]))
        if self.args.image_shape[0] == 1:
            image = image.convert('L')
        elif self.args.image_shape[0] == 3:
            image = image.convert('RGB')
        else:
            raise ValueError
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        tf_idfs = example['tf-idfs']
        labels = example['label_vec']
        sample = (image_id, image, report_ids, report_masks, seq_length, tf_idfs, labels)
        return sample


