import re
import json
import numpy as np
from functools import partial
from collections import Counter


class Tokenizer(object):

	# anno_path: the path to the json annotation file
	# threshold: the minimum frequency of words
	# dataset_name: dataset to be used choosing from IU Chest X-rays, MIMIC Chest X-ray (MIMIC-CXR).
	def __init__(self, anno_path, threshold, dataset_name):
		assert dataset_name in ("iu_xray", "mimic_cxr")
		self.threshold = threshold
		self.dataset_name = dataset_name
		self.annotation = json.loads(open(anno_path, "r").read())
		self.report_preprocessing = partial(Tokenizer.preprocess_report, dataset_name)
		self.token2idx, self.idx2token, self.idx2idf = self.create_vocabulary()

	def create_vocabulary(self):
		tokens = []
		doc_counts = Counter()
		num_docs = len(self.annotation['train'])
		for record in self.annotation['train']:
			cleaned_report = self.report_preprocessing(record["report"])
			tokenized_report = cleaned_report.split()
			tokens.extend(tokenized_report)
			doc_counts += Counter(set(tokenized_report))
		counter = Counter(tokens)
		vocab = sorted([k for k, v in counter.items() if v >= self.threshold] + ["<unk>"])
		token2idx, idx2token, idx2idf = {}, {}, {}
		for idx, token in enumerate(vocab):
			token2idx[token] = idx+1
			idx2token[idx+1] = token
			if token == "<unk>":
				idx2idf[idx+1] = 0
			else:
				idx2idf[idx+1] = np.log(num_docs / doc_counts[token])
		return token2idx, idx2token, idx2idf

	def get_token_by_id(self, id):
		return self.idx2token[id]

	def get_id_by_token(self, token):
		if token not in self.token2idx:
			return self.token2idx["<unk>"]
		return self.token2idx[token]

	def get_vocab_size(self):
		return len(self.token2idx)

	def __call__(self, report):
		tokens = self.report_preprocessing(report).split()
		ids = []
		for token in tokens:
			ids.append(self.get_id_by_token(token))
		ids = [0] + ids + [0]
		return ids

	def decode(self, ids):
		txt = ""
		for i, idx in enumerate(ids):
			if idx > 0:
				if i >= 1:
					txt += " "
				txt += self.idx2token[idx]
			else:
				break
		return txt

	def decode_batch(self, ids_batch):
		out = []
		for ids in ids_batch:
			out.append(self.decode(ids))
		return out

	def preprocess_report(dataset_name, report):
		assert dataset_name in ("iu_xray", "mimic_cxr")
		consistency = None
		if dataset_name == "iu_xray":
			consistency = lambda x: x.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
				.replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
				.replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
				.strip().lower().split('. ')
		elif dataset_name == "mimic_cxr":
			consistency = lambda x: x.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
				.replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
				.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
				.replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
				.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
				.replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
				.replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
				.strip().lower().split('. ')
		tokenizer = lambda x: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', x.replace('"', '').replace('/', '') \
				.replace('\\', '').replace("'", '').strip().lower())
		tokens = [tokenizer(sentence) for sentence in consistency(report) if len(tokenizer(sentence)) > 0]
		report = ' . '.join(tokens) + ' .'
		return report











