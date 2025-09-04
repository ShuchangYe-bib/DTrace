import os
import cv2
import torch
import random
from abc import abstractmethod
from modules.utils import generate_heatmap
from modules.metrics import LabelMetrics

class BaseEvaluator(object):
    def __init__(self, model, criterion, metric_ftns, args, load_model=False, to_print=True):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.label_metrics = LabelMetrics()
        self.to_print = to_print

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

        if load_model:
            self._load_checkpoint(args.model_path)

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        if self.args.device == "cuda":
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
        else:
            device = self.args.device 
            list_ids = list(range(0))
        return device, list_ids

    def _load_checkpoint(self, checkpoint_path):
        if self.to_print:
            print("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['state_dict'])

class Evaluator(BaseEvaluator):
    def __init__(self, model, criterion, metric_ftns, args, dataloader, indent_level=2, split="test", load_model=False, to_print=True):
        super(Evaluator, self).__init__(model, criterion, metric_ftns, args, load_model=load_model, to_print=to_print)
        self.dataloader = dataloader
        self.indent_level = indent_level
        self.split = split

    def print_log(log):
        for key, value in log.items():
            print('\t{:15s}: {}'.format(str(key), value))

    def evaluate(self):
        if self.to_print:
            print(f'Evaluating {self.split} set:', end='')
        log = dict()
        if self.args.n_gpu > 1 and self.args.device == "cuda":
            model = self.model.module.to(self.args.device)
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            evaluate_gts, evaluate_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, tf_idfs, labels) in enumerate(self.dataloader):
                print("\r"+"\t\t"*self.indent_level+f"{self.split.capitalize()} {batch_idx+1}/{len(self.dataloader)}", end='')
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = model(images, mode='sample')
                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                evaluate_res.extend(reports)
                evaluate_gts.extend(ground_truths)

                hidden_number = 0
                _, _, _, _, _, _, origin_images_cls_prob, _ = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                hidden_number = 0.75
                _, _, images_pred, reports_pred, images_mask, reports_mask, _, _ = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                _, _, _, _, generated_images_cls_prob, _ = self.model.train_traceback(images, reports_ids, images_pred, reports_pred, images_mask, reports_mask)
                self.label_metrics.record(labels, origin_images_cls_prob, generated_images_cls_prob)

            evaluate_met = self.metric_ftns({i: [gt] for i, gt in enumerate(evaluate_gts)},
                                        {i: [re] for i, re in enumerate(evaluate_res)})
            evaluate_cls_met = self.label_metrics.compute_score()
        # log = {f'{self.split}_' + k: v for k, v in evaluate_met.items()}
        log = evaluate_met
        log.update(evaluate_cls_met)
        if self.to_print:
            print()
            Evaluator.print_log(log)
        return log

    def plot(self, n_sample=-1):
        sample_idxs = range(len(self.dataloader))
        if n_sample != -1:
            sample_idxs = random.sample(sample_idxs, n_sample)
            sample_idxs.sort()
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        print(f'Plotting {self.split} set:', end='')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        if self.args.n_gpu > 1 and self.args.device == "cuda":
            model = self.model.module.to(self.args.device)
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            i = 0
            sample_idx = sample_idxs[i]
            for batch_idx, (images_id, images, reports_ids, reports_masks, tf_idfs, labels) in enumerate(self.dataloader):
                print("\r"+"\t\t"*self.indent_level+f"{self.split.capitalize()} {i+1}/{n_sample}", end='')
                if batch_idx != sample_idx:
                    continue
                else:
                    i = min(i + 1, n_sample - 1)
                    sample_idx = sample_idxs[i]

                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn, n_channels=self.args.image_shape[0])
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.jpg".format(word_idx, word)), heatmap)
        print()