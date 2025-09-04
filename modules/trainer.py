import os
import csv
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
from modules.utils import freeze_model, unfreeze_model
from modules.metrics import LabelMetrics


class BaseTrainer(object):
    
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.label_metrics = LabelMetrics()
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {self.mnt_metric: self.mnt_best}

    @abstractmethod
    def _run_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0

        csv_file = open("training.csv", "w")
        writer = csv.writer(csv_file)
        writer.writerow(["bleu-1", "bleu-2", "bleu-3", "bleu-4", "meteor", "rouge-l", "cider"])

        for epoch in range(self.start_epoch, self.epochs+1):

            print("Epoch {}/{}".format(epoch, self.epochs), end='')
            result = self._run_epoch(epoch)

            # record val to csv
            try:
                writer.writerow([result["val_BLEU_1"], result["val_BLEU_2"], result["val_BLEU_3"], result["val_BLEU_4"], result["val_METEOR"], result["val_ROUGE_L"], result["val_CIDER"]])
            except:
                pass

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            # save the model if the performance increased (model_best) or otherwise periodically (current_checkpoint)
            if best:
                self._save_checkpoint(epoch, save_best=best)
            elif epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

            print()
        csv_file.close()

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

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'loss': self.loss,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        try:
            improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder[self.mnt_metric]) or \
                           (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder[self.mnt_metric])
        except:
            return

        if improved:
            self.best_recorder.update(log)


class Trainer(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _run_epoch(self, epoch):
        if epoch <= 1:
            self.mnt_mode = "off"
            train_log = self._train_epoch(epoch, "image_generation")
            log = {**train_log}
        elif epoch <= 2:
            self.mnt_mode = self.args.monitor_mode
            train_log = self._train_epoch(epoch)
            val_log = self._val_epoch(epoch)
            log = {**train_log, **val_log}
        else:
            self.mnt_mode = self.args.monitor_mode
            train_log = self._train_epoch(epoch, "report_generation")
            val_log = self._val_epoch(epoch)
            log = {**train_log, **val_log}
        # test_log = self._test_epoch(epoch)
        
        # log = {**train_log, **val_log}
        print()
        return log

    def _train_epoch(self, epoch, strategy="hybrid"):
        train_loss = 0

        prev_ig_loss = 0
        prev_rg_loss = 0

        status = "forward"

        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, tf_idfs, labels) in enumerate(self.train_dataloader):
            print("\r"+"\t\t"+"Batch {}/{}".format(batch_idx+1, len(self.train_dataloader)), end='')

            images, reports_ids, reports_masks, tf_idfs = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device), tf_idfs.to(self.device)
            hidden_number = torch.rand(1).item()
            if strategy == "forward":
                status = "forward"
            elif strategy == "traceback":
                status = "traceback"
            elif strategy == "report_generation":
                status = "report_generation"
            elif strategy == "image_generation":
                status = "image_generation"

            # forward
            if status == "forward":
                stage_forward = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                images_true, reports_true, images_pred, reports_pred, images_mask, reports_mask, img_cls_prob, txt_cls_prob = stage_forward
                image_reconstruction_loss, image_classification_loss, report_generation_loss, word_importance_loss, report_classification_loss = self.criterion((images_true, reports_true[:, 1:], images_pred, reports_pred, images_mask, reports_mask[:, 1:], hidden_number, 1-hidden_number, labels, img_cls_prob, txt_cls_prob, tf_idfs[:, 1:]), "forward")
                loss = image_reconstruction_loss + image_classification_loss + report_generation_loss + word_importance_loss + report_classification_loss
                prev_ig_loss, prev_rg_loss = image_reconstruction_loss.item(), report_generation_loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                status = "traceback"
            # traceback
            elif status == "traceback":
                freeze_model(self.model.get_encoder())
                stage_forward = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                images_true, reports_true, images_pred, reports_pred, images_mask, reports_mask, origin_img_cls_prob, origin_txt_cls_prob = stage_forward
                stage_traceback = self.model.train_traceback(images, reports_ids, images_pred, reports_pred, images_mask, reports_mask)
                images_feature_loss, reports_feature_loss = self.criterion((*stage_traceback, torch.tensor(hidden_number)*torch.exp(torch.tensor(-prev_ig_loss)), torch.tensor(1-hidden_number)**torch.exp(torch.tensor(-prev_rg_loss)), labels), "traceback")
                loss = images_feature_loss + reports_feature_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                unfreeze_model(self.model.encoder_decoder.model.mae.get_encoder())
                status = "forward"
            # report generation
            elif status == "report_generation":
                hidden_number = 0.15 * hidden_number
                stage_forward = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                images_true, reports_true, images_pred, reports_pred, images_mask, reports_mask, img_cls_prob, txt_cls_prob = stage_forward
                _, _, report_generation_loss, word_importance_loss, report_classification_loss = self.criterion((images_true, reports_true[:, 1:], images_pred, reports_pred, images_mask, reports_mask[:, 1:], hidden_number, 1-hidden_number, labels, img_cls_prob, txt_cls_prob, tf_idfs[:, 1:]), "forward")
                loss = report_generation_loss + word_importance_loss + report_classification_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                status = "forward"
            elif status == "image_generation":
                hidden_number = 0.1 * hidden_number + 0.75
                stage_forward = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                images_true, reports_true, images_pred, reports_pred, images_mask, reports_mask, img_cls_prob, txt_cls_prob = stage_forward
                image_reconstruction_loss, _, _, _, _ = self.criterion((images_true, reports_true[:, 1:], images_pred, reports_pred, images_mask, reports_mask[:, 1:], hidden_number, 1-hidden_number, labels, img_cls_prob, txt_cls_prob, tf_idfs[:, 1:]), "forward")
                loss = image_reconstruction_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                status = "forward"

            train_loss += loss.item()
        
        self.loss = train_loss / len(self.train_dataloader)
        log = {'train_loss': self.loss}
        return log

    def _val_epoch(self, epoch):
        if self.args.n_gpu > 1 and self.args.device == "cuda":
            model = self.model.module.to(self.args.device)
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, tf_idfs, labels) in enumerate(self.val_dataloader):
                print("\r"+"\t\t"*2+"Validation {}/{}".format(batch_idx+1, len(self.val_dataloader)), end='')
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = model(images, mode='sample')
                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

                hidden_number = 0
                _, _, _, _, _, _, origin_images_cls_prob, _ = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                hidden_number = 0.75
                _, _, images_pred, reports_pred, images_mask, reports_mask, _, _ = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                _, _, _, _, generated_images_cls_prob, _ = self.model.train_traceback(images, reports_ids, images_pred, reports_pred, images_mask, reports_mask)
                self.label_metrics.record(labels, origin_images_cls_prob, generated_images_cls_prob)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            val_cls_met = self.label_metrics.compute_score()
        log = {'val_' + k: v for k, v in val_met.items()}
        log.update({'val_' + k: v for k, v in val_cls_met.items()})
        log.update({"val_score": (log["val_BLEU_1"]+log["val_BLEU_2"]+log["val_BLEU_3"]+log["val_BLEU_4"])/4 + log["val_METEOR"] + log["val_ROUGE_L"] + log["val_CIDER"]})
        return log

    def _test_epoch(self, epoch):
        if self.args.n_gpu > 1 and self.args.device == "cuda":
            model = self.model.module.to(self.args.device)
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, tf_idfs, labels) in enumerate(self.test_dataloader):
                print("\r"+"\t\t"*3+"\tTest {}/{}".format(batch_idx+1, len(self.test_dataloader)), end='')
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = model(images, mode='sample')
                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

                hidden_number = 0
                _, _, _, _, _, _, origin_images_cls_prob, _ = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                hidden_number = 0.75
                _, _, images_pred, reports_pred, images_mask, reports_mask, _, _ = self.model.train_forward(images, reports_ids, reports_masks, hidden_number)
                _, _, _, _, generated_images_cls_prob, _ = self.model.train_traceback(images, reports_ids, images_pred, reports_pred, images_mask, reports_mask)
                self.label_metrics.record(labels, origin_images_cls_prob, generated_images_cls_prob)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_cls_met = self.label_metrics.compute_score()

        log = {'test_' + k: v for k, v in test_met.items()}
        log.update({'test_' + k: v for k, v in test_cls_met.items()})
        return log
        
