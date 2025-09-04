import torch
import torch.nn as nn
import numpy as np

from torchvision.utils import save_image
from models.layers.visual_extractor import VisualExtractor
from models.layers.encoder_decoder import EncoderDecoder
from modules.utils import patchify, unpatchify


class Model(nn.Module):

	def __init__(self, args, tokenizer):
		super(Model, self).__init__()
		self.args = args
		self.tokenizer = tokenizer
		self.visual_extractor = VisualExtractor(args)
		self.encoder_decoder = EncoderDecoder(args, tokenizer)

	def __str__(self):
		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		params = sum([np.prod(p.size()) for p in model_parameters])
		return super().__str__() + '\nTrainable parameters: {}'.format(params)

	def get_encoder(self):
		encoders = []
		encoders.append(self.visual_extractor)
		encoders.extend(self.encoder_decoder.get_encoder())
		return encoders

	def extract_features(self, images):
		if images.dim() == 5: # batch_size, n_images, num_channels, height, width
			att_feats_arr, fc_feats_arr = [], []
			for i in range(images.shape[1]):
				image = images[:, i]
				att_feats, fc_feats = self.visual_extractor(image)
				att_feats_arr.append(att_feats)
				fc_feats_arr.append(fc_feats)
			att_feats = torch.cat(att_feats_arr, dim=1)
			fc_feats = torch.cat(fc_feats_arr, dim=1)
		elif images.dim() == 4: # batch_size, num_channels, height, width
			att_feats, fc_feats = self.visual_extractor(images)
		return att_feats, fc_feats

	def train_forward(self, images, targets, targets_mask, hidden_number):
		att_feats, fc_feats = self.extract_features(images)
		src_decoded, tgt_decoded, img_mask, report_masks, img_cls_prob, txt_cls_prob = self.encoder_decoder(fc_feats, att_feats, targets, targets_mask, hidden_number, mode='forward')
		images_true = patchify(images, self.args.patch_size, self.args.image_shape)
		reports_true = targets
		images_pred = src_decoded
		reports_pred = tgt_decoded
		images_mask = img_mask
		reports_mask = report_masks
		stage_forward = (images_true, reports_true, images_pred, reports_pred, images_mask, reports_mask, img_cls_prob, txt_cls_prob)
		return stage_forward

	def train_traceback(self, origin_images, origin_reports, generated_images_patches, generated_reports_prob, images_mask, reports_mask):
		origin_images = origin_images
		generated_images_patches = generated_images_patches * images_mask.unsqueeze(2) + patchify(origin_images, self.args.patch_size, self.args.image_shape) * (1-images_mask).unsqueeze(2)
		generated_images = unpatchify(generated_images_patches, self.args.patch_size, self.args.image_shape)
		origin_att_feats, origin_fc_feats = self.extract_features(origin_images)
		generated_att_feats, generated_fc_feats = self.extract_features(generated_images)
		origin_images_feature = self.encoder_decoder.forward_visual_encoder(origin_fc_feats, origin_att_feats)
		generated_images_feature = self.encoder_decoder.forward_visual_encoder(generated_fc_feats, generated_att_feats)
		generated_images_cls_prob = self.encoder_decoder.forward_visual_cls(generated_fc_feats, generated_att_feats)

		origin_reports = origin_reports
		generated_reports = generated_reports_prob.argmax(dim=-1)
		additional_bos = torch.full((generated_reports_prob.shape[0], 1), self.args.bos_idx).to(generated_reports_prob.device)
		generated_reports = torch.cat((additional_bos, generated_reports), dim=1)
		origin_text_feature = self.encoder_decoder.forward_linguistic_encoder(origin_reports)
		generated_text_feature = self.encoder_decoder.forward_linguistic_encoder(generated_reports)
		generated_text_cls_prob = self.encoder_decoder.forward_linguistic_cls(generated_reports)

		stage_traceback = (origin_images_feature, origin_text_feature, generated_images_feature, generated_text_feature, generated_images_cls_prob, generated_text_cls_prob)
		return stage_traceback

	def forward(self, images, targets=None, targets_mask=None, hidden_number=None, mode='train'):
		att_feats, fc_feats = self.extract_features(images)
			
		if mode == 'train':
			# forward
			src_decoded, tgt_decoded, img_mask, report_masks, img_cls_prob, txt_cls_prob = self.encoder_decoder(fc_feats, att_feats, targets, targets_mask, hidden_number, mode='forward')
			images_true = patchify(images, self.args.patch_size, self.args.image_shape)
			reports_true = targets
			images_pred = src_decoded
			reports_pred = tgt_decoded
			images_mask = img_mask
			reports_mask = report_masks
			# traceback
			origin_images = images
			generated_images = unpatchify(src_decoded, self.args.patch_size, self.args.image_shape)
			origin_att_feats, origin_fc_feats = self.extract_features(origin_images)
			generated_att_feats, generated_fc_feats = self.extract_features(generated_images)
			origin_images_feature = self.encoder_decoder.forward_visual_encoder(origin_fc_feats, origin_att_feats)
			generated_images_feature = self.encoder_decoder.forward_visual_encoder(generated_fc_feats, generated_att_feats)
			stage_forward = (images_true, reports_true, images_pred, reports_pred, images_mask, reports_mask, img_cls_prob, txt_cls_prob)
			stage_traceback = (origin_images_feature, generated_images_feature)
			return stage_forward, stage_traceback
		elif mode == 'sample':
			output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
			return output
		else:
			raise ValueError






