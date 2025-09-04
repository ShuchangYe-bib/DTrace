import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.vision_transformer import PatchEmbed


class VisualExtractor(nn.Module):

	def __init__(self, args):
		super(VisualExtractor, self).__init__()
		self.args = args
		self.model_name = args.visual_extractor
		self.device = args.device
		self.model = PatchEmbed(args.image_shape[1], args.patch_size, args.image_shape[0], args.d_vf)
		self.initialize_weights()

	def initialize_weights(self):
		w = self.model.proj.weight.data
		nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

	def forward(self, images):
		patch_feats = self.model(images).to(self.device)
		avg_feats = torch.mean(patch_feats, -1)
		return patch_feats, avg_feats


