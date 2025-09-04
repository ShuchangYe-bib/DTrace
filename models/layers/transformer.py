import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.embedding import WordEmbeddings, PositionalEncoding1D, PositionalEncoding2D
from models.layers.transformer_blocks import TransLayer, TransRMLayer
from models.layers.mlp import ClassificationHead
from models.layers.mae import MAEViT
from modules.utils import random_linguistic_features_removal


class Transformer(nn.Module):

    def __init__(self, args, vocab):
        super(Transformer, self).__init__()
        self.args = args
        self.src_embed = nn.Sequential(
            # PositionalEncoding2D(args.d_model, args.num_patches)
            nn.Identity()
            )
        self.tgt_embed = nn.Sequential(
            WordEmbeddings(args.d_model, vocab), 
            PositionalEncoding1D(args.d_model, args.dropout)
            )
        self.mae = MAEViT(parameter_chkpt_dir=args.mae_checkpoint)
        self.encoder = TransLayer(
            args.num_layers, 
            args.d_model, 
            args.num_heads, 
            args.d_ff, 
            args.dropout
            )
        self.linguistic_encoder = TransLayer(
            args.num_layers, 
            args.d_model, 
            args.num_heads, 
            args.d_ff, 
            args.dropout
            )
        self.linguistic_decoder = TransRMLayer(
            args.num_layers, 
            args.num_heads,
            args.d_model, 
            args.d_ff, 
            args.dropout, 
            args.rm_num_slots, 
            args.rm_d_model,
            args.rm_num_heads
            )
        self.img_head = ClassificationHead(args.d_model, args.num_classes)
        self.txt_head = ClassificationHead(args.d_model, args.num_classes)

    def forward(self, src, tgt, src_mask, tgt_mask, report_ids, report_masks, hidden_number):
        # generation
        if self.args.mask_ratio == "fixed":
            img_mask_ratio = self.args.img_mask_rate
            txt_mask_ratio = self.args.txt_mask_rate
        else:
            img_mask_ratio = hidden_number
            txt_mask_ratio = 1 - img_mask_ratio
        img_embedding = self.src_embed(src)
        txt_embedding = self.tgt_embed(tgt)
        cls_token, img_features, img_mask, img_ids_restore = self.visual_feature_extract(img_embedding, img_mask_ratio)
        txt_removed = random_linguistic_features_removal(txt_embedding, txt_mask_ratio)
        txt_features = self.linguistic_encoder(txt_removed, None)
        src_encoding, image_encoding, text_encoding = self.multi_modal_interaction(img_features, txt_features)
        image_encoding = torch.cat([cls_token.unsqueeze(1), image_encoding], dim=1)
        # image_encoding = torch.cat([cls_token.unsqueeze(1), img_features], dim=1)
        src_decoded = self.mae.forward_decoder(image_encoding, img_ids_restore)
        tgt_decoded = self.decode(src_encoding, None, tgt, tgt_mask)
        # classification
        img_cls_prob = self.img_head(cls_token)
        txt_cls_prob = self.txt_head(txt_features.mean(dim=1))
        return src_decoded, tgt_decoded, img_mask, report_masks, img_cls_prob, txt_cls_prob

    def get_encoder(self):
        encoders = []
        encoders.append(self.src_embed)
        encoders.append(self.tgt_embed)
        encoders.extend(self.mae.get_encoder())
        encoders.append(self.linguistic_encoder)
        encoders.append(self.img_head)
        encoders.append(self.txt_head)
        return encoders

    def encode(self, src, src_mask, left=None, right=None):
        src_embedding = self.src_embed(src)
        _, src_features, _, _ = self.visual_feature_extract(src_embedding, 0)
        src_encoding = self.encoder(src_embedding, src_mask, left, right)
        return src_encoding

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        tgt_embedding = self.tgt_embed(tgt)
        tgt_decoded = self.linguistic_decoder(tgt_embedding, hidden_states, src_mask, tgt_mask)
        return tgt_decoded

    def forward_loss(self, image_true, image_pred, image_mask, image_weight, report_true, report_pred, report_mask, report_weight):
        visual = self.image_reconstruction(image_true, image_pred, image_mask, image_weight)
        linguistic = self.report_generation(report_true, report_pred, report_mask, report_weight)
        loss_dict = {"images" : visual, "reports" : linguistic}
        return loss_dict

    def image_reconstruction(self, origin, reconstructed, mask, weight):
        images = {}
        images["true"] = origin
        images["pred"] = reconstructed
        images["mask"] = mask
        images["weight"] = weight
        return images

    def report_generation(self, origin, decoded, mask, weight):
        reports = {}
        reports["true"] = origin[:, 1:]
        reports["pred"] = decoded
        reports["mask"] = mask[:, 1:]
        reports["weight"] = weight
        return reports

    def visual_feature_extract(self, patched_images, mask_ratio):
        latent, mask, ids_restore = self.mae.forward_encoder(patched_images, mask_ratio)
        cls_token = latent[:, 0, :]
        visual_features = latent[:, 1:, :]
        return cls_token, visual_features, mask, ids_restore

    def get_visual_feature(self, patched_images):
        image_embedding = self.src_embed(patched_images)
        latent, _, _ = self.mae.forward_encoder(patched_images, 0)
        visual_features = latent[:, 1:, :]
        return visual_features

    def get_visual_cls(self, patched_images):
        image_embedding = self.src_embed(patched_images)
        latent, _, _ = self.mae.forward_encoder(patched_images, 0)
        cls_token = latent[:, 0, :]
        cls_prob = self.img_head(cls_token)
        return cls_prob

    def get_linguistic_cls(self, seq):
        txt_embedding = self.tgt_embed(seq)
        txt_features = self.linguistic_encoder(txt_embedding, None)
        cls_prob = self.txt_head(txt_features.mean(dim=1))
        return cls_prob

    def get_linguistic_feature(self, report_ids):
        text_embedding = self.tgt_embed(report_ids)
        text_features = self.linguistic_encoder(text_embedding, None)
        return text_features

    def multi_modal_interaction(self, image_features, text_features):
        left, right = image_features.shape[1], text_features.shape[1]
        src_encoding = self.encoder(torch.cat((image_features, text_features), dim=1), None, left, right)
        image_encoding, text_encoding = src_encoding.split([left, right], dim=1)
        return src_encoding, image_encoding, text_encoding














