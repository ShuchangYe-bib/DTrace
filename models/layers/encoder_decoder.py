import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers.base import Attention
from models.layers.transformer import Transformer
from modules.utils import clones, pack_wrapper


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def closest_factors(n):
    if n <= 1:
        return (1, n)
    a = int(n**(1/2))
    while a > 1:
        if n % a == 0:
            return (a, n // a)
        a -= 1
    return (1, n)


class EncoderDecoder(Attention):

    def make_model(self, args, tgt_vocab):
        model = Transformer(args, tgt_vocab)
        for name, p in model.named_parameters():
            if 'mae' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(args, tgt_vocab)
        self.text_generation = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def get_encoder(self):
        encoders = []
        encoders.append(self.att_embed)
        encoders.extend(self.model.get_encoder())
        return encoders

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None
        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, report_masks, hidden_number, att_masks=None):
        report_ids = seq
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        src_decoded, tgt_decoded, img_mask, report_masks, img_cls_prob, txt_cls_prob = self.model(att_feats, seq, att_masks, seq_mask, report_ids, report_masks, hidden_number)
        tgt_decoded = F.log_softmax(self.text_generation(tgt_decoded), dim=-1)
        return src_decoded, tgt_decoded, img_mask, report_masks, img_cls_prob, txt_cls_prob

    def forward_visual_encoder(self, fc_feats, att_feats):
        att_feats, _, _, _ = self._prepare_feature_forward(att_feats, None, None)
        visual_features = self.model.get_visual_feature(att_feats)
        return visual_features

    def forward_visual_cls(self, fc_feats, att_feats):
        att_feats, _, _, _ = self._prepare_feature_forward(att_feats, None, None)
        cls_prob = self.model.get_visual_cls(att_feats)
        return cls_prob

    def forward_linguistic_cls(self, seq):
        seq = seq[:, :-1]
        cls_prob = self.model.get_linguistic_cls(seq)
        return cls_prob

    def forward_linguistic_encoder(self, seq):
        seq = seq[:, :-1]
        linguistic_features = self.model.get_linguistic_feature(seq)
        return linguistic_features

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]

        