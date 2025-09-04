import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from modules.utils import unpatchify


class LanguageModelCriterion(nn.Module):
    
    def __init__(self, loss_fn=None):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = loss_fn

    def report_generation_loss(reports_true, reports_pred, reports_mask, reports_weight=1):
        pure_loss = nn.CrossEntropyLoss(reduction='none')
        loss = pure_loss(reports_pred.permute(0,2,1), reports_true)
        loss = (loss*reports_mask).sum() / reports_mask.sum()
        batch_loss = loss * reports_weight
        return batch_loss

    def image_reconstruction_loss(images_true, images_pred, images_mask, images_weight=1):
        loss = (images_pred-images_true)**2
        loss = loss.mean(dim=-1)
        loss = (loss*images_mask).sum() / images_mask.sum()
        batch_loss = loss * images_weight

        mask_ratio = torch.sum(images_mask==1) / images_mask.numel()
        if mask_ratio > 0.7:
            images_pred = images_pred * images_mask.unsqueeze(2) + images_true * (1-images_mask).unsqueeze(2)
            norm_target = unpatchify((images_true - images_true.min()) / (images_true.max() - images_true.min()), 16, (3, 224, 224))
            norm_pred = unpatchify((images_pred - images_pred.min()) / (images_pred.max() - images_pred.min()), 16, (3, 224, 224))

        return batch_loss

    def gloria_loss(local_global_image_text_features, labels, temp1=4.0, temp2=5.0, temp3=10.0, eps=1e-8):
        
        def global_semantic_contrastive_loss(contexts, queries, labels, temp1=4.0, temp2=5.0, temp3=10.0, eps=1e-8, agg="mean"):
            assert contexts.shape[0] == queries.shape[0]
            batch_size = contexts.shape[0]
            global_contexts = contexts.mean(axis=1)
            global_queries = queries.mean(axis=1)
            batch_size = contexts.shape[0]
            labels = labels.to(contexts.device)
            label_sim = F.cosine_similarity(labels.unsqueeze(1), labels.unsqueeze(0), dim=2)
            label_sim = (label_sim+1)/2
            if global_contexts.dim() == 2:
                global_contexts = global_contexts.unsqueeze(0)
                global_queries = global_queries.unsqueeze(0)
            global_contexts_norm = torch.norm(global_contexts, 2, dim=2, keepdim=True)
            global_queries_norm = torch.norm(global_queries, 2, dim=2, keepdim=True)
            scores0 = torch.bmm(global_contexts, global_queries.transpose(1, 2))
            norm0 = torch.bmm(global_contexts_norm, global_queries_norm.transpose(1, 2))
            scores0 = scores0 / norm0.clamp(min=eps) * temp3
            scores0 = scores0.squeeze(0)
            scores1 = scores0.transpose(0, 1)
            pure_loss = nn.BCELoss(reduction='none')
            global_loss0 = pure_loss(F.softmax(scores0, 1), F.softmax(label_sim, 1)).mean()
            global_loss1 = pure_loss(F.softmax(scores1, 1), F.softmax(label_sim, 1)).mean()
            batch_loss = (global_loss0+global_loss1)/2
            return batch_loss

        def local_semantic_contrastive_loss(input_features, base_features, labels, temp1=4.0, temp2=5.0, temp3=10.0, eps=1e-8, agg="mean"):
            """
            @param input_features (torch.Tensor):
            Tensor of shape (batch_size, n_input_features_features, n_feature_maps) which maps to the shape of
            base_feature for computing the loss with base_feature

            @param base_features (torch.Tensor):
            Tensor of shape (batch_size, n_base_features, n_feature_maps) representing the base_feature features
            """
            assert input_features.shape[0] == base_features.shape[0]
            batch_size, n_base_features, n_feature_maps = base_features.shape
            # calculating similarity between labels (0: not similar -> 1: similar)
            labels = labels.to(input_features.device)
            label_sim = F.cosine_similarity(labels.unsqueeze(1), labels.unsqueeze(0), dim=2)
            label_sim = (label_sim + 1) / 2
            similarities = []
            for i in range(base_features.shape[0]):
                base_feature = base_features[i, :n_base_features].unsqueeze(0).contiguous()
                base_feature = base_feature.repeat(batch_size, 1, 1)
                # compute attention-weighted image features (batch_size, n_feature_maps, n_base_features)
                weighted_input_features = attn_weighted_input_features(input_features, base_feature, temp1)
                # similarity between the word embedding and the image feature vector
                # - base_feature: (batch_size, n_base_features, n_feature_maps)
                # - weighted_input_features: (batch_size, n_base_features, n_feature_maps)
                # - row_sim: (batch_size, n_base_features)
                row_sim = cosine_similarity(base_feature, weighted_input_features, dim=-1, eps=eps)
                row_sim.mul_(temp2).exp_()
                if agg == "sum":
                    row_sim = row_sim.sum(dim=1, keepdim=True) # (batch_size, 1)
                elif agg == "mean":
                    row_sim = row_sim.mean(dim=1, keepdim=True) # (batch_size, 1)
                else:
                    raise ValueError
                row_sim = torch.log(row_sim)
                similarities.append(row_sim)
            similarities = torch.cat(similarities, 1) # batch_size, batch_size
            similarities0 = similarities * temp3
            similarities1 = similarities0.transpose(0, 1)
            pure_loss = nn.BCELoss(reduction='none')
            local_loss0 = pure_loss(F.softmax(similarities0, dim=1), F.softmax(label_sim, 1)).mean()
            local_loss1 = pure_loss(F.softmax(similarities1, dim=1), F.softmax(label_sim, 1)).mean()
            batch_loss = (local_loss0+local_loss1)/2
            return batch_loss

        local_image_features, global_image_features, local_text_features, global_text_features = local_global_image_text_features
        global_loss = LanguageModelCriterion.global_semantic_contrastive_loss(global_image_features, global_text_features, labels)
        local_loss = LanguageModelCriterion.local_semantic_contrastive_loss(local_image_features, local_text_features, labels)
        return (local_loss+global_loss)/2

    def classification_loss(cls_labels, cls_prob, weight=1):
        cls_labels = ((cls_labels+1)/2).to(cls_prob.device)
        pure_loss = nn.BCELoss(reduction="mean")
        loss = pure_loss(cls_prob, cls_labels)
        batch_loss = loss * weight
        return batch_loss

    def word_importance_loss(reports_true, reports_pred, reports_mask, tf_idfs, weight=1):
        reports_true_expand_dim = reports_true.long().unsqueeze(2)
        prob = reports_pred.gather(2, reports_true_expand_dim).squeeze(2)
        loss = -(tf_idfs * prob * reports_mask).sum() / tf_idfs.sum()
        batch_loss = loss * weight
        return batch_loss

    def feature_difference_loss(feature_true, feature_pred, weight):
        loss = nn.MSELoss(reduction='mean')(feature_true, feature_pred)
        batch_loss = loss*weight
        return batch_loss

    def forward(self, images_true, reports_true, images_pred, reports_pred, images_mask, reports_mask, images_weight, reports_weight, cls_labels, cls_prob, tf_idfs):
        batch_loss = None
        if self.loss_fn == "Hybrid":
            image_reconstruction_loss = LanguageModelCriterion.image_reconstruction_loss(images_true, images_pred, images_mask, images_weight)
            report_generation_loss = LanguageModelCriterion.report_generation_loss(reports_true, reports_pred, reports_mask, reports_weight)
            batch_loss = report_generation_loss + image_reconstruction_loss
        else:
            raise ValueError
        return batch_loss

    def forward_loss(self, images_true, reports_true, images_pred, reports_pred, images_mask, reports_mask, images_weight, reports_weight, cls_labels, img_cls_prob, txt_cls_prob, tf_idfs):
        image_reconstruction_loss = LanguageModelCriterion.image_reconstruction_loss(images_true, images_pred, images_mask, images_weight)
        image_classification_loss = LanguageModelCriterion.classification_loss(cls_labels, img_cls_prob, 1-images_weight)
        report_generation_loss = LanguageModelCriterion.report_generation_loss(reports_true, reports_pred, reports_mask, reports_weight)
        word_importance_loss = LanguageModelCriterion.word_importance_loss(reports_true, reports_pred, reports_mask, tf_idfs, reports_weight)
        report_classification_loss = LanguageModelCriterion.classification_loss(cls_labels, txt_cls_prob, 1-reports_weight)
        return image_reconstruction_loss, image_classification_loss, report_generation_loss, word_importance_loss, report_classification_loss

    def traceback_loss(self, images_true_features, reports_true_features, images_pred_features, reports_pred_features, images_cls_prob, reports_cls_prob, images_weight, reports_weight, cls_labels):
        images_true_features = images_true_features.detach()
        images_feature_loss = LanguageModelCriterion.classification_loss(cls_labels, images_cls_prob)
        reports_feature_loss = LanguageModelCriterion.classification_loss(cls_labels, reports_cls_prob)
        return images_feature_loss, reports_feature_loss

def compute_loss(infos, stage, loss_fn):
    criterion = LanguageModelCriterion(loss_fn)
    if stage == "forward":
        loss = criterion.forward_loss(*infos)
    elif stage == "traceback":
        loss = criterion.traceback_loss(*infos)
    else:
        loss = None
    return loss

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()





