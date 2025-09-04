import cv2
import copy
import nltk
import torch
import random
import textwrap
import numpy as np
import torch.nn as nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def freeze_model(models):
    for model in models:
        if isinstance(model, nn.Parameter):
            model.requires_grad = False
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

def unfreeze_model(models):
    for model in models:
        if isinstance(model, nn.Parameter):
            model.requires_grad = True
        else:
            model.train()
            for param in model.parameters():
                param.requires_grad = True

def patchify(imgs, patch_size, image_shape):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert image_shape[1] == image_shape[2] and image_shape[1] % p == 0

    h = w = image_shape[1] // p
    c = image_shape[0]
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x

def unpatchify(x, patch_size, image_shape):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    c = image_shape[0]
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs

def extract_patches(images, patch_size):
    if images.dim() == 3:
        images = images.unsqueeze(0)
    batch, channels, height, width = images.shape
    height, width = height // patch_size, width // patch_size
    x = images.reshape(batch, channels, height, patch_size, width, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(batch, height * width, patch_size**2 * channels)
    return x

def merge_patches(images, patch_size):
    batch, length, _ = images.shape
    side = int(length**0.5)
    height = width = patch_size * side
    channels = images.shape[2] // (patch_size**2)
    x = images.reshape(batch, side, side, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.reshape(batch, channels, height, width)
    return x

def random_linguistic_features_removal(tensor, mask_ratio):
    # Create a binary mask of the correct length
    length = tensor.size(1)
    num_features_to_keep = max(1, int(length * (1-mask_ratio)))
    # The mask initially has 1's for the features we keep, 0's for the ones we don't
    mask = torch.cat([torch.ones(num_features_to_keep), torch.zeros(length - num_features_to_keep)])
    # We randomly shuffle the mask to decide which features to keep
    mask = mask[torch.randperm(length)]
    # Expand dimensions of the mask for compatibility with the input tensor
    mask = mask.unsqueeze(0).unsqueeze(-1).expand_as(tensor)
    # Apply the mask to the tensor
    masked_tensor = tensor[mask.bool()].view(tensor.size(0), num_features_to_keep, tensor.size(2))
    return masked_tensor

def random_visual_features_removal(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = max(1, int(L * (1 - mask_ratio)))
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore

def visual_features_restore(x, mask_token, ids_restore, cls_token=False):
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    if cls_token:
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    else:
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_
    return x

def random_image_mask(patches, mask_ratio):
    n = patches.shape[0]
    num_delimiters = int(patches.shape[1]**0.5)
    n_patches_to_mask = int(num_delimiters*num_delimiters*mask_ratio)
    mask = torch.ones(n, num_delimiters*num_delimiters).to(patches.device)
    mask[:, :n_patches_to_mask] = 0
    for i in range(n):
        rand_idx = torch.randperm(mask[i].nelement())
        mask[i] = mask[i][rand_idx]
    masked_patches = patches * mask.unsqueeze(2)
    masked_positions = 1 - mask
    return masked_patches, masked_positions

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


def attn_weighted_input_features(input_features, base_feature, temp1=4.0):
    """
    @param input_features (torch.Tensor):
    Tensor of shape (batch_size, n_input_features_features, n_feature_maps) used to compute the
    attention weights with respect to the base_feature features
    
    @param base_feature (torch.Tensor):
    Tensor of shape (batch_size, n_base_features, n_feature_maps) used as a reference to
    compute attention weights with input_features

    @return weighted_input_features (torch.Tensor):
    Tensor of shape (batch_size, n_base_features, n_feature_maps) which has the same
    dimension as base_feature representing the weighted sum of the input features based on the
    attention weights
    """
    input_featuresT = input_features.transpose(1, 2).contiguous()

    # compute attention weights
    # - base_feature: (batch_size, n_base_features, n_feature_maps)
    # - input_featuresT: (batch_size, n_feature_maps, n_input_features_features)
    # - attn: (batch_size, n_base_features, n_input_features_features)
    attn = torch.bmm(base_feature, input_featuresT)
    attn = nn.Softmax(dim=1)(attn)

    # scales the values of the attention weights
    attn = attn * temp1
    attn = nn.Softmax(dim=2)(attn) # (batch_size, n_base_features, n_input_features_features)

    # compute weighted context vector
    # - attn: (batch_size, n_base_features, n_input_features_features)
    # - input_features: (batch_size, n_input_features_features, n_feature_maps)
    # - weighted_input_features: (batch_size, n_base_features, n_feature_maps)
    weighted_input_features = torch.bmm(attn, input_features)

    return weighted_input_features


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x

def blur(img, kernel=(25, 25)):
    result = cv2.blur(img, kernel, 0)
    # result = cv2.addWeighted(img, 0.3, result, 0.7, 0) # surface blur
    return result


def hard_light_blending(img1, img2):
    img1, img2 = img1/255, img2/255
    # Calculate the result of hard light blending mode
    mask = img2 > 0.5
    result = np.zeros_like(img1)
    result[mask] = 1 - (1 - img1[mask]) * (1 - 2*(img2[mask] - 0.5))
    result[~mask] = img1[~mask] * 2 * img2[~mask]
    result = result*255
    return result


def screen_blending(img1, img2):
    result = 1 - ((1 - img2/255) * (1 - img1/255))
    result = result * 255
    return result


def multiply_blending(img1, img2):
    result = img1/255 * img2/255
    result = result * 255
    return result


def darken_blending(img1, img2):
    return np.fmin(heatmap, image)


def generate_heatmap(image, weights, n_channels=3, heatmap_visability=0.7):
    image = image.squeeze(0)
    image = image.transpose(1, 2, 0)
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5)) # 1d to 2d
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights)) # normalize the weights
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    # heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    weights = blur(weights)
    if n_channels == 3:
        weights = cv2.cvtColor(weights, cv2.COLOR_GRAY2RGB)
    elif n_channels == 1:
        weights = np.expand_dims(weights, -1)
    # result = heatmap * heatmap_visability + image * (1-heatmap_visability) # overlay
    result = hard_light_blending(image, multiply_blending(weights, weights))
    
    return np.hstack((image, weights, result))


def post_process_report(report):
    sents = nltk.sent_tokenize(report)
    result = ""
    for sent in sents:
        words = nltk.word_tokenize(sent)
        if words[-1] == ".":
            words = words[:-1]
            result += (" ".join(words) + ". ").capitalize()
        else:
            result += " ".join(words).capitalize()
    return result


def print_report(report, name="", width=60):
    name = name.capitalize()
    lines = textwrap.wrap(report, width=(width-4))

    print("-"*width)
    print(name, "Report:")
    # Print lines with indentation
    for line in lines:
        print(f"{'':4s}" + line)
    print("-"*width)

    