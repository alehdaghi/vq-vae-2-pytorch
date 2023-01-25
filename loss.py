import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import einops

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


class TripletLoss(nn.Module):
    """Hetero-center-triplet-losses-for-VT-Re-ID.

    Reference:
    Parameter Sharing Exploration and Hetero center triplet losses for VT Re-ID,TMM.
    Code imported from https://github.com/hijune6/Hetero-center-triplet-loss-for-VT-Re-ID/blob/main/loss.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        # targets = labels.unique()

        label_num = len(targets)
        # feat = feats.chunk(label_num, 0)
        # center = []
        # for i in range(label_num):
        #     center.append(torch.mean(feat[i], dim=0, keepdim=True))
        # inputs = torch.cat(center)

        # inputs = einops.rearrange(feats, '(p n) ... -> p n ...', p=label_num).mean(dim=1)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge losses
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.mean(dist_ap * weights_ap, dim=1)
        closest_negative = torch.mean(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class CrossTripletLoss(nn.Module):
    '''
    Compute Triplet losses augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin=0.3):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance
        '''
        super(CrossTripletLoss, self).__init__()
        self.margin = margin
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def _label2similarity(sekf, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity):
        hard_p, p_ind = torch.max(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1)
        # hard_p = sorted_mat_distance[:, 0]
        hard_n, n_ind = torch.min(mat_distance + (9999999.) * (mat_similarity), dim=1)
        # hard_n = sorted_mat_distance[:, 0]
        return hard_p, hard_n, p_ind, n_ind


    def forward(self, anchor, pos, neg, label1, label2, label3, with_index=False):
        '''

		:param anchor: torch.Tensor, [m, dim]
		:param pos: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

        mat_dist = pdist_torch(anchor, pos)
        mat_sim = self._label2similarity(label1, label2)
        hard_p, _, p_ind, _ = self._batch_hard(mat_dist, mat_sim.float())

        mat_dist = pdist_torch(anchor, neg)
        mat_sim = self._label2similarity(label1, label3)
        _, hard_n, _, n_ind = self._batch_hard(mat_dist, mat_sim.float())

        margin_label = torch.ones_like(hard_p)

        if with_index:
            return self.margin_loss(hard_n, hard_p, margin_label), p_ind, n_ind
        else:
            return self.margin_loss(hard_n, hard_p, margin_label)
