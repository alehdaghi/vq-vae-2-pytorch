import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import einops


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
