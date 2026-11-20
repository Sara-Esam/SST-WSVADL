"""
Loss functions for Video Anomaly Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AD_Loss(nn.Module):
    """
    Anomaly Detection Loss (exact copy of UR-DMU's AD_Loss)
    """
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
      
    def video_score(self, att):
        """MIL pooling: mean of the top t//16+1 instance scores."""
        t = att.size(1)
        return torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()

        triplet = result["triplet_margin"]
        att = result['frame']
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]
        b = _label.size(0)//2
        t = att.size(1)      
        
        anomaly = self.video_score(att)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt, t//16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b), device=_label.device))
        
        A_att = torch.topk(A_att, t//16 + 1, dim=-1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b), device=_label.device))

        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))

        cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        loss['kl_loss'] = kl_loss
        return cost, loss




class SpatialAD_Loss(AD_Loss):
    """
    AD_Loss for the patch branch, with terms that ask for one compact region.

    Patch scores arrive flat as (b, T*H*W) in (tubelet-frame, row, col) order; the
    DTFE model masks tokens during training instead of reordering them, so the
    grid can be rebuilt with a reshape.

    pool="window": the MIL instance is the best k x k window of the average-pooled
    grid rather than the top t//16+1 patches, so the region that must fire is
    connected by construction.

    contrast_weight: on anomalous clips, hinge on (best region - mean of the
    bottom_k patches) >= margin per tubelet-frame. A constant map fails it.

    sparsity_weight: mean patch score on anomalous clips, so only part of the
    frame fires.
    """

    def __init__(self, grid_shape, pool="topk", window=3, contrast_weight=0.0,
                 contrast_margin=0.3, bottom_k=16, sparsity_weight=0.0):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.pool = pool
        self.window = window
        self.contrast_weight = contrast_weight
        self.contrast_margin = contrast_margin
        self.bottom_k = bottom_k
        self.sparsity_weight = sparsity_weight

    def grid(self, att):
        t, h, w = self.grid_shape
        if att.size(1) != t * h * w:
            raise ValueError(f"{att.size(1)} patch scores do not fit grid {self.grid_shape}")
        return att.reshape(att.size(0), t, h, w)

    def region_scores(self, att):
        """Per-frame best-window score, shape (b, t)."""
        t, h, w = self.grid_shape
        pooled = F.avg_pool2d(self.grid(att).reshape(-1, 1, h, w), self.window, stride=1)
        return pooled.reshape(att.size(0), t, -1).max(-1)[0]

    def video_score(self, att):
        if self.pool == "window":
            return self.region_scores(att).max(-1)[0]
        return super().video_score(att)

    def forward(self, result, _label):
        cost, loss = super().forward(result, _label)
        att = result["frame"]
        b = _label.size(0) // 2
        anomalous = att[b:]  # batches are [normal, abnormal]

        if self.contrast_weight > 0:
            t, h, w = self.grid_shape
            if self.pool == "window":
                top = self.region_scores(anomalous)
            else:
                top = torch.topk(self.grid(anomalous).reshape(b, t, -1), h * w // 16 + 1, dim=-1)[0].mean(-1)
            bottom = -torch.topk(-self.grid(anomalous).reshape(b, t, -1), self.bottom_k, dim=-1)[0].mean(-1)
            contrast = torch.relu(self.contrast_margin - (top - bottom)).mean()
            cost = cost + self.contrast_weight * contrast
            loss["contrast"] = contrast

        if self.sparsity_weight > 0:
            sparsity = anomalous.mean()
            cost = cost + self.sparsity_weight * sparsity
            loss["sparsity"] = sparsity

        loss["total_loss"] = cost
        return cost, loss
