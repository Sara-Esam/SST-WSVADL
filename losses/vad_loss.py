"""
Loss functions for Video Anomaly Detection
"""
import torch
import torch.nn as nn


class AD_Loss(nn.Module):
    """
    Anomaly Detection Loss (exact copy of UR-DMU's AD_Loss)
    """
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
      
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
        
        anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
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


