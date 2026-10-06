import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, hidden_norm=True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.hidden_norm = hidden_norm

    def forward(self, hidden):
        if self.hidden_norm:
            hidden = F.normalize(hidden, dim=-1)

        hidden1, hidden2 = torch.chunk(hidden, 2, dim=0)
        batch_size = hidden1.shape[0]
        device = hidden.device

        # Positive pairs: (i, i+batch_size)
        labels = torch.arange(batch_size, device=device)

        # Similarity matrices
        logits_ab = torch.matmul(hidden1, hidden2.T) / self.temperature
        logits_ba = torch.matmul(hidden2, hidden1.T) / self.temperature
        logits_aa = torch.matmul(hidden1, hidden1.T) / self.temperature
        logits_bb = torch.matmul(hidden2, hidden2.T) / self.temperature

        # Mask out self-similarity using fill_diagonal_
        logits_aa = logits_aa.clone().fill_diagonal_(float('-inf'))
        logits_bb = logits_bb.clone().fill_diagonal_(float('-inf'))

        # For hidden1: positive is hidden2[i], negatives are all others
        logits_a = torch.cat([logits_ab, logits_aa], dim=1)  # (batch, 2*batch)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        loss_a = F.cross_entropy(logits_a, labels)
        loss_b = F.cross_entropy(logits_b, labels)
        # Official google-research/simclr sums loss_a + loss_b (no averaging)
        loss = loss_a + loss_b

        return loss, logits_ab, labels
