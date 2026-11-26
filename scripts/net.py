# Extracted from original_net.py for portable use. See manifests/code_origins.csv.
import torch


class Net(torch.nn.Module):

    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.event = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )
        self.att = torch.nn.Linear(32, 1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                146 + (64 if arm in ["Set", "Attention", "PCRE"] else 0), 128
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),
        )

    def forward(self, x, tok, mask, gate):
        if self.arm in ["Set", "Attention", "PCRE"]:
            h = self.event(tok)
            m = mask.float()
            weights = m / m.sum(2, keepdim=True).clamp_min(1)
            if self.arm != "Set":
                score = self.att(h).squeeze(-1).masked_fill(~mask, -10000.0)
                weights = torch.softmax(score, dim=2) * m
                weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-12)
            h = (h * weights[..., None]).sum(2)
            if self.arm == "PCRE":
                h = h * gate[..., None]
            x = torch.cat([x, h.flatten(1)], 1)
        return self.head(x)
