"""Inductive spatio-temporal quantile backbone for zero-shot cross-city transfer.

Design:
- No node-ID embeddings: every representation is built from transferable inputs
  (static region attributes, frozen LLM geographic embeddings, calendar features),
  so one parameter set serves any city.
- The kNN graph is built from node representations at inference time; the target
  city needs no training-time graph.
- The head outputs a quantile vector per (region, hour), consumed by the
  calibration and decision layers.
- Generative (zero-shot) mode: when the target city has no history window, the
  temporal input is replaced by a calendar-derived "typical-day" query.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_adj(feat: torch.Tensor, k: int) -> torch.Tensor:
    """kNN adjacency by cosine similarity (with self-loops), row-normalized [N,N]."""
    sim = F.normalize(feat, dim=-1) @ F.normalize(feat, dim=-1).T
    n = sim.size(0)
    topk = sim.topk(min(k + 1, n), dim=-1).indices
    adj = torch.zeros_like(sim)
    adj.scatter_(1, topk, 1.0)
    return adj / adj.sum(-1, keepdim=True)


class GraphMix(nn.Module):
    """One message-passing layer: H' = W1 H + W2 (A H)."""

    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)

    def forward(self, h, adj):
        return F.relu(self.w1(h) + self.w2(adj @ h))


class ZeroShotSTBackbone(nn.Module):
    def __init__(self, static_dim_in, llm_dim, hidden, out_len, quantiles, tcn_layers=3, knn_k=8):
        super().__init__()
        self.quantiles = quantiles
        self.out_len = out_len
        self.knn_k = knn_k

        self.static_enc = nn.Sequential(nn.Linear(static_dim_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.llm_enc = nn.Sequential(nn.Linear(llm_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        # Dilated causal convolutions encode the history window; in zero-shot mode
        # the input is a calendar-derived typical-day query instead.
        self.tcn = nn.ModuleList(
            [nn.Conv1d(1 if i == 0 else hidden, hidden, 3, dilation=2 ** i, padding=2 ** i) for i in range(tcn_layers)]
        )
        self.calendar_enc = nn.Linear(24 + 7, hidden)  # hour + day-of-week one-hot
        # Retrieval augmentation: typical-day profiles of similar source regions
        # (log1p space, [B,N,out_len]); optional, used only in ablations.
        self.retr_enc = nn.Sequential(nn.Linear(out_len, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.gnn = nn.ModuleList([GraphMix(hidden) for _ in range(2)])
        self.head = nn.Linear(hidden, out_len * len(quantiles))

    def encode_history(self, x):  # x: [B, N, T]
        b, n, t = x.shape
        h = x.reshape(b * n, 1, t)
        for conv in self.tcn:
            h = F.relu(conv(h))[..., :t]
        return h[..., -1].reshape(b, n, -1)  # [B, N, hidden]

    def forward(self, static, llm_emb, calendar, history=None, retr=None):
        """static [N,Fs]; llm_emb [N,Fl]; calendar [B, 31]; history [B,N,T] or None
        (zero-shot); retr [B,N,out_len] retrieved typical-day profiles (optional)."""
        node = self.static_enc(static) + self.llm_enc(llm_emb)          # [N, h]
        adj = knn_adj(node.detach(), self.knn_k)
        cal = self.calendar_enc(calendar).unsqueeze(1)                   # [B, 1, h]
        if history is not None:
            h = self.encode_history(history) + node.unsqueeze(0) + cal   # history mode
        else:
            h = node.unsqueeze(0).expand(cal.size(0), -1, -1) + cal      # generative mode
        if retr is not None:
            h = h + self.retr_enc(retr)
        for layer in self.gnn:
            h = layer(h, adj)
        out = self.head(h)                                               # [B, N, out_len*Q]
        b, n, _ = out.shape
        return out.reshape(b, n, self.out_len, len(self.quantiles))
