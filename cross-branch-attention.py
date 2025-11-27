import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionHead(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, hidden_dim: int):
        super().__init__()
        self.W_Q = nn.Linear(dim_q, hidden_dim)
        self.W_K = nn.Linear(dim_kv, hidden_dim)
        self.W_V = nn.Linear(dim_kv, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, uni_tokens: torch.Tensor, kv: torch.Tensor):
        """
        Inputs:
          uni_tokens : (B, T, dim_q)    — transformer tokens
          kv         : (B, HW, dim_kv)  — CNN features (already flattened)

        Outputs:
          out        : (B, T, hidden_dim)
          alpha      : (B, T, HW)       — attention weights
        """
        Q = self.W_Q(uni_tokens) # (B, T, hidden_dim)
        K = self.W_K(kv) # (B, HW, hidden_dim)
        V = self.W_V(kv) # (B, HW, hidden_dim)

        # Attention
        scores = (Q @ K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        alpha = torch.softmax(scores, dim=-1)
        out = alpha @ V

        return out, alpha


class CrossBranchMultiHeadedAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, n_hidden: int, num_heads: int):
        super().__init__()
        self.W_O = nn.Linear(num_heads * n_hidden, dim_q)
        self.heads = nn.ModuleList([
            CrossAttentionHead(dim_q, dim_kv, n_hidden) 
            for _ in range(num_heads)
        ])
        self.norm = nn.LayerNorm(dim_q)
        
    def forward(self, uni_tokens: torch.Tensor, cnn_feat: torch.Tensor):
        """
        uni_tokens: (B, T, dim_q)
        cnn_feat: (B, C, H, W)
        """
        B, C, H, W = cnn_feat.shape
    
        kv = cnn_feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        head_outputs = []
        head_alphas = []
        
        for head in self.heads:
            out, alpha = head(uni_tokens, kv)
            head_outputs.append(out)
            head_alphas.append(alpha)
        
        concatenated = torch.cat(head_outputs, dim=-1)
        attn_output = self.W_O(concatenated)
        attn_output = self.norm(uni_tokens + attn_output)
        attn_alphas = torch.stack(head_alphas, dim=1)
        
        return attn_output, attn_alphas