import torch
import torch.nn as nn
import torch.nn.functional as F

# Single attention head for Cross-Branch Attention
class CrossAttentionHead(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, hidden_dim: int):
        """
        dim_q = dimension of UNI transformer tokens (1536 for UNI2-h)
        dim_kv = dimension of CNN feature channels (C)
        hidden_dim = per-head hidden size (e.g., 128, 256)
        """
        super().__init__()

        # Q comes from UNI tokens
        # K, V come from CNN features
        self.W_Q = nn.Linear(dim_q, hidden_dim)
        self.W_K = nn.Linear(dim_kv, hidden_dim)
        self.W_V = nn.Linear(dim_kv, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, uni_tokens: torch.Tensor, kv: torch.Tensor):
        """
        uni_tokens : (B, T, dim_q)
            T = number of UNI tokens = 256 patch tokens + 8 reg tokens = 264

        kv : (B, HW, dim_kv)
            CNN features flattened:
            - CNN produces (B, C, 16, 16)
            - reshape to (B, 256, C)
        """
        Q = self.W_Q(uni_tokens) # (B, T, hidden_dim)
        K = self.W_K(kv) # (B, HW, hidden_dim)
        V = self.W_V(kv) # (B, HW, hidden_dim)

        # Attention
        scores = (Q @ K.transpose(-2, -1)) / (self.hidden_dim ** 0.5) # (B, T, HW)
        alpha = torch.softmax(scores, dim=-1) # (B, T, HW)
        out = alpha @ V # (B, T, hidden_dim)

        return out, alpha


# Multi-head Cross-Branch Attention
class CrossBranchMultiHeadedAttention(nn.Module):
    """
    dim_q = transformer token dimension (1536 for UNI2-h)
    dim_kv = CNN feature dimension (channels out of CNN)
    n_hidden = per-head hidden dimension
    num_heads = number of attention heads (e.g., 4, 6, 8)
    """
    def __init__(self, dim_q: int, dim_kv: int, n_hidden: int, num_heads: int):
        super().__init__()
        self.W_O = nn.Linear(num_heads * n_hidden, dim_q)
        self.heads = nn.ModuleList([
            CrossAttentionHead(dim_q, dim_kv, n_hidden) 
            for _ in range(num_heads)
        ])
        # Residual + layernorm after combining heads
        self.norm = nn.LayerNorm(dim_q)
        
    def forward(self, uni_tokens: torch.Tensor, cnn_feat: torch.Tensor):
        """
        uni_tokens: (B, T, dim_q)
            B = batch size
            T = 264 tokens (256 patches + 8 reg tokens)
            dim_q = 1536

        cnn_feat: (B, C, H, W)
            Typically (B, out_channels, 16, 16)
        """
        B, C, H, W = cnn_feat.shape
    
        kv = cnn_feat.permute(0, 2, 3, 1).reshape(B, H*W, C) # Convert (B, C, H, W) -> (B, HW, C)
        
        head_outputs = []
        head_alphas = []
        
        for head in self.heads:
            out, alpha = head(uni_tokens, kv)
            head_outputs.append(out) # (B, T, n_hidden)
            head_alphas.append(alpha) # (B, T, HW)
        
        concatenated = torch.cat(head_outputs, dim=-1) # (B, T, n_hidden * num_heads)
        attn_output = self.W_O(concatenated) # (B, T, dim_q)
        attn_output = self.norm(uni_tokens + attn_output) # residual + layer norm
        attn_alphas = torch.stack(head_alphas, dim=1) # stack heads: (B, num_heads, T, HW)
        
        return attn_output, attn_alphas