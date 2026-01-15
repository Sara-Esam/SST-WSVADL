import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention(nn.Module):
    """
    Cross-attention module for enabling gradient flow between UR-DMU models.
    Query comes from snippet-level model, Key/Value from patch-level model.
    This allows snippet-level model to gain knowledge from patch-level model.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        
        # Separate projections for query (snippet-level) and key-value (patch-level)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, key_value):
        """
        Args:
            query: Features from snippet-level model [batch, seq_len, dim]
            key_value: Features from patch-level model [batch, seq_len, dim]
        Returns:
            Cross-attended features [batch, seq_len, dim]
        """
        b, n, d = query.size()
        
        # Project query, key, value
        q = self.to_q(query)
        kv = self.to_kv(key_value)
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape to multi-head format
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """
    Complete cross-attention block with residual connection and feed-forward
    """
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.):
        super().__init__()
        self.cross_attn = CrossAttention(dim, heads, dim_head, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key_value):
        """
        Args:
            query: Features from patch-level model
            key_value: Features from snippet-level model
        Returns:
            Cross-attended features with residual connections
        """
        # Cross-attention with residual connection
        attended = self.cross_attn(self.norm1(query), key_value)
        query = query + attended
        
        # Feed-forward with residual connection
        query = query + self.ff(self.norm2(query))
        
        return query



