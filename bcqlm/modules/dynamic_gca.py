import torch
import torch.nn as nn

class DynamicGatedCrossAttention(nn.Module):


    def __init__(self, img_dim: int, text_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        # Projection layers (if text_dim != img_dim, align dims)
        self.text_proj = nn.Linear(text_dim, img_dim) if text_dim != img_dim else nn.Identity()
        self.text_global_proj = nn.Linear(text_dim, img_dim) if text_dim != img_dim else nn.Identity()

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # Gating network
        self.gate_mlp = nn.Sequential(
            nn.Linear(img_dim * 2, img_dim),
            nn.ReLU(),
            nn.Linear(img_dim, 1),
            nn.Sigmoid()
        )

        # Normalization & FFN
        self.norm = nn.LayerNorm(img_dim)
        self.ffn = nn.Sequential(
            nn.Linear(img_dim, img_dim),
            nn.ReLU(),
            nn.Linear(img_dim, img_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_features, text_tokens, text_global):


        # Flatten spatial if needed
        if img_features.ndim == 4:
            B, C, H, W = img_features.shape
            img_features = img_features.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]

        # Project text features to img_dim
        proj_text = self.text_proj(text_tokens)

        # Cross-attention
        attn_out, attn_weights = self.cross_attn(
            query=img_features, key=proj_text, value=proj_text
        )

        # Global text projection
        text_global_proj = self.text_global_proj(text_global)
        B, N, D = img_features.shape
        text_global_exp = text_global_proj.unsqueeze(1).expand(B, N, D)

        # Compute gate
        gate = self.gate_mlp(torch.cat([img_features, text_global_exp], dim=-1))

        # Residual + norm + FFN
        updated = self.norm(img_features + gate * self.dropout(attn_out))
        updated = self.norm(updated + self.ffn(updated))

        return updated, attn_weights