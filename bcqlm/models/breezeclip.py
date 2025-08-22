import torch
import torch.nn as nn
import torch.nn.functional as F

class BreezeCLIP(nn.Module):

    def __init__(self, image_encoder, text_encoder, image_proj, text_proj, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.temperature = temperature

    def forward(self, pixel_values, input_ids, attention_mask):

        # Encode image
        image_features = self.image_encoder(pixel_values)
        image_emb = F.normalize(self.image_proj(image_features), dim=-1)

        # Encode text (use CLS token representation)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_features = text_outputs.last_hidden_state[:, 0, :]
        text_emb = F.normalize(self.text_proj(cls_features), dim=-1)

        return image_emb, text_emb