"""
SIMPLEST POSSIBLE TRANSFORMER DECODER
No cross-attention, no complexity, just: text + point + time → generate
"""

import torch
import torch.nn as nn
from src.models.architectures import PositionalEncoding

class Decoder_TRANSFORMER_SIMPLE(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, 
                 pose_rep, glob, glob_rot, latent_dim=256, ff_size=1024, num_layers=4, 
                 num_heads=4, dropout=0.1, activation="gelu", ablation=None, **kargs):
        super().__init__()
        
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        
        # 1. Point embeddings (initial positions → identity)
        self.point_encoder = nn.Linear(nfeats, latent_dim)
        
        # 2. Time embeddings (frame index → temporal position)
        self.time_encoder = PositionalEncoding(latent_dim, dropout)
        
        # 3. CLIP text projection
        self.clip_proj = nn.Linear(512, latent_dim)
        
        # 4. Simple transformer: just self-attention on [text + point + time] tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output layer
        self.finallayer = nn.Linear(latent_dim, nfeats)
    
    def forward(self, batch, use_text_emb=False):
        z = batch["clip_text_emb"] if use_text_emb else batch["z"]
        mask = batch["mask"]
        initial_grid = batch["initial_grid"]  # [bs, njoints, 2]
        
        bs, nframes = mask.shape
        njoints = self.njoints
        
        # Project CLIP text to latent
        if z.dim() == 3:
            z = z[0] if z.shape[0] == 1 else z[:, 0]
        if z.shape[1] == 512:
            z = self.clip_proj(z)  # [bs, latent_dim]
        
        # 1. Point embeddings: [bs, njoints, latent_dim]
        point_emb = self.point_encoder(initial_grid)
        
        # 2. Time embeddings: [nframes, bs, latent_dim]
        time_emb = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        time_emb = self.time_encoder(time_emb)
        
        # 3. Text embedding: [1, bs, latent_dim] (broadcast to all frames/points)
        text_emb = z.unsqueeze(0)  # [1, bs, latent_dim]
        
        # COMBINE: For each (point, time) pair: text + point + time
        # Create sequence: [text_token, point0_time0, point0_time1, ..., point35_time29]
        # Total: 1 + (njoints * nframes) tokens
        
        # Expand text to all positions
        text_expanded = text_emb.expand(nframes * njoints, -1, -1)  # [nframes*njoints, bs, latent_dim]
        
        # Expand points to all frames
        point_expanded = point_emb.unsqueeze(0).expand(nframes, -1, -1)  # [nframes, bs, njoints, latent_dim]
        point_expanded = point_expanded.reshape(nframes * njoints, bs, latent_dim)  # [nframes*njoints, bs, latent_dim]
        
        # Expand time to all points
        time_expanded = time_emb.unsqueeze(2).expand(-1, -1, njoints, -1)  # [nframes, bs, njoints, latent_dim]
        time_expanded = time_expanded.reshape(nframes * njoints, bs, latent_dim)  # [nframes*njoints, bs, latent_dim]
        
        # Combine: text + point + time
        combined = text_expanded + point_expanded + time_expanded  # [nframes*njoints, bs, latent_dim]
        
        # Add text token at the beginning
        sequence = torch.cat([text_emb, combined], dim=0)  # [1 + nframes*njoints, bs, latent_dim]
        
        # Transformer: self-attention on all tokens
        # Text token can attend to all (point, time) pairs
        # Each (point, time) can attend to text and other (point, time) pairs
        mask_seq = torch.cat([
            torch.ones(1, bs, dtype=bool, device=mask.device),  # Text token always valid
            mask.T.unsqueeze(1).expand(-1, njoints, -1).reshape(nframes * njoints, bs).T.bool()
        ], dim=1)  # [bs, 1 + nframes*njoints]
        
        output_seq = self.transformer(sequence, src_key_padding_mask=~mask_seq)
        
        # Take only the (point, time) tokens (skip text token)
        point_time_tokens = output_seq[1:]  # [nframes*njoints, bs, latent_dim]
        
        # Generate displacements
        displacements_flat = self.finallayer(point_time_tokens)  # [nframes*njoints, bs, nfeats]
        displacements = displacements_flat.reshape(nframes, njoints, bs, nfeats)
        displacements = displacements.permute(2, 1, 3, 0)  # [bs, njoints, nfeats, nframes]
        
        # Force frame 0 to be zero
        displacements[:, :, :, 0] = 0.0
        
        # Add to initial grid
        initial_expanded = initial_grid.unsqueeze(-1).expand(-1, -1, -1, nframes)
        output = initial_expanded + displacements
        
        batch["txt_output" if use_text_emb else "output"] = output
        return batch

