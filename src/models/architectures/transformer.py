import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", clip_model=None, **kargs):
        super().__init__()
        
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        
        # SIMPLEST: Just encode trajectories
        # Note: "trajectory_embedding" (not "skeleton") - we encode point trajectories, not skeleton joints
        self.input_feats = self.njoints * self.nfeats
        self.trajectory_embedding = nn.Linear(self.input_feats, self.latent_dim)
        # Keep old name for backwards compatibility
        self.skelEmbedding = self.trajectory_embedding
        
        # Velocity/acceleration embeddings (if augmented input)
        self.velocityEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        self.accelerationEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        # SIMPLEST: Use CLIP image encoder for overlays (frozen)
        self.clip_model = clip_model
        if clip_model is not None:
            # Project CLIP image features to latent_dim
            self.image_proj = nn.Linear(512, self.latent_dim)
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout)
        
        # Learnable aggregation query (from original MotionCLIP)
        # muQuery learns to aggregate temporal information through attention
        # This is MUCH better than mean pooling which loses info when frames are zero-centered
        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # No CLIP projection needed - latent_dim (512) == CLIP dim
        # Just normalize for alignment

    def forward(self, batch):
        x, mask = batch["x"], batch["mask"]
        bs, njoints, nfeats, nframes = x.shape
        
        # 1. Encode trajectories
        if nfeats == 6:
            # Positions + velocities + accelerations
            positions = x[:, :, :2, :].permute(3, 0, 1, 2).reshape(nframes, bs, njoints * 2)
            velocities = x[:, :, 2:4, :].permute(3, 0, 1, 2).reshape(nframes, bs, njoints * 2)
            accelerations = x[:, :, 4:6, :].permute(3, 0, 1, 2).reshape(nframes, bs, njoints * 2)
            pos_embed = self.skelEmbedding(positions)
            vel_embed = self.velocityEmbedding(velocities)
            acc_embed = self.accelerationEmbedding(accelerations)
            traj_features = pos_embed + vel_embed + acc_embed  # [nframes, bs, latent_dim]
        elif nfeats == 4:
            positions = x[:, :, :2, :].permute(3, 0, 1, 2).reshape(nframes, bs, njoints * 2)
            velocities = x[:, :, 2:, :].permute(3, 0, 1, 2).reshape(nframes, bs, njoints * 2)
            pos_embed = self.skelEmbedding(positions)
            vel_embed = self.velocityEmbedding(velocities)
            traj_features = pos_embed + vel_embed
        else:
            # Just positions
            x_flat = x.permute(3, 0, 1, 2).reshape(nframes, bs, njoints * nfeats)
            traj_features = self.skelEmbedding(x_flat)  # [nframes, bs, latent_dim]
        
        # 2. Encode overlayed images (if available)
        if 'trajectory_overlays' in batch and self.clip_model is not None:
            overlays = batch['trajectory_overlays']  # [bs, nframes, 3, 224, 224]
            frames_flat = overlays.view(-1, 3, 224, 224)  # [bs*nframes, 3, 224, 224]
            
            with torch.no_grad():
                # CLIP image encoder (frozen)
                image_features = self.clip_model.encode_image(frames_flat).float()  # [bs*nframes, 512]
            
            # Project to latent_dim
            image_features = self.image_proj(image_features)  # [bs*nframes, latent_dim]
            image_features = image_features.view(bs, nframes, self.latent_dim).permute(1, 0, 2)  # [nframes, bs, latent_dim]
            
            # Combine: trajectory + image features
            combined_features = traj_features + image_features  # [nframes, bs, latent_dim]
        else:
            combined_features = traj_features
        
        # 3. Add positional encoding
        combined_features = self.sequence_pos_encoder(combined_features)  # [nframes, bs, latent_dim]
        
        # 4. Prepend muQuery (learnable aggregation token) - from original MotionCLIP
        # muQuery learns to attend to all frames and aggregate temporal information
        bs = combined_features.shape[1]
        mu_query = self.muQuery.expand(1, bs, -1)  # [1, bs, latent_dim]
        xseq = torch.cat([mu_query, combined_features], dim=0)  # [1 + nframes, bs, latent_dim]
        
        # 5. Transformer (self-attention across muQuery + all frames)
        # Extend mask to include muQuery (always valid, so True)
        mu_query_mask = torch.ones(bs, 1, dtype=torch.bool, device=mask.device)
        maskseq = torch.cat([mu_query_mask, mask], dim=1)  # [bs, 1 + nframes]
        
        output = self.transformer(xseq, src_key_padding_mask=~maskseq.bool())  # [1 + nframes, bs, latent_dim]
        
        # 6. Extract mu from muQuery output (NOT mean pooling!)
        # muQuery has attended to all frames and aggregated the information
        mu = output[0]  # [bs, latent_dim] - muQuery after attention
        
        # 7. Normalize for CLIP alignment (no projection needed - latent_dim == CLIP dim)
        mu_clip_norm = mu / mu.norm(dim=-1, keepdim=True)
        
        return {
            "mu": mu,
            "mu_clip": mu_clip_norm
        }


class Decoder_MLP(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="linear",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        self.input_feats = self.njoints * self.nfeats
        self.activation = activation  # CRITICAL: Store activation type!
        
        # Calculate output dimension: [njoints, nfeats, nframes]
        self.output_dim = self.njoints * self.nfeats * self.num_frames
        
        # FIXED MLP - With LayerNorm and skip connections to prevent collapse
        # This is CRITICAL for normalized CLIP inputs!
        self.fc1 = nn.Linear(latent_dim, 2048)
        self.ln1 = nn.LayerNorm(2048)  # Normalize activations
        
        self.fc2 = nn.Linear(2048, 2048)
        self.ln2 = nn.LayerNorm(2048)
        
        self.fc3 = nn.Linear(2048, 2048)
        self.ln3 = nn.LayerNorm(2048)
        
        self.fc4 = nn.Linear(2048, 1024)
        self.ln4 = nn.LayerNorm(1024)
        
        self.fc5 = nn.Linear(1024, self.output_dim)
        
        # Input projection for skip connection
        self.input_proj = nn.Linear(latent_dim, 2048) if latent_dim != 2048 else nn.Identity()
        
        # Point identity: encode initial positions to give each point its identity
        self.point_pos_encoder = nn.Linear(nfeats, latent_dim)
        # Projection to combine point-conditioned z back to latent_dim
        self.point_condition_proj = nn.Linear(2 * njoints * latent_dim, latent_dim)
        
    def forward(self, batch, use_text_emb=False):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
        bs, nframes = mask.shape
        # Support both single vector and token sequences: reduce to [bs, latent]
        if z.dim() == 3:
            if z.shape[0] == bs:
                # [bs, seq, dim] → take CLS token
                z = z[:, 0, :]
            elif z.shape[1] == bs:
                # [seq, bs, dim] → take CLS token across seq dim
                z = z[0, :, :]
            elif z.shape[0] == 1:
                # [1, bs, dim]
                z = z.squeeze(0)
        if use_text_emb:
            z = batch["clip_text_emb"]
        
        njoints, nfeats = self.njoints, self.nfeats
        
        # CRITICAL: Condition on point identities from initial_grid
        # MLP outputs all points at once, so we need to inject point identity into the latent
        if "initial_grid" in batch:
            initial_grid = batch["initial_grid"]  # [bs, njoints, nfeats]
            # Encode initial positions: [bs, njoints, nfeats] -> [bs, njoints, latent_dim]
            pos_encoded = self.point_pos_encoder(initial_grid)
            # Flatten point encodings and concatenate with z (gives MLP point identity info)
            pos_flat = pos_encoded.reshape(bs, -1)  # [bs, njoints*latent_dim]
            # Expand z to match: repeat z for each point's encoding
            z_expanded = z.unsqueeze(1).expand(-1, njoints, -1).reshape(bs, -1)  # [bs, njoints*latent_dim]
            # Concatenate: [bs, (njoints*latent_dim)*2]
            z_combined = torch.cat([z_expanded, pos_flat], dim=1)  # [bs, 2*njoints*latent_dim]
            # Project back to original latent_dim for MLP input
            z = self.point_condition_proj(z_combined)  # [bs, latent_dim]
        
        # MLP with LayerNorm + Skip Connections (ResNet-style)
        # This prevents collapse with normalized CLIP inputs!
        act = F.gelu if self.activation == 'gelu' else F.relu
        
        # Project input for skip connection
        z_proj = self.input_proj(z)  # [bs, 2048]
        
        # Layer 1 with skip connection
        x = self.fc1(z)
        x = self.ln1(x)
        x = act(x)
        x = x + z_proj  # Skip connection! Preserves input information
        
        # Layer 2 with skip connection  
        identity = x
        x = self.fc2(x)
        x = self.ln2(x)
        x = act(x)
        x = x + identity  # Skip connection
        
        # Layer 3 with skip connection
        identity = x
        x = self.fc3(x)
        x = self.ln3(x)
        x = act(x)
        x = x + identity  # Skip connection
        
        # Layer 4 (no skip - dimension changes)
        x = self.fc4(x)
        x = self.ln4(x)
        x = act(x)
        
        # Output layer (no activation)
        # CRITICAL: Generate DISPLACEMENTS, not absolute positions!
        displacements = self.fc5(x)  # [bs, njoints*nfeats*nframes]
        
        # Reshape to [bs, njoints, nfeats, max_frames]
        displacements = displacements.view(bs, njoints, nfeats, self.num_frames)
        
        # Crop to actual sequence length
        if nframes < self.num_frames:
            displacements = displacements[:, :, :, :nframes]
        
        # CRITICAL FIX: Add displacements to initial grid positions
        # If initial_grid is provided, use it as the starting point
        if "initial_grid" in batch:
            initial_grid = batch["initial_grid"]  # [bs, njoints, 2] - frame 0 positions
            
            # FORCE frame 0 displacement to be zero (no movement at start)
            # This ensures output[:,:,:,0] = initial_grid exactly!
            displacements[:, :, :, 0] = 0
            
            # Expand to all frames: [bs, njoints, 2, 1] -> [bs, njoints, 2, nframes]
            initial_grid_expanded = initial_grid.unsqueeze(-1).expand(-1, -1, -1, nframes)
            # Add displacements to get final positions
            output = initial_grid_expanded + displacements
        else:
            # Fallback: Use displacements as absolute positions (old behavior)
            output = displacements
        
        # Apply mask (zero for padded areas)
        output = output * mask.unsqueeze(1).unsqueeze(1)
        
        if use_text_emb:
            batch["txt_output"] = output
        else:
            batch["output"] = output
        return batch


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, 
                 pose_rep, glob, glob_rot, latent_dim=256, ff_size=1024, num_layers=4, 
                 num_heads=4, dropout=0.1, activation="linear", ablation=None, **kargs):
        super().__init__()
        
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        
        # Point identity encoding (for initial_grid conditioning)
        self.point_encoder = nn.Linear(nfeats, latent_dim)
        
        # Text embedding projection (if needed)
        self.clip_proj = nn.Linear(512, latent_dim)
        
        # Time encoding (for timequeries)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        
        # TransformerDecoder: Cross-attention (like original MotionCLIP)
        # timequeries attend to z (memory) via cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=False
        )
        self.seqTransDecoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output: Generate displacement per point per frame
        # CRITICAL FIX: Use frame index directly instead of collapsing frame embeddings
        # The TransformerDecoder collapses all frames to identical embeddings, so we bypass it
        # Each point's displacement is conditioned on: frame_index + point_initial_position
        # frame_index_emb: [latent_dim] - learned embedding of frame index
        # point_emb: [latent_dim] - embedding of initial grid position
        self.frame_index_embedding = nn.Embedding(num_frames, latent_dim)  # Learn frame-specific features
        self.finallayer = nn.Linear(3 * latent_dim, nfeats)  # [frame_emb+z_context + point_emb + prev_frame_emb] -> [nfeats]
        
    def forward(self, batch, use_text_emb=False):
        z = batch["clip_text_emb"] if use_text_emb else batch["z"]
        mask = batch["mask"]
        initial_grid = batch["initial_grid"]  # [bs, njoints, 2]
        
        bs, nframes = mask.shape
        njoints = self.njoints
        nfeats = self.nfeats
        
        # Format z
        if z.dim() == 3:
            z = z[0] if z.shape[0] == 1 else z[:, 0]
        
        # Project z to latent_dim if needed
        if z.shape[1] == self.latent_dim:
            z_proj = z
        elif z.shape[1] == 512:
            z_proj = self.clip_proj(z)  # [bs, latent_dim]
        else:
            z_proj = z
        
        # CRITICAL FIX: Use frame index embeddings instead of collapsing TransformerDecoder
        # The TransformerDecoder was collapsing all frames to identical embeddings,
        # causing all points to get similar displacements (simple parallel lines)
        # Instead, use learned frame index embeddings that preserve temporal diversity
        
        # Embed initial grid for point conditioning
        initial_emb = self.point_encoder(initial_grid)  # [bs, njoints, latent_dim]
        
        # Create frame indices [0, 1, 2, ..., nframes-1]
        frame_indices = torch.arange(nframes, device=z.device).unsqueeze(1).expand(-1, bs)  # [nframes, bs]
        
        # Get frame embeddings from learned embeddings (bypass collapsing decoder)
        frame_embeddings = self.frame_index_embedding(frame_indices)  # [nframes, bs, latent_dim]
        
        # Combine frame embeddings with motion context (temporal context)
        # This creates a combined temporal context that varies per frame
        z_context = z_proj.unsqueeze(0).expand(nframes, -1, -1)  # [nframes, bs, latent_dim]
        frame_z_combined = frame_embeddings + z_context  # [nframes, bs, latent_dim] - combined temporal context
        
        # CRITICAL FIX: Autoregressive generation during inference to fix train-test mismatch
        # During training: parallel generation with teacher forcing (GT previous frames)
        # During inference: autoregressive generation (use generated previous frames)
        # This prevents trajectory flattening by ensuring model sees proper temporal context at inference
        
        if self.training and 'x_original' in batch:
            # TRAINING: Parallel generation with teacher forcing (use GT previous frames)
            # This is efficient and allows model to learn from correct previous frames
            # DEBUG: Verify training path is being used
            # if hasattr(self, '_debug_training_path') or True:  # Always print for debugging
                # print(f"🔵 TRAINING PATH: Using teacher forcing with direct addition (prev_pos + displacement)")
            gt_frames = batch['x_original']  # [bs, njoints, nfeats, nframes]
            prev_positions = torch.zeros_like(gt_frames)
            prev_positions[:, :, :, 0] = initial_grid  # Frame 0 uses initial grid
            prev_positions[:, :, :, 1:] = gt_frames[:, :, :, :-1]  # Frame t uses GT frame t-1
            prev_frame_emb = self.point_encoder(prev_positions.permute(3, 0, 1, 2))  # [nframes, bs, njoints, latent_dim]
            prev_frame_emb_exp = prev_frame_emb  # [nframes, bs, njoints, latent_dim]
            
            # Generate all points for each frame in parallel
            frame_z_exp = frame_z_combined.unsqueeze(2).expand(-1, -1, njoints, -1)  # [nframes, bs, njoints, latent_dim]
            initial_emb_exp = initial_emb.unsqueeze(0).expand(nframes, -1, -1, -1)  # [nframes, bs, njoints, latent_dim]
            frame_point_emb = torch.cat([frame_z_exp, initial_emb_exp, prev_frame_emb_exp], dim=-1)  # [nframes, bs, njoints, 3*latent_dim]
            
            # Generate displacements for each point at each frame
            frame_point_flat = frame_point_emb.reshape(nframes * bs * njoints, 3 * self.latent_dim)
            displacements_flat = self.finallayer(frame_point_flat)  # [nframes*bs*njoints, nfeats]
            frame_outputs = displacements_flat.reshape(nframes, bs, njoints, nfeats)  # [nframes, bs, njoints, nfeats]
            displacements = frame_outputs.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
            
            # CRITICAL FIX: In training, displacements are relative to prev_positions (GT frames)
            # So output should be: prev_positions + displacements (direct addition, not cumsum)
            # This ensures model learns to generate correct displacements that work when accumulated
            displacements[:, :, :, 0] = 0.0  # Frame 0 has no displacement
            output = prev_positions + displacements  # Direct addition: output[t] = prev_positions[t] + displacements[t]
            # This equals: output[0] = initial_grid, output[t] = gt_frames[t-1] + displacement[t] = gt_frames[t]
            
            batch["txt_output" if use_text_emb else "output"] = output
            return batch
            
        else:
            # INFERENCE: Choose between autoregressive (default) and static-initial (old) modes
            # Toggle using optional flag on batch for ablation/testing:
            #   batch['inference_prev_mode'] in {'autoregressive', 'static'}
            # DEBUG: Verify inference path is being used
            # if hasattr(self, '_debug_inference_path') or True:  # Always print for debugging
                # print(f"🔴 INFERENCE PATH: self.training={self.training}, 'x_original' in batch={'x_original' in batch}")
            inference_mode = batch.get('inference_prev_mode', 'autoregressive')

            displacements = torch.zeros(bs, njoints, nfeats, nframes, device=z.device)

            if inference_mode == 'static':
                # OLD BEHAVIOR (for testing): use static initial_grid as previous frame for all t
                prev_positions = initial_grid.unsqueeze(-1).expand(-1, -1, -1, nframes)  # [bs, njoints, nfeats, nframes]
                prev_frame_emb = self.point_encoder(prev_positions.permute(3, 0, 1, 2))  # [nframes, bs, njoints, latent_dim]

                # Parallel generation with static previous frame
                frame_z_exp = frame_z_combined.unsqueeze(2).expand(-1, -1, njoints, -1)
                initial_emb_exp = initial_emb.unsqueeze(0).expand(nframes, -1, -1, -1)
                frame_point_emb = torch.cat([frame_z_exp, initial_emb_exp, prev_frame_emb], dim=-1)
                frame_point_flat = frame_point_emb.reshape(nframes * bs * njoints, 3 * self.latent_dim)
                displacements_flat = self.finallayer(frame_point_flat)
                frame_outputs = displacements_flat.reshape(nframes, bs, njoints, nfeats)
                displacements = frame_outputs.permute(1, 2, 3, 0)

            else:
                # DEFAULT: Autoregressive generation (use generated previous frames)
                current_positions = initial_grid.clone()  # [bs, njoints, nfeats]
                for t in range(nframes):
                    # Encode current positions (previous frame output for frame t)
                    current_pos_emb = self.point_encoder(current_positions)
                    # Get frame t embedding
                    frame_t_emb = frame_z_combined[t:t+1, :, :]
                    frame_t_emb_exp = frame_t_emb.unsqueeze(2).expand(-1, -1, njoints, -1)
                    initial_emb_exp_t = initial_emb.unsqueeze(0)
                    current_pos_emb_exp = current_pos_emb.unsqueeze(0)
                    # Combine: frame context + initial grid + previous frame position
                    frame_point_emb_t = torch.cat([frame_t_emb_exp, initial_emb_exp_t, current_pos_emb_exp], dim=-1)
                    # Generate displacement for frame t
                    frame_point_flat_t = frame_point_emb_t.reshape(bs * njoints, 3 * self.latent_dim)
                    displacements_flat_t = self.finallayer(frame_point_flat_t)
                    displacements_t = displacements_flat_t.reshape(bs, njoints, nfeats)
                    # Store displacement
                    if t == 0:
                        displacements[:, :, :, 0] = 0.0
                    else:
                        displacements[:, :, :, t] = displacements_t
                        # Update current positions for next frame (autoregressive)
                        current_positions = current_positions + displacements_t
            
            # INFERENCE: Accumulate displacements (frame_t = frame_{t-1} + displacement_t)
            # In autoregressive mode, displacements are already relative to previous generated frame
            # So we accumulate them to get final positions
            displacements[:, :, :, 0] = 0.0  # Frame 0 has no displacement
            displacements_accumulated = torch.cumsum(displacements, dim=3)  # [bs, njoints, nfeats, nframes]
            
            # Add accumulated displacements to initial grid
            initial_expanded = initial_grid.unsqueeze(-1).expand(-1, -1, -1, nframes)  # [bs, njoints, nfeats, nframes]
            output = initial_expanded + displacements_accumulated
            
            batch["txt_output" if use_text_emb else "output"] = output
            return batch