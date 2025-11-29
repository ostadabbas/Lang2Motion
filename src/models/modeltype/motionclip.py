import numpy as np
import torch
import torch.nn as nn
import clip
from src.models.architectures import *
from src.models.tools.losses import get_loss_function
import torch.nn.functional as F
from src.utils.render_points import render_point_tracks_batch, preprocess_for_clip

loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss()

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
from tqdm import tqdm


class MOTIONCLIP(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, clip_lambdas={}, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.outputxyz = outputxyz

        self.lambdas = lambdas
        self.clip_lambdas = clip_lambdas
        self.lambda_text_recon = kwargs.get('lambda_text_recon', 0.0)  # Dual forward pass weight

        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans

        self.clip_model = kwargs['clip_model']
        self.clip_training = kwargs.get('clip_training', False)
        if self.clip_training and self.clip_model:
            self.clip_model.training = True
        else:
            if self.clip_model:
                assert self.clip_model.training == False  # make sure clip is frozen

        self.losses = list(self.lambdas) + ["mixed"]

        # Disable SMPL functionality for PointCLIP
        self.rotation2xyz = None  # Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}

    def rot2xyz(self, x, mask, get_rotations_back=False, **kwargs):
        # For PointCLIP, we work directly with point coordinates
        # No rotation to xyz conversion needed
        if self.rotation2xyz is None:
            # Return input as-is for point trajectories
            return x
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, get_rotations_back=get_rotations_back, **kargs)

    def compute_loss(self, batch):

        # DUAL FORWARD PASS: Train decoder to decode from CLIP text features
        # Add text_output to batch BEFORE loss computation so text_rc loss function can use it
        if 'clip_text' in batch and self.training:
            try:
                # Get CLIP text features - USE FULL 77-TOKEN SEQUENCE
                texts = clip.tokenize(batch['clip_text']).to(self.device)
                with torch.no_grad():
                    # Extract full token embeddings (before final projection)
                    # This gives us 77 contextualized tokens instead of just 1 CLS token
                    x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)  # [bs, 77, 512]
                    x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
                    x = x.permute(1, 0, 2)  # [77, bs, 512] for transformer
                    x = self.clip_model.transformer(x)
                    x = x.permute(1, 0, 2)  # [bs, 77, 512]
                    x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
                    
                    # x now contains 77 contextualized token embeddings
                    clip_text_features = x.float()  # [bs, 77, 512] instead of [bs, 512]
                    
                    # Also extract CLS token the same way as contrastive loss (for alignment check)
                    # This matches what encode_text() returns
                    clip_text_features_cls = self.clip_model.encode_text(texts).float()  # [bs, 512] - CLS token
                    clip_text_features_cls = clip_text_features_cls / clip_text_features_cls.norm(dim=-1, keepdim=True)
                    
                    # Store 77-token sequence in batch for use in contrastive loss (ensures consistency)
                    batch['clip_text_features_77'] = clip_text_features
                    batch['clip_text_features_cls'] = clip_text_features_cls  # For alignment check
                    
                    # Normalize each token separately (keep them as unit vectors)
                    clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)
                    
                

                
                # Check if mu_clip exists (from encoder forward pass)
                if 'mu_clip' in batch:
                    mu_clip = batch['mu_clip']
                    
                    
                    # For 3D tensor [bs, 77, 512], compute stats over all tokens
                    # Mean/std per dimension across all tokens
                    mean_all_tokens = clip_text_features.mean(dim=[0, 1])  # Average over batch and tokens
                    std_all_tokens = clip_text_features.std(dim=[0, 1])
                    # Norm per token (each token should be normalized)
                    norm_per_token = clip_text_features.norm(dim=-1)  # [bs, 77]
                    
                    # Compute cosine similarity between matching pairs
                    # Use the CLS token extracted the same way as contrastive loss (for consistency)
                    if 'clip_text_features_cls' in batch:
                        clip_text_cls = batch['clip_text_features_cls']  # [bs, 512] - same extraction as contrastive loss
                        if mu_clip.shape[0] == clip_text_cls.shape[0]:
                            # Compare to CLS token (matches what contrastive loss uses)
                            cos_sim_cls = (mu_clip * clip_text_cls).sum(dim=1)  # [bs]
                            
                            # Also compare to mean of all 77 tokens (alternative)
                            mean_tokens = clip_text_features.mean(dim=1)  # [bs, 512] - average over tokens
                            cos_sim_mean = (mu_clip * mean_tokens).sum(dim=1)  # [bs]
                            
                            
                            # Use CLS token for warnings (standard CLIP representation, matches contrastive loss)
                            cos_sim = cos_sim_cls
                    elif mu_clip.shape[0] == clip_text_features.shape[0]:
                        # Fallback: Compare to first token (not ideal, but works)
                        cls_token = clip_text_features[:, 0, :]  # [bs, 512] - first token
                        cos_sim_cls = (mu_clip * cls_token).sum(dim=1)  # [bs]
                        cos_sim = cos_sim_cls
                        if cos_sim.mean().item() < 0.5:
                            print(f"  🔴 WARNING: Low cosine similarity (< 0.5)")
                            print(f"     mu_clip and CLS token occupy different CLIP space regions!")
                            print(f"     This may cause poor text-to-motion generation despite low loss.")
                        elif cos_sim.mean().item() < 0.7:
                            print(f"  ⚠️  MODERATE: Cosine similarity between 0.5-0.7")
                            print(f"     Some alignment, but could be better.")
                        else:
                            print(f"  ✅ GOOD: High cosine similarity (> 0.7)")
                            print(f"     mu_clip and CLS token are well aligned.")
                    else:
                        print(f"  ⚠️  Batch size mismatch - cannot compute pairwise similarity")
                else:
                    print(f"  ⚠️  mu_clip not found in batch - encoder may not have run yet")
                
                
                # NOISE AUGMENTATION REMOVED
                # (Previously added noise for stochastic sampling, but removed per request)
                
                # Create batch for text-to-motion forward pass
                # Decoder uses single CLS token
                clip_text_cls = clip_text_features_cls if 'clip_text_features_cls' in batch else clip_text_features[:, 0, :]
                batch_text = {
                    'clip_text_emb': clip_text_cls,  # [bs, 512] - CLS token for decoder
                    'y': batch['y'],
                    'mask': batch['mask'],
                    'lengths': batch['lengths'],
                    'initial_grid': batch['initial_grid']  # CRITICAL: Use same initial grid!
                }
                
                # Decode from CLIP text features (use_text_emb=True)
                text_output = self.decoder(batch_text, use_text_emb=True)
                
                # Add to batch so loss functions can use it
                # Decoder returns 'txt_output' when use_text_emb=True
                batch['txt_output'] = text_output['txt_output']
                # Also keep 'text_output' for text_recon_loss compatibility
                batch['text_output'] = text_output['txt_output']
                
            except Exception as e:
                # CRITICAL: Don't silently fail - this breaks text_recon loss
                # If text-to-motion generation fails, we should know about it
                import traceback
                error_msg = f"❌ text_output computation failed: {e}\n{traceback.format_exc()}"
                print(error_msg)
                
                # If text_recon loss is enabled, we need text_output - fail loudly
                if self.lambda_text_recon > 0:
                    raise RuntimeError(
                        f"Text-to-motion generation failed but lambda_text_recon={self.lambda_text_recon} > 0. "
                        f"Original error: {e}"
                    ) from e
                
                # Otherwise, don't set text_output (loss functions check for its absence)
                # Loss functions will return 0.0 if 'text_output' not in batch

        # compute all losses other than clip (including text_rc now that text_output is in batch)
        mixed_loss = 0.
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            
            # CRITICAL FIX: Apply vel, rc, range, and spatial losses to txt_output if available
            # This ensures trajectory shape AND granularity are enforced for text-to-motion generation
            # Without this, these losses only apply to reconstruction, not text-to-motion
            if ltype in ['vel', 'rc', 'range', 'spatial'] and 'txt_output' in batch:
                # Apply to BOTH reconstruction AND text-to-motion
                # CRITICAL FIX: Don't average - apply both separately with full weight
                # Averaging was causing model to converge to simpler "middle ground" trajectories
                loss_recon = loss_function(self, batch, use_txt_output=False)
                loss_text = loss_function(self, batch, use_txt_output=True)
                # Sum both losses (not average) - this preserves detail by not forcing middle ground
                loss = loss_recon + loss_text
                # Track both losses separately for monitoring
                losses[f'{ltype}_recon'] = loss_recon.item()
                losses[f'{ltype}_text'] = loss_text.item()
            else:
                loss = loss_function(self, batch)
            
            mixed_loss += loss * lam
            losses[ltype] = loss.item()

        # compute clip losses
        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)

        # mix and add clip losses
        mixed_loss_with_clip = mixed_loss + mixed_clip_loss
        losses.update(clip_losses)
        losses["mixed_without_clip"] = mixed_loss.item()
        losses["mixed_clip_only"] = mixed_clip_loss if isinstance(mixed_clip_loss, (float, int)) else mixed_clip_loss.item()
        # text_recon_only will be computed by text_rc loss function
        losses["mixed_with_clip"] = mixed_loss_with_clip if isinstance(mixed_loss_with_clip, float) else mixed_loss_with_clip.item()

        return mixed_loss_with_clip, losses

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0
        clip_losses = {}
        
        if self.clip_training is None:
            return torch.tensor(0.0, device=self.device), {}
        
        # NOTE: Original MotionCLIP uses COSINE loss, not CE contrastive loss
        # The first path (for loop) was using CE, but MotionCLIP uses cosine
        # So we skip the first path and only use the second path (clip_lambdas) with cosine loss
        # This matches: --clip_text_losses cosine --clip_image_losses cosine from MotionCLIP repo
        # Reference: https://github.com/GuyTevet/MotionCLIP/tree/main
        
        # Use clip_lambdas path (which uses cosine loss matching MotionCLIP)
        for d in self.clip_lambdas.keys():
            if len(self.clip_lambdas[d].keys()) == 0:
                continue
            
            with torch.no_grad():
                if d == 'image':
                    features = self.clip_model.encode_image(
                        batch['clip_images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    # Use pooled CLIP features (mean of 77 tokens) for cosine loss
                    if 'clip_pooled' in batch:
                        features = batch['clip_pooled']
                    else:
                        texts = clip.tokenize(batch['clip_text']).to(self.device)
                        features = self.clip_model.encode_text(texts).float()
                elif d == 'rendered_image':
                    # Render GROUND TRUTH motion (FIRST FRAME ONLY) - like MotionCLIP!
                    # Use INPUT motion, not reconstructed output!
                    if 'x_xyz' in batch:
                        ground_truth_motion = batch['x_xyz']  # Ground truth!
                    elif 'x' in batch:
                        ground_truth_motion = batch['x']  # Ground truth!
                    else:
                        continue  # Skip if no input
                    
                    # Render ground truth to images
                    rendered_images = render_point_tracks_batch(
                        ground_truth_motion, 
                        image_size=224,
                        show_trails=False  # First frame only!
                    )
                    rendered_images_norm = preprocess_for_clip(rendered_images).to(self.device)
                    features = self.clip_model.encode_image(rendered_images_norm).float()
                elif d == 'trajectory_overlays':
                    # NEW: Trajectory overlays on actual video frames with temporal pooling
                    if 'trajectory_overlays' not in batch:
                        continue  # Skip if not available
                    
                    overlay_frames = batch['trajectory_overlays']  # [bs, T, 3, 224, 224]
                    bs, nframes = overlay_frames.shape[:2]
                    
                    # Flatten temporal dimension for batch processing
                    frames_flat = overlay_frames.view(-1, 3, 224, 224)  # [bs*T, 3, 224, 224]
                    
                    # Process all frames through CLIP (temporal pooling approach)
                    frame_features = self.clip_model.encode_image(frames_flat).float()  # [bs*T, 512]
                    frame_features_norm = frame_features / frame_features.norm(dim=-1, keepdim=True)
                    
                    # Reshape and average across time (temporal pooling)
                    frame_features_video = frame_features_norm.view(bs, nframes, 512)  # [bs, T, 512]
                    features = frame_features_video.mean(dim=1)  # [bs, 512] - temporal average
                else:
                    raise ValueError(f'Invalid clip domain [{d}]')

            # Normalize CLIP features
            features_norm = features / features.norm(dim=-1, keepdim=True)
            
            # Use normalized features (mu_clip is mu normalized, no projection needed)
            # latent_dim (512) == CLIP dim, so mu is already in CLIP space
            motion_features_norm = batch["mu_clip"]  # Already normalized in encoder

            if 'ce' in self.clip_lambdas[d].keys():
                logit_scale = self.clip_model.logit_scale.exp()
                logits_per_motion = logit_scale * motion_features_norm @ features_norm.t()
                logits_per_d = logits_per_motion.t()

                batch_size = batch['x'].shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss * self.clip_lambdas[d]['ce']

            if 'mse' in self.clip_lambdas[d].keys():
                # batch["z"] is now [bs, 77, 512], but features is [bs, 512]
                # Compare features to CLS token (first token) or mean of tokens
                z_for_loss = batch["z"]
                if len(z_for_loss.shape) == 3:
                    # Use CLS token (position 0) for MSE loss comparison
                    z_for_loss = z_for_loss[:, 0, :]  # [bs, 512] - CLS token
                mse_clip_loss = loss_mse(features, z_for_loss)
                clip_losses[f'{d}_mse'] = mse_clip_loss.item()
                mixed_clip_loss += mse_clip_loss * self.clip_lambdas[d]['mse']

            if 'cosine' in self.clip_lambdas[d].keys():
                cos = cosine_sim(features_norm, motion_features_norm)
                cosine_loss = (1 - cos).mean()
                clip_losses[f'{d}_cosine'] = cosine_loss.item()
                mixed_clip_loss += cosine_loss * self.clip_lambdas[d]['cosine']

        return mixed_clip_loss, clip_losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate_one(self, cls, duration, fact=1, xyz=False):
        y = torch.tensor([cls], dtype=int, device=self.device)[None]
        lengths = torch.tensor([duration], dtype=int, device=self.device)
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(self.latent_dim, device=self.device)[None]  # [1, 512]

        # Decoder ignores memory; pass single-token placeholder
        z_mem = (fact * z).unsqueeze(0)  # [1, 1, 512]

        batch = {"z": z_mem, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if not xyz:
            return batch["output"][0]

        output_xyz = self.rot2xyz(batch["output"], batch["mask"])

        return output_xyz[0]

    def generate(self, classes, durations, nspa=1,
                 # noise_same_action="random", noise_diff_action="random",
                 # fact=1,
                 is_amass=False, is_clip_features=False,
                 # input_type="motion",
                 textual_labels=None):
        clip_dim = self.clip_model.ln_final.normalized_shape[0]
        if is_clip_features:
            # classes can be:
            # - [nspa, nats, 512] - CLS tokens only (old)
            # - [nspa, 77, 512] - full token sequence (NEW!)
            assert len(classes.shape) == 3
            assert classes.shape[-1] == clip_dim
            
            # Check if this is full token sequence (77 tokens) or CLS only (nats classes)
            if classes.shape[1] == 77:
                # Full token sequence! Keep as is for decoder
                clip_features = classes  # [nspa, 77, 512]
                nspa = classes.shape[0]
            else:
                # Old format: CLS tokens only
                clip_features = classes.reshape([-1, clip_dim])
                nspa, nats = classes.shape[:2]
            
            # y = torch.zeros(y_action_names.shape, dtype=int)
            y = clip_features if len(clip_features.shape) == 2 else clip_features[:, 0, :]  # Use first token for y
            if textual_labels is not None:
                y = np.array(textual_labels).reshape([-1])

        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
        else:
            lengths = durations.to(self.device).reshape(clip_features.shape[0])

        mask = self.lengths_to_mask(lengths)

        # Decoder uses self-attention over queries; memory is ignored. Provide CLS as a placeholder token.
        z_mem = clip_features[:, 0, :] if classes.shape[1] == 77 else clip_features  # [nspa, 512]
        z_mem = z_mem.unsqueeze(0)  # [1, nspa, 512]
        batch = {"z": z_mem,
                 "y": y,
                 "mask": mask, "lengths": lengths}

        if not is_clip_features:
            batch['y'] = y

        batch = self.decoder(batch)

        if is_amass:  # lose global orientation for amass dataset
            batch['output'][:, 0] = torch.tensor([1, 0, 0, 0, -1, 0]).unsqueeze(0).unsqueeze(2)

        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]

        return batch

    def generate_from_embedding(self, classes, durations, nspa=1, is_amass=False, classes_gaussians=None):

        if nspa is None:
            nspa = 1
        nats = len(classes)

        y = classes.to(self.device).repeat(nspa)  # (view(nspa, nats))
        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
        else:
            lengths = durations.to(self.device).reshape(y.shape)
        mask = self.lengths_to_mask(lengths)
        classes_np = classes.cpu().detach().numpy()

        motion_samples_ = np.zeros((classes_np.shape[0], 512), dtype='float32')
        for class_label in tqdm(np.unique(classes_np), total=len(np.unique(classes_np))):
            class_mask = np.where(classes_np == class_label)[0]
            sample_mu = classes_gaussians[class_label]['mu']
            sample_var = classes_gaussians[class_label]['var']

            sample = np.random.multivariate_normal(sample_mu, sample_var, size=len(class_mask))
            motion_samples_[class_mask, :] = sample

        zz = torch.from_numpy(motion_samples_).to(self.device)  # [nspa, 512]

        # Decoder ignores memory; pass single-token placeholder
        z_mem = zz.unsqueeze(0)  # [1, nspa, 512]

        batch = {"z": z_mem,
                 "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if is_amass:  # lose global orientation for amass dataset
            batch['output'][:, 0] = torch.tensor([1, 0, 0, 0, -1, 0]).unsqueeze(0).unsqueeze(2)

        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def forward(self, batch):
        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        
        # Augment input with velocity AND acceleration features for better motion encoding
        # Input: [bs, njoints, nfeats, nframes]
        x = batch["x"]
        
        
        # Compute velocities: v(t) = x(t) - x(t-1)
        velocities = torch.zeros_like(x)
        velocities[:, :, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]  # First frame has zero velocity
        
        # Compute accelerations: a(t) = v(t) - v(t-1)
        accelerations = torch.zeros_like(x)
        accelerations[:, :, :, 2:] = velocities[:, :, :, 2:] - velocities[:, :, :, 1:-1]  # First two frames have zero acceleration
        
        # Concatenate positions + velocities + accelerations along feature dimension
        # Original: [bs, njoints, 2, nframes] (x, y)
        # Augmented: [bs, njoints, 6, nframes] (x, y, vx, vy, ax, ay)
        x_with_motion = torch.cat([x, velocities, accelerations], dim=2)
        
        
        # Replace input with augmented version
        batch["x_original"] = batch["x"]  # Keep original for losses
        batch["x"] = x_with_motion
        
        # encode - encoder outputs single latent mu [bs, 512]
        batch.update(self.encoder(batch))
        
        # Restore original input (decoder outputs positions only)
        batch["x"] = batch["x_original"]
        
        # CRITICAL FIX: Use CLIP-projected features for decoder (NOT raw mu)
        # This ensures decoder trains in the SAME space it will generate in
        # Training: decoder sees mu_clip (normalized CLIP space from encoder)
        # Inference: decoder sees clip_text_emb (normalized CLIP space from text)
        # Without this fix, decoder trains on mu (raw latent) but generates from CLIP → space mismatch!
        
        # Use single-token memory for reconstruction (match text generation format)
        mu_clip = batch["mu_clip"]  # [bs, 512]
        batch["z"] = mu_clip.unsqueeze(0)  # [1, bs, 512]
        
        
        # CRITICAL FIX: Provide initial grid positions to decoder
        # Extract frame 0 positions as the initial grid: [bs, njoints, 2]
        # Use x_original (not x) to ensure correct shape and values
        if 'x_original' not in batch:
            raise ValueError("batch['x_original'] is required for initial_grid extraction")
        
        batch["initial_grid"] = batch["x_original"][:, :, :, 0]  # [bs, njoints, 2]
        
        # Verify shape
        expected_shape = (batch['x_original'].shape[0], batch['x_original'].shape[1], batch['x_original'].shape[2])
        actual_shape = batch['initial_grid'].shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"Initial grid shape mismatch! Expected {expected_shape} ([bs, njoints, 2]), "
                f"got {actual_shape}. This will cause decoder to fail."
            )
        
        # decode from CLIP-space latent
        # Decoder will generate displacements and add them to initial_grid
        batch.update(self.decoder(batch))

        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        
        return batch
