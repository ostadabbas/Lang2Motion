import torch
import torch.nn.functional as F
from .hessian_penalty import hessian_penalty
from .mmd import compute_mmd


def compute_rc_loss(model, batch, use_txt_output=False):
    """
    Simple reconstruction loss: Match GT point-by-point, frame-by-frame.
    
    No averaging, no complexity - just direct L1 matching.
    Every point, every frame must match.
    CRITICAL: Normalized by batch size to ensure consistent loss scale across different batch sizes.
    """
    x = batch.get("x_original", batch["x"])      # [bs, N, 2, T]
    output = batch["txt_output"] if use_txt_output else batch["output"]
    mask = batch["mask"]                         # [bs, T]
    batch_size = x.shape[0]

    # Simple: compute error, mask, sum everything
    err = torch.abs(output - x)  # [bs, npoints, nfeats, nframes]
    mask_exp = mask.unsqueeze(1).unsqueeze(1).expand_as(err)  # [bs, npoints, nfeats, nframes]
    
    # Sum all errors, then normalize by batch size (ensures consistent loss scale)
    total_loss = (err * mask_exp).sum() / batch_size
    
    return total_loss


def compute_rcxyz_loss(model, batch, use_txt_output=False):
    """NOT USED - keeping for compatibility"""
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    mask = batch["mask"]
    if use_txt_output:
        output = batch["txt_output_xyz"]
    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]

    loss = F.l1_loss(gtmasked, outmasked, reduction='mean')
    return loss


def compute_vel_loss(model, batch, use_txt_output=False):
    """
    Simple velocity loss: Match GT velocities point-by-point, frame-by-frame.
    
    No averaging, no complexity - just direct L1 matching.
    Every velocity must match.
    CRITICAL: Normalized by batch size to ensure consistent loss scale across different batch sizes.
    """
    x = batch.get("x_original", batch["x"])
    output = batch["output"]
    if use_txt_output:
        output = batch["txt_output"]
    batch_size = x.shape[0]

    gtvel = x[..., 1:] - x[..., :-1]  # [bs, npoints, nfeats, nframes-1]
    outvel = output[..., 1:] - output[..., :-1]
    mask = batch["mask"][..., 1:]  # [bs, nframes-1]
    
    # Simple: compute error, mask, sum everything
    err = torch.abs(outvel - gtvel)  # [bs, npoints, nfeats, nframes-1]
    mask_exp = mask.unsqueeze(1).unsqueeze(1).expand_as(err)  # [bs, npoints, nfeats, nframes-1]
    
    # Sum all errors, then normalize by batch size (ensures consistent loss scale)
    total_loss = (err * mask_exp).sum() / batch_size
    
    return total_loss



def compute_velxyz_loss(model, batch, use_txt_output=False):
    """NOT USED - keeping for compatibility"""
    x = batch["x_xyz"]
    output = batch["output_xyz"]
    if use_txt_output:
        output = batch["txt_output_xyz"]
    gtvel = (x[..., 1:] - x[..., :-1])
    outputvel = (output[..., 1:] - output[..., :-1])

    mask = batch["mask"][..., 1:]

    gtvelmasked = gtvel.permute(0, 3, 1, 2)[mask]
    outvelmasked = outputvel.permute(0, 3, 1, 2)[mask]

    loss = F.mse_loss(gtvelmasked, outvelmasked, reduction='mean')
    return loss


def compute_hp_loss(model, batch):
    """NOT USED - keeping for compatibility"""
    loss = hessian_penalty(model.return_latent, batch, seed=torch.random.seed())
    return loss


def compute_mmd_loss(model, batch):
    """NOT USED - keeping for compatibility"""
    z = batch["z"]
    true_samples = torch.randn(z.shape, requires_grad=False, device=model.device)
    loss = compute_mmd(true_samples, z)
    return loss


def compute_range_loss(model, batch, use_txt_output=False):
    """
    Simple acceleration loss: Match GT accelerations point-by-point, frame-by-frame.
    
    Acceleration = how velocity changes. This preserves frame-by-frame detail.
    CRITICAL: Normalized by batch size to ensure consistent loss scale across different batch sizes.
    """
    x = batch.get("x_original", batch["x"])
    output = batch["output"]
    if use_txt_output:
        output = batch["txt_output"]
    batch_size = x.shape[0]
    
    # Compute velocities
    gt_vel = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B, 36, 2, 29]
    out_vel = output[:, :, :, 1:] - output[:, :, :, :-1]
    
    # Compute accelerations (change in velocity)
    gt_accel = gt_vel[:, :, :, 1:] - gt_vel[:, :, :, :-1]  # [B, 36, 2, 28]
    out_accel = out_vel[:, :, :, 1:] - out_vel[:, :, :, :-1]
    
    mask = batch["mask"][:, 2:]  # [B, 28]
    
    # Simple: compute error, mask, sum everything
    err = torch.abs(out_accel - gt_accel)  # [B, 36, 2, 28]
    mask_exp = mask.unsqueeze(1).unsqueeze(1).expand_as(err)  # [B, 36, 2, 28]
    
    # Sum all errors, then normalize by batch size (ensures consistent loss scale)
    total_loss = (err * mask_exp).sum() / batch_size
    
    return total_loss



def compute_velocity_distribution_loss(model, batch, use_txt_output=False):
    """NOT USED - keeping for compatibility"""
    x = batch["x"]
    output = batch["output"]
    if use_txt_output:
        output = batch["txt_output"]
    
    gt_vel = x[:, :, :, 1:] - x[:, :, :, :-1]
    out_vel = output[:, :, :, 1:] - output[:, :, :, :-1]
    
    mask = batch["mask"][:, 1:]
    
    gt_vel_masked = gt_vel.permute(0, 3, 1, 2)[mask]
    out_vel_masked = out_vel.permute(0, 3, 1, 2)[mask]
    
    gt_vel_mag = torch.sqrt(gt_vel_masked[:, 0]**2 + gt_vel_masked[:, 1]**2)
    out_vel_mag = torch.sqrt(out_vel_masked[:, 0]**2 + out_vel_masked[:, 1]**2)
    
    vel_mean_loss = F.mse_loss(out_vel_mag.mean(), gt_vel_mag.mean())
    vel_std_loss = F.mse_loss(out_vel_mag.std(), gt_vel_mag.std())
    
    vel_x_mean_loss = F.mse_loss(out_vel_masked[:, 0].mean(), gt_vel_masked[:, 0].mean())
    vel_y_mean_loss = F.mse_loss(out_vel_masked[:, 1].mean(), gt_vel_masked[:, 1].mean())
    vel_x_std_loss = F.mse_loss(out_vel_masked[:, 0].std(), gt_vel_masked[:, 0].std())
    vel_y_std_loss = F.mse_loss(out_vel_masked[:, 1].std(), gt_vel_masked[:, 1].std())
    
    gt_vel_range = gt_vel_mag.max() - gt_vel_mag.min()
    out_vel_range = out_vel_mag.max() - out_vel_mag.min()
    vel_range_loss = F.mse_loss(out_vel_range, gt_vel_range)
    
    total_loss = (vel_mean_loss + vel_std_loss + 
                  vel_x_mean_loss + vel_y_mean_loss + 
                  vel_x_std_loss + vel_y_std_loss + 
                  0.5 * vel_range_loss)
    
    return total_loss


def compute_motion_diversity_loss(model, batch, use_txt_output=False):
    """NOT USED - keeping for compatibility"""
    x = batch["x"]
    output = batch["output"]
    if use_txt_output:
        output = batch["txt_output"]
    
    mask = batch["mask"]
    
    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]
    
    gt_points = gtmasked.view(-1, 2)
    out_points = outmasked.view(-1, 2)
    
    gt_dists = torch.cdist(gt_points, gt_points, p=2)
    out_dists = torch.cdist(out_points, out_points, p=2)
    
    triu_mask = torch.triu(torch.ones_like(gt_dists), diagonal=1).bool()
    gt_dists_triu = gt_dists[triu_mask]
    out_dists_triu = out_dists[triu_mask]
    
    dist_mean_loss = F.mse_loss(out_dists_triu.mean(), gt_dists_triu.mean())
    dist_std_loss = F.mse_loss(out_dists_triu.std(), gt_dists_triu.std())
    
    diversity_loss = dist_mean_loss + dist_std_loss
    
    return diversity_loss

def compute_spatial_coherence_loss(model, batch, use_txt_output=False):
    """
    Spatial coherence - averaging kept, point-wise relative positions matched.
    CRITICAL: Normalized by batch size to ensure consistent loss scale across different batch sizes.
    """
    x = batch.get("x_original", batch["x"])
    output = batch["output"]
    if use_txt_output:
        output = batch["txt_output"]
    
    B, N, C, T = x.shape
    batch_size = B
    
    # Part 1: Preserve center of mass trajectory (KEEP THIS)
    gt_com = x.mean(dim=1)  # [B, 2, T]
    out_com = output.mean(dim=1)
    # Use sum for consistency with other losses, then normalize by batch size
    com_loss = F.l1_loss(out_com, gt_com, reduction='sum') / batch_size
    
    # Part 2: Preserve ACTUAL relative positions (NOT just statistics!)
    gt_relative = x - gt_com.unsqueeze(1)  # [B, N, 2, T]
    out_relative = output - out_com.unsqueeze(1)
    
    mask = batch["mask"]
    gt_rel_masked = gt_relative.permute(0, 3, 1, 2)[mask]  # [valid_frames, N, 2]
    out_rel_masked = out_relative.permute(0, 3, 1, 2)[mask]
    
    # OLD (only statistics - CAUSES AVERAGING):
    # rel_std_x_loss = F.mse_loss(out_rel_masked[:, :, 0].std(), gt_rel_masked[:, :, 0].std())
    # rel_std_y_loss = F.mse_loss(out_rel_masked[:, :, 1].std(), gt_rel_masked[:, :, 1].std())
    # total_loss = com_loss + 0.5 * (rel_std_x_loss + rel_std_y_loss)
    
    # NEW (point-wise matching - PRESERVES INDIVIDUAL TRAJECTORIES):
    # Use sum instead of mean for consistency with other losses, then normalize by batch size
    relative_loss = F.l1_loss(out_rel_masked, gt_rel_masked, reduction='sum') / batch_size
    
    total_loss = com_loss + relative_loss
    
    return total_loss


def compute_text_rc_loss(model, batch):
    """NOT USED (replaced by textrecon) - keeping for compatibility"""
    if 'text_output' not in batch:
        return torch.tensor(0.0, device=batch['x'].device)
    
    x = batch.get("x_original", batch["x"])
    output = batch["text_output"]
    mask = batch["mask"]
    
    gtmasked = x.permute(0, 3, 1, 2)[mask]
    outmasked = output.permute(0, 3, 1, 2)[mask]
    
    loss = F.l1_loss(gtmasked, outmasked, reduction='mean')
    return loss


def compute_text_recon_loss(model, batch):
    """
    Simple text-to-motion reconstruction loss: Match GT point-by-point, frame-by-frame.
    
    Same as rc_loss - just direct matching, no complexity.
    CRITICAL: Normalized by batch size to ensure consistent loss scale across different batch sizes.
    """
    if 'text_output' not in batch:
        return torch.tensor(0.0, device=batch['x'].device)
    
    x = batch.get("x_original", batch["x"])
    output = batch["text_output"]
    mask = batch["mask"]
    batch_size = x.shape[0]
    
    # Simple: compute error, mask, sum everything (same as rc_loss)
    err = torch.abs(output - x)  # [bs, npoints, nfeats, nframes]
    mask_exp = mask.unsqueeze(1).unsqueeze(1).expand_as(err)  # [bs, npoints, nfeats, nframes]
    
    # Sum all errors, then normalize by batch size (ensures consistent loss scale)
    recon_loss = (err * mask_exp).sum() / batch_size
    
    return recon_loss


_matching_ = {"rc": compute_rc_loss, "hp": compute_hp_loss,
              "mmd": compute_mmd_loss, "rcxyz": compute_rcxyz_loss,
              "vel": compute_vel_loss, "velxyz": compute_velxyz_loss,
              "range": compute_range_loss,
              "veldist": compute_velocity_distribution_loss,
              "diversity": compute_motion_diversity_loss,
              "spatial": compute_spatial_coherence_loss,
              "text_rc": compute_text_rc_loss,
              "textrecon": compute_text_recon_loss}


def get_loss_function(ltype):
    return _matching_[ltype]


def get_loss_names():
    return list(_matching_.keys())
