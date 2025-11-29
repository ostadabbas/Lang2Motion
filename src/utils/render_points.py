"""
Render point tracks as images for visual CLIP loss
Similar to how MotionCLIP renders 3D human motion
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def render_point_tracks_to_image(point_tracks, image_size=224, show_trails=False, trail_length=10):
    """
    Render point tracks as an image for CLIP visual encoding
    
    IMPORTANT: Uses FIRST FRAME ONLY (like MotionCLIP)
    - Shows initial spatial configuration of points
    - Provides strong spatial initialization signal
    - Model learns to reconstruct full trajectory from this
    
    Args:
        point_tracks: [num_points, 2, num_frames] point trajectories
        image_size: Output image size (CLIP expects 224x224)
        show_trails: Whether to show motion trails (default: False)
        trail_length: Number of frames to show in trail (if enabled)
    
    Returns:
        image_tensor: [3, image_size, image_size] RGB image tensor
    """
    num_points, _, num_frames = point_tracks.shape
    
    # Use FIRST FRAME (frame 0) - matches MotionCLIP approach
    # This shows the initial spatial configuration of points
    frame_idx = 0
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(image_size/100, image_size/100), dpi=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')  # No axes for clean image
    
    # Color map for points
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    
    if show_trails:
        # Optional: Show motion trails (past trajectory)
        # Note: Disabled by default for spatial initialization focus
        start_frame = max(0, frame_idx - trail_length)
        for i in range(num_points):
            x_trail = point_tracks[i, 0, start_frame:frame_idx+1]
            y_trail = point_tracks[i, 1, start_frame:frame_idx+1]
            ax.plot(x_trail, y_trail, '-', color=colors[i], alpha=0.3, linewidth=1)
    
    # Show FIRST FRAME positions (larger points)
    for i in range(num_points):
        x = point_tracks[i, 0, frame_idx]
        y = point_tracks[i, 1, frame_idx]
        ax.plot(x, y, 'o', color=colors[i], markersize=8, markeredgecolor='white', markeredgewidth=0.5)
    
    # Convert figure to image tensor
    fig.tight_layout(pad=0)
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Load as PIL image
    pil_image = Image.open(buf).convert('RGB')
    pil_image = pil_image.resize((image_size, image_size), Image.LANCZOS)
    
    # Convert to tensor [3, H, W]
    image_array = np.array(pil_image).transpose(2, 0, 1)  # HWC -> CHW
    image_tensor = torch.from_numpy(image_array).float() / 255.0
    
    plt.close(fig)
    buf.close()
    
    return image_tensor


def render_point_tracks_batch(point_tracks_batch, image_size=224, show_trails=False, trail_length=10):
    """
    Render a batch of point tracks as images
    
    IMPORTANT: Uses FIRST FRAME ONLY by default (like MotionCLIP)
    - Renders initial spatial configuration
    - Strong spatial initialization signal for CLIP
    - Model learns to predict full trajectory from this
    
    Args:
        point_tracks_batch: [batch_size, num_points, 2, num_frames]
        image_size: Output image size
        show_trails: Whether to show motion trails (default: False)
        trail_length: Number of frames in trail (if enabled)
    
    Returns:
        images: [batch_size, 3, image_size, image_size] batch of images
    """
    batch_size = point_tracks_batch.shape[0]
    device = point_tracks_batch.device
    
    # Convert to numpy for rendering
    point_tracks_np = point_tracks_batch.detach().cpu().numpy()
    
    # Render each sample
    images = []
    for i in range(batch_size):
        image = render_point_tracks_to_image(
            point_tracks_np[i],
            image_size=image_size,
            show_trails=show_trails,
            trail_length=trail_length
        )
        images.append(image)
    
    # Stack and move to device
    images_batch = torch.stack(images).to(device)
    
    return images_batch


def preprocess_for_clip(images):
    """
    Preprocess rendered images for CLIP
    CLIP expects normalized images with mean=[0.48145466, 0.4578275, 0.40821073]
                                      std=[0.26862954, 0.26130258, 0.27577711]
    
    Args:
        images: [batch_size, 3, 224, 224] images in [0, 1] range
    
    Returns:
        normalized_images: [batch_size, 3, 224, 224] normalized for CLIP
    """
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
    
    normalized = (images - mean) / std
    return normalized


def render_and_encode_for_clip(point_tracks_batch, clip_model, image_size=224):
    """
    Render point tracks and encode with CLIP (end-to-end)
    
    Args:
        point_tracks_batch: [batch_size, num_points, 2, num_frames]
        clip_model: CLIP model for encoding
        image_size: Image size (224 for CLIP)
    
    Returns:
        clip_features: [batch_size, 512] CLIP visual features
    """
    # Render to images
    images = render_point_tracks_batch(point_tracks_batch, image_size=image_size)
    
    # Preprocess for CLIP
    images_normalized = preprocess_for_clip(images)
    
    # Encode with CLIP
    with torch.no_grad():
        clip_features = clip_model.encode_image(images_normalized)
        clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
    
    return clip_features.float()


# Test function
if __name__ == "__main__":
    print("Testing point track rendering...")
    
    # Create dummy point tracks
    num_points = 36
    num_frames = 30
    batch_size = 4
    
    # Generate some motion (circular motion for testing)
    t = np.linspace(0, 2*np.pi, num_frames)
    point_tracks = np.zeros((batch_size, num_points, 2, num_frames))
    
    for b in range(batch_size):
        for i in range(num_points):
            radius = 0.5 + 0.3 * (i / num_points)
            angle_offset = (i / num_points) * 2 * np.pi
            point_tracks[b, i, 0, :] = radius * np.cos(t + angle_offset + b*0.5)
            point_tracks[b, i, 1, :] = radius * np.sin(t + angle_offset + b*0.5)
    
    point_tracks_tensor = torch.from_numpy(point_tracks).float()
    
    # Render
    images = render_point_tracks_batch(point_tracks_tensor, image_size=224)
    
    print(f"✅ Rendered batch shape: {images.shape}")
    print(f"✅ Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Save first image for inspection
    first_image = images[0].permute(1, 2, 0).numpy()
    plt.imsave('test_rendered_points.png', first_image)
    print("✅ Saved test image: test_rendered_points.png")
