import json
import os
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from .dataset import Dataset
from ..utils.tensors import collate
from ..utils.misc import to_torch


class MeVisDataset(Dataset):
    def __init__(self, data_path="/mnt/titan/mevis", split="train", 
                 num_frames=60, max_len=120, grid_size=16, **kwargs):
        """
        MeViS Dataset for PointCLIP
        
        Args:
            data_path: Path to MeViS dataset
            split: 'train' or 'val' 
            num_frames: Number of frames to sample from each video
            max_len: Maximum video length to consider
            grid_size: CoTracker3 grid size (grid_size^2 = number of points)
        """
        # Filter out conflicting parameters for parent class
        parent_kwargs = kwargs.copy()
        parent_kwargs.pop('pose_rep', None)
        parent_kwargs.pop('translation', None) 
        parent_kwargs.pop('glob', None)
        parent_kwargs.pop('data_path', None)
        parent_kwargs.pop('grid_size', None)
        
        # Initialize parent class with PointCLIP parameters
        super().__init__(
            num_frames=num_frames, 
            split=split, 
            max_len=max_len,
            pose_rep="xyz",  # We use point coordinates directly
            translation=False,  # No translation for points
            glob=False,  # No global rotation
            **parent_kwargs
        )
        
        # Dataset paths
        self.data_path = Path(data_path)
        self.split = split
        self.grid_size = grid_size
        
        # PointCLIP specific parameters - configurable grid size
        self.njoints = grid_size * grid_size  # Number of tracked points (grid_size^2)
        self.nfeats = 2  # x, y coordinates
        self.num_classes = 1  # Not used but required by base class
        
        # Initialize CoTracker3 (will be loaded lazily to avoid multiprocessing issues)
        self.cotracker = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._cotracker_loaded = False
        
        # Load dataset metadata
        self.video_ids = self._scan_videos()
        self.captions = self._load_captions()
        
        # For debugging/proof-of-concept: limit to small subset BEFORE creating splits
        debug_subset_size = kwargs.get('debug_subset_size', -1)
        target_videos = kwargs.get('target_videos', [])
        target_expressions = kwargs.get('target_expressions', {})
        
        # Store target expressions for specific video-expression combinations
        self._target_expressions = target_expressions
        
        if debug_subset_size > 0:
            original_size = len(self.video_ids)
            
            # If specific target videos are requested, prioritize them
            if target_videos:
                # First, add target videos if they exist
                priority_videos = [vid for vid in target_videos if vid in self.video_ids]
                # Then add remaining videos to reach subset size
                remaining_videos = [vid for vid in self.video_ids if vid not in priority_videos]
                
                subset_videos = priority_videos + remaining_videos[:debug_subset_size - len(priority_videos)]
                self.video_ids = subset_videos[:debug_subset_size]
                
                print(f"DEBUG: Using subset of {len(self.video_ids)}/{original_size} videos")
                if priority_videos:
                    print(f"       Prioritized {len(priority_videos)} target videos: {priority_videos}")
            else:
                # Original behavior: take first N videos
                self.video_ids = self.video_ids[:debug_subset_size]
                print(f"DEBUG: Using subset of {len(self.video_ids)}/{original_size} videos for proof-of-concept")
        
        # Create train/test splits AFTER subset is applied
        # The parent class expects _train and _test attributes
        if split == 'train':
            self._train = list(range(len(self.video_ids)))
            self._test = []
        else:
            self._train = []
            self._test = list(range(len(self.video_ids)))
        
        print(f"MeViS Dataset initialized: {len(self.video_ids)} videos, "
              f"{self.njoints} points ({self.grid_size}x{self.grid_size} grid), split={split}")
    
    def _load_cotracker(self):
        """Load CoTracker3 model - using local development version"""
        if not hasattr(self, 'cotracker') or self.cotracker is None:
            print(f"Loading CoTracker3 (local dev version) in process {os.getpid()}...")
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use EXACT demo pattern from co-tracker/demo.py - ALWAYS try this first
            checkpoint_path = "/media/galoaab/Documents/MotionCLIP/co-tracker/checkpoints/scaled_offline.pth"
            
            try:
                from cotracker.predictor import CoTrackerPredictor
                self.cotracker = CoTrackerPredictor(
                    checkpoint=checkpoint_path,
                    v2=False,  # CoTracker3
                    offline=True,
                    window_len=60
                )
                self.cotracker = self.cotracker.to(self._device)
                print(f"✅ CoTracker3 (local with checkpoint) loaded on {self._device}")
                return  # Success!
            except Exception as e:
                print(f"⚠️ Local CoTracker3 failed: {e}")
                
            # If local fails, try torch.hub as fallback
            try:
                self.cotracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline', trust_repo=True)
                self.cotracker = self.cotracker.to(self._device)
                self.cotracker.eval()
                print(f"✅ CoTracker3 (torch.hub fallback) loaded on {self._device}")
                return  # Success!
            except Exception as e2:
                print(f"❌ All CoTracker3 loading methods failed: {e2}")
                self.cotracker = None
                raise RuntimeError("Could not load CoTracker3 - both local and torch.hub failed")
    
    def _scan_videos(self):
        """Scan video directories and return list of video IDs"""
        video_dir = self.data_path / self.split / "JPEGImages"
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        video_ids = [d.name for d in video_dir.iterdir() if d.is_dir()]
        print(f"Found {len(video_ids)} videos in {self.split} split")
        return sorted(video_ids)
    
    def _load_captions(self):
        """Load caption metadata"""
        caption_file = self.data_path / self.split / "meta_expressions.json"
        if not caption_file.exists():
            raise FileNotFoundError(f"Caption file not found: {caption_file}")
        
        with open(caption_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded captions for {len(data['videos'])} videos")
        return data['videos']
    
    def _create_splits(self):
        """Create train/test splits - for now just use all data as train"""
        # This will be called AFTER the subset is applied
        indices = list(range(len(self.video_ids)))
        
        if self.split == 'train':
            self._train = indices
            self._test = []
        else:
            self._train = []
            self._test = indices
        
        # Required by base class
        self._actions = [0] * len(self.video_ids)  # Dummy actions
        self._num_frames_in_video = [self.num_frames] * len(self.video_ids)  # Dummy frame counts
    
    def _load_video_frames(self, video_id, max_frames=None):
        """
        Load video frames for a given video ID
        
        Returns:
            video_tensor: [1, T, 3, H, W] tensor normalized to [0, 1]
        """
        video_dir = self.data_path / self.split / "JPEGImages" / video_id
        frame_files = sorted(list(video_dir.glob("*.jpg")))
        
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        if len(frame_files) == 0:
            raise ValueError(f"No frames found for video {video_id}")
        
        frames = []
        for frame_file in frame_files:
            # Load and convert BGR to RGB
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and normalize to [0, 1]
            frame = torch.from_numpy(frame).float() / 255.0
            frames.append(frame)
        
        if len(frames) == 0:
            raise ValueError(f"No valid frames loaded for video {video_id}")
        
        # Stack to [T, H, W, 3] then permute to [T, 3, H, W]
        video_tensor = torch.stack(frames).permute(0, 3, 1, 2)
        
        # Add batch dimension: [1, T, 3, H, W]
        return video_tensor.unsqueeze(0)
    
    def _create_mask_based_grid(self, mask, grid_size=12):
        """
        Create a FIXED grid of points ONLY ON ACTUAL MASK PIXELS.
        Distributes points evenly across the mask region, not the bounding box!
        
        Args:
            mask: Binary mask [H, W] where >0 indicates object
            grid_size: Target grid size (will generate grid_size^2 points)
        
        Returns:
            points: [N, 2] array of (x, y) coordinates, exactly grid_size^2 points
        """
        import cv2
        
        H, W = mask.shape
        num_points = grid_size * grid_size
        
        # Erode mask by 3 pixels to stay away from borders
        # Use 3x3 kernel with 3 iterations to erode by 3 pixels
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=3)
        
        # If erosion removes everything, use slightly less (2 pixels)
        if np.sum(eroded_mask > 0) < num_points:
            eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
        
        # Use the eroded mask - points will be INSIDE, away from borders
        object_pixels = np.where(eroded_mask > 0)
        
        if len(object_pixels[0]) == 0:
            raise ValueError(f"❌ No object pixels found in mask! Cannot create mask-based grid.")
        
        # Get mask bounding box to create a uniform grid
        y_min, y_max = object_pixels[0].min(), object_pixels[0].max()
        x_min, x_max = object_pixels[1].min(), object_pixels[1].max()
        
        # CRITICAL FIX: Add margin INSIDE the bounding box so grid doesn't touch edges!
        # linspace includes endpoints, so we need to shrink the range
        margin_y = (y_max - y_min) * 0.1  # 10% margin on each side
        margin_x = (x_max - x_min) * 0.1
        
        # Create a UNIFORM grid INSIDE the bounding box (not touching edges)
        y_coords = np.linspace(y_min + margin_y, y_max - margin_y, grid_size, dtype=np.float32)
        x_coords = np.linspace(x_min + margin_x, x_max - margin_x, grid_size, dtype=np.float32)
        
        # Create grid points
        selected_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Grid point in bounding box
                grid_x = x_coords[j]
                grid_y = y_coords[i]
                
                # Find the nearest mask pixel to this grid point
                # This ensures the point is ON the mask while maintaining grid structure
                distances = (object_pixels[1] - grid_x)**2 + (object_pixels[0] - grid_y)**2
                nearest_idx = np.argmin(distances)
                
                # Get the nearest mask pixel
                nearest_x = object_pixels[1][nearest_idx]
                nearest_y = object_pixels[0][nearest_idx]
                
                selected_points.append([nearest_x, nearest_y])
        
        selected_points = np.array(selected_points, dtype=np.float32)
        
        # Points are from eroded mask, so they're automatically away from borders
        return selected_points.astype(np.float32)

    def _filter_tracks_by_mask(self, tracks, visibility, mask, video_shape):
        """
        Filter tracks to keep only points that start on the mask
        
        Args:
            tracks: [1, T, N, 2] point trajectories
            visibility: [1, T, N] visibility flags  
            mask: [H, W] binary mask
            video_shape: (H, W) video dimensions
            
        Returns:
            filtered_tracks: [1, T, M, 2] where M <= N
            filtered_visibility: [1, T, M]
        """
        H, W = video_shape
        
        # Resize mask to match video if needed
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Get first frame positions
        first_frame_tracks = tracks[0, 0, :, :]  # [N, 2]
        
        # Find which points start on the mask
        mask_indices = []
        for i, point in enumerate(first_frame_tracks):
            x, y = int(point[0].item()), int(point[1].item())
            if 0 <= x < W and 0 <= y < H and mask[y, x] > 0:
                mask_indices.append(i)
        
        # ALWAYS return exactly grid_size^2 points for consistent batching
        target_points = self.njoints  # Configurable number (grid_size^2)
        
        if len(mask_indices) == 0:
            print("⚠️ No points found on mask, using center region")
            # Fallback: create points in center region
            H, W = video_shape
            center_y, center_x = H//2, W//2
            radius = min(H, W) // 6
            selected_indices = []
            for i in range(target_points):
                # Create spiral pattern in center
                angle = i * 0.5
                r = (i / target_points) * radius
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))
                # Find closest actual track point
                distances = [(x - tracks[0, 0, j, 0].item())**2 + (y - tracks[0, 0, j, 1].item())**2 
                           for j in range(tracks.shape[2])]
                closest_idx = np.argmin(distances)
                selected_indices.append(closest_idx)
        elif len(mask_indices) >= target_points:
            # Sample evenly from mask points
            step = len(mask_indices) / target_points
            selected_indices = [mask_indices[int(i * step)] for i in range(target_points)]
        else:
            # Repeat mask points to reach target
            selected_indices = (mask_indices * (target_points // len(mask_indices) + 1))[:target_points]
        
        # Filter tracks and visibility
        selected_indices = torch.tensor(selected_indices, dtype=torch.long)
        filtered_tracks = tracks[:, :, selected_indices, :]
        filtered_visibility = visibility[:, :, selected_indices]
        
        print(f"🎯 Filtered to exactly {filtered_tracks.shape[2]} points for consistent batching")
        return filtered_tracks, filtered_visibility

    def _load_mask(self, video_id, exp_id=None, frame_idx=0):
        """
        Load object mask for specific expression
        
        Args:
            video_id: Video identifier
            exp_id: Expression ID (if None, uses first expression)
            frame_idx: Frame index to load mask for
        """
        # Load metadata and mask dict (cache them)
        if not hasattr(self, '_meta_data'):
            meta_file = self.data_path / self.split / "meta_expressions.json"
            with open(meta_file, 'r') as f:
                self._meta_data = json.load(f)
        
        if not hasattr(self, '_mask_dict'):
            mask_file = self.data_path / self.split / "mask_dict.json"
            with open(mask_file, 'r') as f:
                self._mask_dict = json.load(f)
        
        # Get video metadata
        if video_id not in self._meta_data['videos']:
            return None
        
        video_data = self._meta_data['videos'][video_id]
        
        # Get specified expression's anno_ids (or first if not specified)
        expressions = video_data.get('expressions', {})
        if len(expressions) == 0:
            return None
        
        # Use specified expression ID, or fallback to first
        if exp_id is None or exp_id not in expressions:
            exp_id = list(expressions.keys())[0]
        
        exp_data = expressions[exp_id]
        anno_ids = [str(x) for x in exp_data['anno_id']]
        
        # Load mask using EXACT working code from proper_mevis_loading.py
        combined_mask = None
        
        for anno_id in anno_ids:
            if anno_id in self._mask_dict and frame_idx < len(self._mask_dict[anno_id]):
                frame_anno = self._mask_dict[anno_id][frame_idx]
                if frame_anno is not None:
                    try:
                        from pycocotools import mask as coco_mask
                        decoded_mask = coco_mask.decode(frame_anno)
                        
                        if combined_mask is None:
                            combined_mask = decoded_mask.astype(np.float32)
                        else:
                            combined_mask += decoded_mask.astype(np.float32)
                            
                    except Exception as e:
                        continue
        
        if combined_mask is not None:
            # Clip to [0, 1] and convert to uint8 (same as working version)
            combined_mask = np.clip(combined_mask, 0, 1)
            return (combined_mask * 255).astype(np.uint8)
        
        return None

    def _run_cotracker(self, video_tensor, video_id=None, exp_id=None):
        """
        Run CoTracker3 on video tensor with STRICT mask-based initialization
        Uses the segm_mask parameter which is the proper way to do mask-based tracking
        
        Args:
            video_tensor: [1, T, 3, H, W]
            video_id: Video identifier for loading mask (REQUIRED)
            exp_id: Expression ID for loading correct mask (REQUIRED for correct text-mask matching)
            
        Returns:
            tracks: [1, T, N, 2] point trajectories where N=grid_size^2
            visibility: [1, T, N] visibility flags where N=grid_size^2
        """
        self._load_cotracker()
        
        if video_id is None:
            raise ValueError("❌ video_id is required for mask-based tracking!")
        
        with torch.no_grad():
            video_tensor = video_tensor.to(self._device)
            
            # Load mask for specified expression - skip videos without masks
            mask = self._load_mask(video_id, exp_id=exp_id, frame_idx=0)
            if mask is None:
                return None
            
            H, W = video_tensor.shape[3], video_tensor.shape[4]
            
            # Resize mask to match video dimensions if needed
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # SIMPLE APPROACH: Create exactly grid_size^2 points ON the mask
            # Use CoTracker3 queries parameter with (t, x, y) format
            
            # Create exactly grid_size^2 points on the mask
            mask_points = self._create_mask_based_grid(mask, grid_size=self.grid_size)  # This creates grid_size^2 points
            
            # Convert to CoTracker3 queries format: (B, N, 3) with (t, x, y)
            B = video_tensor.shape[0]
            N = len(mask_points)
            queries = torch.zeros(B, N, 3, device=self._device)
            queries[:, :, 0] = 0  # All points start at frame 0
            queries[:, :, 1] = torch.from_numpy(mask_points[:, 0]).float()  # x coordinates
            queries[:, :, 2] = torch.from_numpy(mask_points[:, 1]).float()  # y coordinates
            
            # Use CoTracker3 with queries - SIMPLE AND DIRECT
            tracks, visibility = self.cotracker(video_tensor, queries=queries)
            
            return tracks.cpu(), visibility.cpu()
                
    
    def _deduplicate_initial_positions(self, tracks):
        """
        Fix duplicate positions at frame 0 by adding small random offsets.
        
        Args:
            tracks: [T, N, 2] point trajectories in pixel coordinates
            
        Returns:
            tracks_fixed: [T, N, 2] tracks with duplicate positions fixed
        """
        T, N, _ = tracks.shape
        tracks_fixed = tracks.clone()
        
        # Get frame 0 positions
        frame0_positions = tracks[0]  # [N, 2]
        
        # Find duplicate positions (within 1 pixel tolerance)
        duplicate_threshold = 1.0  # pixels
        positions_fixed = frame0_positions.clone()
        
        # Check for duplicates
        for i in range(N):
            for j in range(i + 1, N):
                dist = torch.norm(frame0_positions[i] - frame0_positions[j]).item()
                if dist < duplicate_threshold:
                    # Add small random offset to break tie
                    # Offset: random direction, small magnitude (1-3 pixels)
                    import random
                    angle = random.uniform(0, 2 * np.pi)
                    offset_magnitude = random.uniform(1.0, 3.0)
                    offset_x = offset_magnitude * np.cos(angle)
                    offset_y = offset_magnitude * np.sin(angle)
                    
                    # Apply offset to point j (keep point i unchanged)
                    positions_fixed[j] = positions_fixed[j] + torch.tensor([offset_x, offset_y], 
                                                                           device=positions_fixed.device)
                    
                    # Propagate offset to all frames (maintain relative positions)
                    tracks_fixed[:, j, :] = tracks_fixed[:, j, :] + torch.tensor([offset_x, offset_y],
                                                                                  device=tracks_fixed.device)
        
        # Update frame 0
        tracks_fixed[0] = positions_fixed
        
        return tracks_fixed
    
    def _process_tracks(self, tracks, image_shape):
        """
        Process tracks for MotionCLIP format
        
        Args:
            tracks: [1, T, N, 2] point trajectories in pixel coordinates
            image_shape: (H, W) image dimensions
            
        Returns:
            tracks_processed: [N, 2, T] normalized and reshaped tracks
        """
        H, W = image_shape
        
        # Remove batch dimension: [1, T, N, 2] -> [T, N, 2]
        tracks = tracks.squeeze(0)
        
        # FIX: Deduplicate initial positions
        tracks = self._deduplicate_initial_positions(tracks)
        
        # Normalize coordinates to [-1, 1]
        tracks_norm = tracks.clone()
        tracks_norm[..., 0] = (tracks[..., 0] / W) * 2 - 1  # x: [0,W] -> [-1,1]
        tracks_norm[..., 1] = (tracks[..., 1] / H) * 2 - 1  # y: [0,H] -> [-1,1]
        
        # Reshape to MotionCLIP format: [T, N, 2] -> [N, 2, T]
        tracks_processed = tracks_norm.permute(1, 2, 0)
        
        return tracks_processed.float()
    
    def _get_random_expression(self, video_id):
        """Get a random expression ID and text for the given video ID"""
        if video_id not in self.captions:
            return None, "A video sequence"  # Fallback caption
        
        expressions = self.captions[video_id]['expressions']
        if len(expressions) == 0:
            return None, "A video sequence"
        
        # Check if this is a target panda video with specific expression preference
        target_expressions = getattr(self, '_target_expressions', {})
        if video_id in target_expressions:
            preferred_exp_id = target_expressions[video_id]
            if preferred_exp_id in expressions:
                return preferred_exp_id, expressions[preferred_exp_id]['exp']
        
        # Randomly select one expression and return BOTH ID and text
        exp_id = random.choice(list(expressions.keys()))
        return exp_id, expressions[exp_id]['exp']
    
    def _render_tracks(self, tracks, frame_idx=0, image_size=(224, 224)):
        """
        Render point trajectories for CLIP image input
        
        Args:
            tracks: [N, 2, T] point trajectories in [-1, 1] range
            frame_idx: Which frame to render
            image_size: Output image size for CLIP
            
        Returns:
            rendered_image: [3, H, W] RGB tensor for CLIP
        """
        H, W = image_size
        
        # Create black background
        image = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Get current point positions: [N, 2]
        if frame_idx >= tracks.shape[2]:
            frame_idx = tracks.shape[2] - 1
        
        current_points = tracks[:, :, frame_idx]  # [N, 2]
        
        # Convert from [-1, 1] to pixel coordinates
        points_pixel = current_points.clone()
        points_pixel[:, 0] = (points_pixel[:, 0] + 1) * W / 2  # x
        points_pixel[:, 1] = (points_pixel[:, 1] + 1) * H / 2  # y
        
        # Draw points
        for i, (x, y) in enumerate(points_pixel):
            x, y = int(x.item()), int(y.item())
            if 0 <= x < W and 0 <= y < H:
                # Use different colors for different points
                color = (
                    int(255 * (i % 3) / 3),
                    int(255 * ((i // 3) % 3) / 3), 
                    int(255 * ((i // 9) % 3) / 3)
                )
                cv2.circle(image, (x, y), 2, color, -1)
        
        # Convert to tensor: [H, W, 3] -> [3, H, W]
        image_tensor = torch.from_numpy(image).float() / 255.0
        return image_tensor.permute(2, 0, 1)
    
    def _create_trajectory_overlays(self, tracks_processed, video_tensor):
        """
        Create trajectory overlays on actual video frames
        
        Args:
            tracks_processed: [N, 2, T] normalized trajectories in [-1, 1]
            video_tensor: [1, T, 3, H, W] video frames in [0, 1]
            
        Returns:
            overlays: [T, 3, 224, 224] frames with trajectory overlays
        """
        # Remove batch dimension from video
        video_frames = video_tensor.squeeze(0)  # [T, 3, H, W]
        T, _, H, W = video_frames.shape
        N = tracks_processed.shape[0]
        
        # Resize frames to 224x224 for CLIP compatibility
        overlaid_frames = []
        
        for t in range(T):
            # Get current frame and resize to 224x224
            frame = video_frames[t]  # [3, H, W]
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
            frame_resized = cv2.resize(frame_np, (224, 224))
            
            # Overlay trajectories on this frame
            img = frame_resized.copy()
            
            # Draw trajectories up to current frame
            for point_idx in range(N):
                trajectory = tracks_processed[point_idx].cpu().numpy()  # [2, T]
                
                if not np.isfinite(trajectory).any():
                    continue
                
                # Convert from [-1, 1] to [0, 223] pixel coordinates
                x_coords = ((trajectory[0] + 1) / 2 * 223).astype(int)
                y_coords = ((trajectory[1] + 1) / 2 * 223).astype(int)
                x_coords = np.clip(x_coords, 0, 223)
                y_coords = np.clip(y_coords, 0, 223)
                
                # Use unified bright cyan color for abstract visualization
                unified_color = (0, 255, 255)  # Bright cyan - pops on any background
                
                # Draw trajectory up to current frame with fading
                for frame_idx in range(min(t, len(x_coords) - 1)):
                    if (np.isfinite(x_coords[frame_idx]) and np.isfinite(y_coords[frame_idx]) and
                        np.isfinite(x_coords[frame_idx + 1]) and np.isfinite(y_coords[frame_idx + 1])):
                        
                        # Fade older trajectory segments
                        alpha = max(0.4, (frame_idx + 1) / (t + 1))
                        faded_color = tuple(int(c * alpha) for c in unified_color)
                        
                        # Use anti-aliased lines for smooth appearance
                        cv2.line(img, 
                                (int(x_coords[frame_idx]), int(y_coords[frame_idx])),
                                (int(x_coords[frame_idx + 1]), int(y_coords[frame_idx + 1])),
                                faded_color, 2, cv2.LINE_AA)
                
                # Draw current point with smooth circles
                if t < len(x_coords) and np.isfinite(x_coords[t]) and np.isfinite(y_coords[t]):
                    cv2.circle(img, (int(x_coords[t]), int(y_coords[t])), 6, (255, 255, 255), 2, cv2.LINE_AA)  # White outline
                    cv2.circle(img, (int(x_coords[t]), int(y_coords[t])), 4, unified_color, -1, cv2.LINE_AA)  # Cyan center
            
            # Convert back to tensor format
            frame_tensor = torch.from_numpy(img).float() / 255.0  # [224, 224, 3]
            frame_tensor = frame_tensor.permute(2, 0, 1)  # [3, 224, 224]
            overlaid_frames.append(frame_tensor)
        
        # Stack all frames: [T, 3, 224, 224]
        return torch.stack(overlaid_frames)
    
    def _get_item_data_index(self, data_index):
        """Override base class method for MeViS-specific data loading"""
        video_id = self.video_ids[data_index]
        
        try:
            # 1. Get random expression ID and text TOGETHER (FIX: ensures text-mask match!)
            exp_id, caption = self._get_random_expression(video_id)
            
            # 2. Load video frames
            video_tensor = self._load_video_frames(video_id, max_frames=self.num_frames)
            
            # Skip videos with fewer frames than required
            actual_frames = video_tensor.shape[1]
            if actual_frames < self.num_frames:
                return None  # Skip short videos
            
            # 3. Run CoTracker3 with mask for SAME expression (FIX: uses same exp_id!)
            result = self._run_cotracker(video_tensor, video_id=video_id, exp_id=exp_id)
            if result is None:
                # No mask found, skip this video
                return None
            tracks, visibility = result
            
            # 4. Process tracks for MotionCLIP format
            image_shape = video_tensor.shape[-2:]  # (H, W)
            tracks_processed = self._process_tracks(tracks, image_shape)
            
            # 5. NEW: Create trajectory overlays on actual video frames
            trajectory_overlays = self._create_trajectory_overlays(tracks_processed, video_tensor)
            
            # 6. Create output compatible with MotionCLIP
            # The collate function expects 'target' to be a scalar label, not a tensor
            output = {
                'inp': tracks_processed,  # [N, 2, T]
                'target': 0,  # Dummy scalar label (not used in PointCLIP)
                'clip_text': caption,
                'trajectory_overlays': trajectory_overlays  # [T, 3, 224, 224] - NEW multimodal input
                # Note: Removed 'clip_image' since we're not using static rendering anymore
            }
            
            # Explicit memory cleanup
            del video_tensor, tracks
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return output
            
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            # Cleanup on error
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # Return None to skip this sample (handled by collate function)
            return None
    
    def get_action(self, ind):
        """Override base class method - not used for PointCLIP"""
        return 0
    
    def get_label(self, ind):
        """Override base class method - not used for PointCLIP"""
        return 0
    
    @property
    def dataname(self):
        """Dataset name for compatibility"""
        return "mevis"
