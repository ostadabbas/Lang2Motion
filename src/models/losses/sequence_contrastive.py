"""
Sequence-level contrastive loss
Aligns temporal motion sequences with text token sequences
Preserves structure instead of compressing to single vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceContrastiveLoss(nn.Module):
    """
    Compute contrastive loss between temporal sequences
    
    Motion: [bs, num_frames, dim]
    Text: [bs, num_tokens, dim]
    
    Uses attention to compute sequence-to-sequence similarity
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, motion_seq, text_seq, logit_scale=None):
        """
        Args:
            motion_seq: [bs, num_frames, dim] temporal motion latents
            text_seq: [bs, num_tokens, dim] text token embeddings
            logit_scale: scalar for CLIP-style scaling
            
        Returns:
            loss: contrastive loss
            logits: similarity matrix [bs, bs]
        """
        bs = motion_seq.shape[0]
        
        # Normalize sequences
        motion_seq = F.normalize(motion_seq, dim=-1)  # [bs, num_frames, dim]
        text_seq = F.normalize(text_seq, dim=-1)  # [bs, num_tokens, dim]
        
        # Compute pairwise sequence similarities
        # For each (motion_i, text_j) pair, compute attention-weighted similarity
        similarities = torch.zeros(bs, bs, device=motion_seq.device)
        
        for i in range(bs):
            for j in range(bs):
                # motion_i: [num_frames, dim]
                # text_j: [num_tokens, dim]
                motion_i = motion_seq[i]  # [num_frames, dim]
                text_j = text_seq[j]  # [num_tokens, dim]
                
                # Compute frame-to-token similarities
                # [num_frames, dim] @ [dim, num_tokens] = [num_frames, num_tokens]
                frame_token_sim = motion_i @ text_j.T
                
                # Aggregate: mean over both dimensions
                # This gives overall sequence-to-sequence similarity
                seq_sim = frame_token_sim.mean()
                
                similarities[i, j] = seq_sim
        
        # Apply temperature scaling
        if logit_scale is not None:
            logits = logit_scale * similarities
        else:
            logits = similarities / self.temperature
        
        # Contrastive loss (symmetric)
        labels = torch.arange(bs, device=motion_seq.device)
        loss_motion = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        loss = (loss_motion + loss_text) / 2
        
        return loss, logits


class AttentionPooling(nn.Module):
    """
    Attention-based pooling from sequence to single vector
    Better than mean pooling - learns to weight important frames/tokens
    """
    
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Linear(dim, 1)
        
    def forward(self, seq):
        """
        Args:
            seq: [bs, seq_len, dim]
        Returns:
            pooled: [bs, dim]
        """
        # Compute attention weights
        attn_weights = self.attention(seq)  # [bs, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # [bs, seq_len, 1]
        
        # Weighted sum
        pooled = (seq * attn_weights).sum(dim=1)  # [bs, dim]
        
        return pooled


class SequenceContrastiveWithPooling(nn.Module):
    """
    Hybrid approach:
    1. Use attention pooling to get single vectors
    2. Compute contrastive loss on pooled vectors
    
    Better than mean pooling, but still loses some structure
    """
    
    def __init__(self, dim, temperature=0.07):
        super().__init__()
        self.motion_pooling = AttentionPooling(dim)
        self.text_pooling = AttentionPooling(dim)
        self.temperature = temperature
        
    def forward(self, motion_seq, text_seq, logit_scale=None):
        """
        Args:
            motion_seq: [bs, num_frames, dim]
            text_seq: [bs, num_tokens, dim]
            
        Returns:
            loss: contrastive loss
            logits: similarity matrix [bs, bs]
        """
        # Pool sequences to single vectors
        motion_pooled = self.motion_pooling(motion_seq)  # [bs, dim]
        text_pooled = self.text_pooling(text_seq)  # [bs, dim]
        
        # Normalize
        motion_pooled = F.normalize(motion_pooled, dim=-1)
        text_pooled = F.normalize(text_pooled, dim=-1)
        
        # Compute similarities
        logits = motion_pooled @ text_pooled.T  # [bs, bs]
        
        # Apply scaling
        if logit_scale is not None:
            logits = logit_scale * logits
        else:
            logits = logits / self.temperature
        
        # Contrastive loss
        bs = motion_seq.shape[0]
        labels = torch.arange(bs, device=motion_seq.device)
        loss_motion = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        loss = (loss_motion + loss_text) / 2
        
        return loss, logits


class HierarchicalContrastiveLoss(nn.Module):
    """
    Multi-level contrastive loss:
    1. Sequence-level (preserves temporal structure)
    2. Pooled-level (global alignment)
    
    Combines benefits of both
    """
    
    def __init__(self, dim, temperature=0.07, seq_weight=0.5, pooled_weight=0.5):
        super().__init__()
        self.seq_loss = SequenceContrastiveLoss(temperature)
        self.pooled_loss = SequenceContrastiveWithPooling(dim, temperature)
        self.seq_weight = seq_weight
        self.pooled_weight = pooled_weight
        
    def forward(self, motion_seq, text_seq, logit_scale=None):
        """
        Args:
            motion_seq: [bs, num_frames, dim]
            text_seq: [bs, num_tokens, dim]
            
        Returns:
            loss: combined loss
            metrics: dict with individual losses
        """
        # Sequence-level loss
        seq_loss, seq_logits = self.seq_loss(motion_seq, text_seq, logit_scale)
        
        # Pooled-level loss
        pooled_loss, pooled_logits = self.pooled_loss(motion_seq, text_seq, logit_scale)
        
        # Combined loss
        total_loss = (self.seq_weight * seq_loss + 
                     self.pooled_weight * pooled_loss)
        
        metrics = {
            'seq_loss': seq_loss.item(),
            'pooled_loss': pooled_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics


def test_sequence_contrastive():
    """Test sequence contrastive loss"""
    
    print("Testing Sequence Contrastive Loss")
    print("="*70)
    
    # Create dummy data
    bs = 4
    num_frames = 30
    num_tokens = 77
    dim = 512
    
    motion_seq = torch.randn(bs, num_frames, dim).cuda()
    text_seq = torch.randn(bs, num_tokens, dim).cuda()
    
    # Test 1: Basic sequence contrastive
    print("\n1. Basic Sequence Contrastive")
    print("-"*70)
    loss_fn = SequenceContrastiveLoss().cuda()
    loss, logits = loss_fn(motion_seq, text_seq)
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Diagonal (correct pairs): {torch.diag(logits).mean().item():.4f}")
    print(f"Off-diagonal (wrong pairs): {logits[~torch.eye(bs, dtype=bool)].mean().item():.4f}")
    
    # Test 2: Attention pooling
    print("\n2. Attention Pooling Contrastive")
    print("-"*70)
    loss_fn = SequenceContrastiveWithPooling(dim).cuda()
    loss, logits = loss_fn(motion_seq, text_seq)
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Diagonal (correct pairs): {torch.diag(logits).mean().item():.4f}")
    print(f"Off-diagonal (wrong pairs): {logits[~torch.eye(bs, dtype=bool)].mean().item():.4f}")
    
    # Test 3: Hierarchical
    print("\n3. Hierarchical Contrastive")
    print("-"*70)
    loss_fn = HierarchicalContrastiveLoss(dim).cuda()
    loss, metrics = loss_fn(motion_seq, text_seq)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Sequence loss: {metrics['seq_loss']:.4f}")
    print(f"Pooled loss: {metrics['pooled_loss']:.4f}")
    
    print("\n" + "="*70)
    print("All tests passed!")


if __name__ == "__main__":
    test_sequence_contrastive()
