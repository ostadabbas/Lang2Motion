#!/usr/bin/env python3
"""
Lang2Motion Generation Script
Generate motion trajectories from text descriptions
"""

import os
import sys
sys.path.append('.')

import torch
import numpy as np
from src.models.modeltype.motionclip import MOTIONCLIP
from src.parser.generate import parser
from src.utils.get_model_and_data import get_model_and_data
import clip

def generate_from_text(model, text, device, num_samples=1):
    """Generate motion from text description"""
    # Tokenize text
    text_tokens = clip.tokenize([text]).to(device)
    
    # Get CLIP text features
    with torch.no_grad():
        text_features = model.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Generate motion
    model.eval()
    with torch.no_grad():
        generated_motions = []
        for _ in range(num_samples):
            # Add small noise for diversity
            noise = torch.randn_like(text_features) * 0.1
            noisy_features = text_features + noise
            
            # Decode to motion
            motion = model.decoder(noisy_features)
            generated_motions.append(motion.cpu().numpy())
    
    return np.array(generated_motions)

def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading Lang2Motion model...")
    parameters = {'model': 'MOTIONCLIP',
                  'latent_dim': 512,
                  **vars(args)}
    
    model, datasets = get_model_and_data(parameters)
    model = model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    
    # Generate from text
    text = args.text or "a person walking forward"
    print(f"Generating motion for: '{text}'")
    
    motions = generate_from_text(model, text, device, num_samples=args.num_samples)
    
    # Save output
    output_path = args.output or f"generated_{text.replace(' ', '_')}.npy"
    np.save(output_path, motions)
    print(f"Saved {motions.shape[0]} motion(s) to {output_path}")

if __name__ == "__main__":
    main()
