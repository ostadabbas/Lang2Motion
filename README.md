# Lang2Motion

Lang2Motion is a neural system that generates human motion trajectories from natural language descriptions using point-based representations and CLIP alignment.

## Overview

Lang2Motion learns to map text descriptions to motion trajectories by:
- Encoding point trajectories using transformer architectures
- Aligning motion and text embeddings in CLIP space
- Generating diverse motions from language prompts

## Key Features

- **Text-to-Motion Generation**: Generate point trajectories from natural language descriptions
- **CLIP Alignment**: Leverage CLIP's multimodal understanding for text-motion correspondence
- **Point-Based Representation**: Efficient 64-point (8x8 grid) motion encoding
- **Multimodal Training**: Support for text and trajectory overlay modalities

## Architecture

Lang2Motion uses a dual-encoder approach with multiple decoder options:

- **Encoder**: Transformer-based motion encoder (4 layers, 4 attention heads)
- **Decoder Options**:
  - **Transformer Autoregressive**: Sequential generation with causal attention
  - **MLP**: Direct mapping from latent to trajectories (tested as baseline)
- **CLIP Integration**: Text-motion alignment using CLIP embeddings
- **Loss Functions**: 
  - Reconstruction loss (L1/L2)
  - Velocity consistency for temporal smoothness
  - Cosine similarity loss for CLIP alignment (not contrastive)

## Quick Start

### Installation

```bash
# Clone the repository
git clone git@bitbucket.org:aclabneu/lang2motion.git
cd lang2motion

# Install dependencies
conda env create -f environment.yml
conda activate lang2motion
```

### Training

```bash
# Train on MeViS dataset
python train_pointclip.py --dataset MeViS --batch_size 32 --epochs 200
```

### Generation

```bash
# Generate motion from text
python generate.py --text "a person walking forward" --output output.npy
```

## Dataset

Lang2Motion uses the MeViS dataset with CoTracker3 point trajectories:
- 1662 video clips with human motion
- Point tracks extracted using CoTracker3
- Text descriptions for each video

## Results

- **Velocity Preservation**: 0.899-1.029 (near-perfect motion preservation)
- **CLIP Alignment**: 9.42 separation score (strong text-motion discrimination)
- **Diversity**: Stochastic sampling for varied motion generation

## Citation

```bibtex
@article{lang2motion2024,
  title={Lang2Motion: Text-Driven Human Motion Generation},
  author={ACLab NEU},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
