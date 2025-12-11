# Lang2Motion

Lang2Motion is a framework for language-guided point trajectory generation by aligning motion manifolds with joint embedding spaces. Unlike prior work focusing on human motion or video synthesis, we generate explicit trajectories for arbitrary objects using motion extracted from real-world videos via point tracking.

## Overview

Lang2Motion learns trajectory representations through dual supervision: textual motion descriptions and rendered trajectory visualizations, both mapped through CLIP's frozen encoders. Our transformer-based auto-encoder supports multiple decoder architectures including autoregressive and MLP variants.

## Key Results

- **Text-to-Trajectory Retrieval**: 34.2% Recall@1, outperforming video-based methods by 12.5 points
- **Motion Accuracy**: 33-52% improvement (12.4 ADE vs 18.3-25.3) compared to video generation baselines
- **Action Recognition**: 88.3% Top-1 accuracy on human actions despite training on diverse object motions
- **Applications**: Style transfer, semantic interpolation, and latent-space editing through CLIP-aligned representations

## Architecture

- **Encoder**: Transformer-based motion encoder with point trajectory inputs
- **Decoder Options**:
  - **Transformer Autoregressive**: Sequential generation with causal attention
  - **MLP**: Direct mapping from latent to trajectories
- **CLIP Integration**: Dual supervision through text and trajectory visualizations
- **Loss Functions**: Reconstruction, velocity consistency, and cosine similarity alignment

---

## 🎨 Visualizations

### Framework Overview
<div align="center">
<img src="visualizations/lang2motion_concept.png" alt="Lang2Motion Concept" width="800">
</div>

### Method Architecture
<div align="center">
<img src="visualizations/lang2motion_method.png" alt="Lang2Motion Method" width="800">
</div>

### Teaser Results
<div align="center">
<img src="visualizations/lang2motion_teaser.png" alt="Lang2Motion Teaser" width="800">
</div>

### Motion Interpolation and Trajectory Generation

<div align="center">

**Trajectory Generation from Initial Grid and Latent Space Interpolation**

Given identical text descriptions, Lang2Motion generates different motion interpretations based on initial grid placement and demonstrates semantic interpolation in CLIP's joint embedding space.

<table>
<tr>
<td width="50%" align="center">
<img src="visualizations/interpolation_133c461a4340_exp2_to_6084240e75fa_exp0_fixed.gif" width="350">
<br>
<strong>Left:</strong> From a dancing pose grid, the model emphasizes panda appearance: <span style="color: #3498db;">dancing</span> <span style="color: #ff6b35;">p</span><span style="color: #f7931e;">a</span><span style="color: #2ecc71;">n</span><span style="color: #3498db;">d</span><span style="color: #9b59b6;">a</span>
</td>
<td width="50%" align="center">
<img src="visualizations/interpolation_6084240e75fa_exp0_to_133c461a4340_exp2_fixed.gif" width="350">
<br>
<strong>Right:</strong> From a panda's grid, the model emphasizes dancing motion: <span style="color: #ff6b35;">d</span><span style="color: #f7931e;">a</span><span style="color: #2ecc71;">n</span><span style="color: #3498db;">c</span><span style="color: #9b59b6;">i</span><span style="color: #e91e63;">n</span><span style="color: #e74c3c;">g</span> <span style="color: #3498db;">panda</span>
</td>
</tr>
</table>

**Key Insights:**
- Given identical text *"dancing panda"*, Lang2Motion generates different motion interpretations based on initial grid placement
- Initial grids use automatically retrieved masks; initial video frames shown for visualization only
- Demonstrates semantic interpolation in CLIP's joint embedding space
- Smooth transition between different motion styles while maintaining semantic coherence

</div>

---

## 🚀 Quick Start

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

Lang2Motion uses point trajectories extracted from real-world videos:
- **Source**: Diverse video datasets with object and human motion
- **Tracking**: Point trajectories extracted via CoTracker3
- **Supervision**: Text descriptions and rendered trajectory visualizations
- **Scope**: Arbitrary objects, not limited to human motion

## Results

- **Text-to-Trajectory Retrieval**: 34.2% Recall@1
- **Motion Accuracy**: 12.4 ADE (vs 18.3-25.3 for video baselines)
- **Action Recognition**: 88.3% Top-1 accuracy (cross-domain transfer)
- **Applications**: Style transfer, semantic interpolation, latent-space editing

## Citation

```bibtex
@article{lang2motion2025,
  title={Lang2Motion: Language-Guided Point Trajectory Generation},
  author={Bishoy Galoaa, Xiangyu Bai, Sarah Ostadabbas},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
