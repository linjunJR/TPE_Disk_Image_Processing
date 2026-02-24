# TPE Disk Image Processing

This repository contains a three-step image analysis pipeline for tracking photoelastic disks, detecting contacts, and computing force vectors in granular material experiments.

## Workflow Overview

The analysis pipeline consists of three sequential notebooks that process experimental images to extract particle trajectories and force networks:

```mermaid

flowchart TD
    I1[Green Images] --> B[01.TPE_disk_tracking_stardist.ipynb]
    I2[UV Image] --> B
    I3[PE Image] --> D
    B --> C[Trajectory .pkl File<br/> -> positions, angles, IDs]
    C --> D[02.TPE_contact_detect.ipynb]
    D --> E[Contact Bond .pkl File<br/> -> pairs, positions, angles]
    E --> F[03.TPE_solve_force_vector_with_ResNet_guess.ipynb]
    F --> G[Force .pkl File<br/> -> magnitudes & angles of contact forces]
        
    style I1 stroke:#4CAF50,stroke-width:3px
    style I2 stroke:#2196F3,stroke-width:3px
    style I3 stroke:#FFC107,stroke-width:3px
    style C stroke:#FFA726,stroke-width:2px
    style E stroke:#FFA726,stroke-width:2px
    style G stroke:#FFA726,stroke-width:2px
    style B stroke:#9C27B0,stroke-width:2px
    style D stroke:#9C27B0,stroke-width:2px
    style F stroke:#9C27B0,stroke-width:2px
```

## Pipeline Steps

### Step 1: Disk Tracking with StarDist
**Notebook:** `01.TPE_disk_tracking_stardist.ipynb`

This notebook performs automated detection and tracking of photoelastic disks throughout the experiment.

**Key Features:**
- Disk detection using pre-trained StarDist2D model
- Particle linking into trajectories using Trackpy
- Rotation angle computation via PCA on disk orientation markers
- Boundary particle identification

**Inputs:**
- Raw experimental images
- StarDist model for disk segmentation

**Outputs:**
- Pickle file containing:
  - Particle positions (x, y) for each frame in pixels
  - Particle IDs and trajectories
  - Disk radii (rpx) in pixels
  - Angular positions (theta)
  - Boundary particle tags

### Step 2: Contact Detection
**Notebook:** `02. TPE_contact_detect.ipynb`

Identifies and classifies contacts between particles using a trained CNN model.

**Key Features:**
- Neighbor detection based on distance threshold
- Contact classification using neural network

**Inputs:**
- Trajectory pickle file from Step 1
- PE images
- Pre-trained contact detection model

**Outputs:**
- Contact dataframe with:
  - Contact pairs (i, j)
  - Contact positions (xi, yi, xj, yj)
  - Contact angles (beta)
  - Classification scores

### Step 3: Force Vector Computation
**Notebook:** `03. TPE_solve_force_vector_with_ResNet_guess.ipynb`

Computes force magnitudes and directions at each contact using photoelastic image analysis and optimization.

**Key Features:**
- Initial force guess using ResNet regression model
- Force optimization with equilibrium constraints (∑F=0, ∑τ=0)

**Inputs:**
- Contact data from Step 2
- Photoelastic images
- Pre-trained force prediction model

**Outputs:**
- Force vectors (magnitude and angle) at each contact
- Total force on each particle

## Setup

### Environment Overview

This pipeline uses **two separate conda environments** due to TensorFlow 2.10's native Windows GPU requirement (NumPy 1.x ABI) conflicting with PyTorch 2.x (NumPy 2.x):

| Environment | Notebook | GPU backend | Key packages |
|---|---|---|---|
| `stardist_env` | 01 — Disk tracking | TF 2.10 + CUDA 11.2 | TensorFlow-GPU, StarDist, CSBDeep, Trackpy |
| `torch_env` | 02 — Contact detect<br/>03 — Force solve | PyTorch + CUDA 12.6 | PyTorch 2.6+cu126, Torchvision |



### Installation

**Create both environments:**
```bash
cd environments/
conda env create -f stardist_env.yml
conda env create -f torch_env.yml
```

**Prerequisites for `stardist_env` GPU support:**
- NVIDIA driver ≥ 450.80.02
- CUDA 11.2 system libraries (installed automatically via `cudatoolkit=11.2`)

**Prerequisites for `torch_env` GPU support:**
- NVIDIA driver ≥ 525.0 (for CUDA 12.6)
- PyTorch CUDA libraries are bundled in the pip wheel — no system CUDA install needed

### Kernel Selection

When opening a notebook in VS Code / JupyterLab, select the matching kernel:

- **Notebook 01** → select `stardist_env` kernel
- **Notebooks 02 & 03** → select `torch_env` kernel

## Usage Outline

1. **Update experiment parameters** in each notebook:
   - `IMG_DIR`: Directory containing experimental images
   - `EXP_FOLDER`: Experiment folder name

2. Follow the instructions in each notebook to run the analysis steps sequentially

3. **Output files** are saved as pickle files in the specified output directory

## File Structure

```
TPE_image_process_pipeline/
├── 01. TPE_disk_tracking_stardist.ipynb    # Disk detection & tracking
├── 02. TPE_contact_detect.ipynb            # Contact detection
├── 03. TPE_solve_force_vector_with_ResNet_guess.ipynb  # Force computation
├── environments/
│   ├── stardist_env.yml                    # Notebook 01 — TF 2.10 + StarDist (CUDA 11.2)
│   └── torch_env.yml                       # Notebooks 02 & 03 — PyTorch 2.6 (CUDA 12.6)
├── README.md                                # This file
└── .gitignore                               # Git ignore rules
```

