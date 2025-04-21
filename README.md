---

## Deep Learning 3D Reconstruction Notebook

This Jupyter Notebook implements a convolutional pipeline for 3D volumetric reconstruction. Given multi‑view RGB inputs, it learns to predict a voxel grid representation of an object.

### 🚀 Project Goal
Train a deep neural network that, for each batch of input images shaped `[B, 3, H, W]`, outputs a volumetric occupancy grid. A composite loss (reconstruction + regularization) ensures accurate, smooth 3D shapes.

---

### 📑 Notebook Sections

1. **Project Goal**  
   High‑level description of objectives and evaluation metrics.

2. **Dataset Acquisition and Preparation**  
   - Mount Google Drive (Colab)  
   - Download and cache multi‑view RGB images and ground‑truth voxel data  
   - Custom dataset class with on‑the‑fly transforms  

3. **Reconstruction Network**  
   - **Configuration Management** (`cfg` dict): hyperparameters & paths  
   - **Data Loading**: `VoxelDataset` plus caching logic  
   - **Model Architecture**: 3D‑conv decoder paired with 2D‑conv encoder  
   - **Loss Composition**: Binary cross‑entropy, IoU, plus weight penalties  
   - **Visualization Helpers**: Live plots via Matplotlib & Plotly  
   - **Training Loop**: `train_model(cfg)` with mixed precision & TensorBoard  

4. **Experiment Entry Point**  
   Script‑style launcher to swap configs, resume checkpoints, or run ablations.

5. **Inference & Final Results**  
   - Load best checkpoint  
   - Generate voxel predictions  
   - Render meshes with Trimesh & Plotly for interactive viewing  

---

### ⚙️ Prerequisites

- **Python** ≥ 3.8  
- **PyTorch** & **torchvision**  
- **NumPy**, **Pillow**  
- **trimesh**  
- **plotly**, **matplotlib**  
- **tensorboard**

Install via:
```bash
pip install torch torchvision numpy pillow trimesh plotly matplotlib tensorboard
```

---

### ▶️ Usage

1. **Clone & open** the notebook:
   ```bash
   git clone https://github.com/mattibuzzo13/Deep-Learning-Projectwork.git
   cd Deep-Learning-Projectwork
   jupyter notebook Projectwork_Deepl_Learning.ipynb
   ```
2. **Run cells in order**:
   - Sections 1–2: data setup  
   - Section 3: model definition & training  
   - Section 4: experiment runner  
   - Section 5: inference & visualization  

3. **Monitor** training metrics:
   ```bash
   tensorboard --logdir runs/
   ```

4. **Customize** by editing the top‑cell `cfg` dictionary or passing flags in Section 4.

---

> **Tip:** For a fully persistent Colab workflow, mount your Google Drive at the start to store data and checkpoints seamlessly.
