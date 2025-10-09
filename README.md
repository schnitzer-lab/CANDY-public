# CANDY: Extracting task-relevant preserved dynamics from contrastive aligned neural recordings
**CANDY** (**C**ontrastive **A**ligned **N**eural **DY**namics) is an end-to-end framework that aligns neural and behavioral data using rank-based contrastive learning, adapted for continuous behavioral variables, to project neural activity from different sessions onto a shared low-dimensional embedding space. CANDY fits a shared linear dynamical system to the aligned embeddings, enabling an interpretable model of the conserved temporal structure in the latent space.

# Publication
Jiang, Y.\*, Sheng, K.\*, Gao, Y., Buchanan, K., Shikano, Y., Kim, T.H., Zhao, Y., Woo, S.J., Dinc, F., Linderman, S.W., Schnitzer, M.J., **Extracting task-relevant preserved dynamics from contrastive aligned neural recordings.** *NeurIPS 2025 (spotlight).* [paper]

\* equal contribution

# Quick Start
## Installation

### Recommended: Conda Environment
Use the provided environment file with the core dependencies needed by CANDY:

```bash
conda env create -f environment.yml
conda activate candy
```

### Alternative: CPU-only Installation
For users without GPU support:

```bash
conda env create -f environment_cpu.yml
conda activate candy-cpu
```

### Manual Installation
For custom installations or if you prefer pip:

```bash
# Create a new conda environment
conda create -n candy python=3.10
conda activate candy

# Install PyTorch (choose your version)
# For GPU users:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# For CPU users:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
conda install numpy pandas scikit-learn matplotlib scipy pyyaml threadpoolctl -c conda-forge
pip install torchmetrics yacs wandb pynapple pynwb h5py hdmf
```

**Core Dependencies:**
- `torch` + `torchvision` + `torchaudio`: PyTorch deep learning framework
- `numpy` + `pandas`: Data manipulation and numerical computing  
- `scikit-learn`: Machine learning utilities (preprocessing, train/test splits)
- `matplotlib`: Plotting and visualization
- `scipy`: Scientific computing utilities
- `yacs`: Configuration management
- `torchmetrics`: Metrics computation
- `wandb`: Experiment tracking (optional)
- `pyyaml`: YAML configuration file parsing
- `pynapple` + `pynwb`: Neuroscience data format support
- `threadpoolctl`: Threading control for linear decoders

## Usage

CANDY provides two main training scripts for different use cases:

### Basic Training

Train a CANDY model from scratch using the main training script:

```bash
python train.py \
    --model_type CANDY \
    --model_config ./config/model_config/candy_cbatch64_ctemp0.2_cs0.1_cts0.yaml \
    --data_config ./config/data_config/mouse_wheel.yaml \
    --decoder_config ./config/decoder_config/linear.yaml \
    --latent_dim 8 \
    --model_seed 0 \
    --save_path ./results/candy_experiment \
    --brain_area str \
    --data_folder /path/to/your/data
```

**Key Arguments:**
- `--model_type`: Model architecture to use (currently supports `CANDY`)
- `--model_config`: Path to model configuration YAML file
- `--data_config`: Path to dataset configuration YAML file  
- `--decoder_config`: Path to behavior decoder configuration YAML file
- `--latent_dim`: Dimensionality of the latent embedding space
- `--model_seed`: Random seed for reproducibility
- `--save_path`: Directory to save model checkpoints and results
- `--brain_area`: Brain region of interest (for mouse data) or task type (for monkey data)
- `--data_folder`: Root directory containing your neural data

**Optional Arguments:**
- `--behv_sup_off`: Disable behavior supervision (contrastive learning only)
- `--contrastive_off`: Disable contrastive learning (behavior supervision only)
- `--train_frac`: Fraction of training data to use (default: 1.0)
- `--not_save_latent`: Skip saving latent representations

### Fine-tuning Training

For fine-tuning a pre-trained model or advanced training scenarios:

```bash
python fine_tuning_train.py \
    --model_type CANDY \
    --model_config ./config/model_config/candy_cbatch64_ctemp0.2_cs0.1_cts0.yaml \
    --data_config ./config/data_config/mouse_wheel.yaml \
    --latent_dim 8 \
    --model_seed 0 \
    --save_path ./results/candy_finetuned \
    --brain_area str \
    --data_folder /path/to/your/data
```

### Configuration Files

CANDY uses YAML configuration files to specify model parameters, data settings, and decoder configurations:

#### Model Configuration
Defines CANDY-specific hyperparameters (see `config/model_config/`):
- Learning rate, regularization, and training epochs
- Contrastive learning parameters (temperature, scaling)
- Network architecture (hidden layers, activation functions)
- Behavior supervision settings

#### Data Configuration  
Specifies dataset parameters (see `config/data_config/`):
- Data paths and subject information
- Trial length constraints and preprocessing options
- Train/validation/test split ratios
- Normalization methods for neural and behavioral data

#### Decoder Configuration
Sets behavior decoding parameters (see `config/decoder_config/`):
- Linear (`linear.yaml`) or nonlinear (`onelayer.yaml`) decoders
- Hyperparameter grids for cross-validation
- Training procedures and regularization

### Example Workflows

**Mouse wheel task with striatal data:**
```bash
python train.py \
    --model_type CANDY \
    --model_config ./config/model_config/candy_cbatch64_ctemp0.5_cs0.5_cts0.yaml \
    --data_config ./config/data_config/mouse_wheel.yaml \
    --latent_dim 8 \
    --brain_area str \
    --save_path ./results/mouse_str_experiment
```

**Monkey reaching task:**
```bash
python train.py \
    --model_type CANDY \
    --model_config ./config/model_config/candy_cbatch64_ctemp0.2_cs0.1_cts0.yaml \
    --data_config ./config/data_config/perich_monkey.yaml \
    --latent_dim 8 \
    --brain_area CO \
    --save_path ./results/monkey_co_experiment
```

### Output

CANDY generates the following outputs in the specified save directory:
- **Checkpoints**: Trained model parameters saved as `.pth` files
- **Latent representations**: Low-dimensional embeddings for each session
- **Training logs**: Loss curves and metrics during training
- **Behavior decoding results**: Performance metrics for behavioral predictions
- **Visualizations**: Plots of latent dynamics and training progress

## Tutorial


# Citation

If you use **CANDY** in your research or build upon it, please cite:

```bibtex
@inproceedings{jiang2025candy,
  title={Extracting task-relevant preserved dynamics from contrastive aligned neural recordings},
  author={Jiang, Y. and Sheng, K. and Gao, Y. and Buchanan, K. and Shikano, Y. and Kim, T. H. and Zhao, Y. and Woo, S. J. and Dinc, F. and Linderman, S. W. and Schnitzer, M. J.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

