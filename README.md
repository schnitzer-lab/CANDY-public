# CANDY: Extracting task-relevant preserved dynamics from contrastive aligned neural recordings
**CANDY** (**C**ontrastive **A**ligned **N**eural **DY**namics) is an end-to-end framework that aligns neural and behavioral data using rank-based contrastive learning, adapted for continuous behavioral variables, to project neural activity from different sessions onto a shared low-dimensional embedding space. CANDY fits a shared linear dynamical system to the aligned embeddings, enabling an interpretable model of the conserved temporal structure in the latent space.

# Publication
Jiang, Y.\*, Sheng, K.\*, Gao, Y., Buchanan, K., Shikano, Y., Kim, T.H., Zhao, Y., Woo, S.J., Dinc, F., Linderman, S.W., Schnitzer, M.J., **Extracting task-relevant preserved dynamics from contrastive aligned neural recordings.** *NeurIPS 2025 (spotlight).* [paper](https://openreview.net/forum?id=uvTea5Rfek)

# Installation
### Recommended Conda enviornmenm
```bash
# Create and activate the GPU environment
conda env create -f environment.yml
conda activate candy

# (Alternative) For CPU version, please use
conda env create -f environment_cpu.yml
conda activate candy-cpu

# Install CANDY as a package
pip install -e .
```

### Alternative: Manual Installation
For custom installations or if you prefer pip:

```bash
# Create a new conda environment
conda create -n candy python=3.10
conda activate candy

# Install PyTorch (choose your version)
# For GPU users: 
# Ensure `pytorch-cuda` matches your NVIDIA driver. For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# For CPU users:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
conda install numpy pandas scikit-learn matplotlib scipy pyyaml threadpoolctl -c conda-forge
pip install torchmetrics yacs wandb pynapple pynwb h5py hdmf

# Install CANDY as a package
pip install -e .
```

## Usage
### Basic Training

Train a CANDY model from scratch using the main training script:

```bash
python train.py \
    --model_type CANDY \
    --model_config ./config/model_config/candy_cbatch64_ctemp0.2_cs0.1_cts0.yaml \
    --data_config ./config/data_config/mouse_wheel.yaml \
    --decoder_config ./config/decoder_config/linear.yaml \
    --data_folder /path/to/your/data
```

**Required:**
- `--model_type`: Model type.
- `--model_config`: Model YAML configuration file path.

**Common options:**
- `--data_config`: Dataset configuration path (default: mouse_wheel.yaml).
- `--decoder_config`: Behavior decoder YAML configuration file path (default: ./config/decoder_config/linear.yaml).
- `--latent_dim`: Latent subspace dimension (default: 8).
- `--model_seed`: Model seed (default: 0).
- `--save_path`: Saving parent folder path (default: ./test).
- `--brain_area`: Brain areas [mouse wheel] or Task type [monkey] (default: None).
- `--data_folder`: Data parent folder (default: G:\My Drive\Research).

**Training toggles:**
- `--behv_sup_off`: Turn off the behavior supervision.
- `--contrastive_off`: Turn off contrastive learning.
- `--train_frac float`: Actual fraction of training data to use (default: 1.0). Useful for comparison of performance as a function of training data size while fixing the testing data.
- `--not_save_latent`: Turn off latent save.

**Multi‑GPU:**
- `--multi_gpu`: Enable multi‑GPU training.
- `--gpu_ids`: Comma-separated GPU IDs to use (e.g., `"0,1,2"`). If not specified, all available GPUs will be used.
- `--data_parallel_type`: Type of data parallelism to use (default: DataParallel).

### Fine-tuning Training

For fine-tuning a pre-trained model or advanced training scenarios:

```bash
python fine_tuning_train.py \
    --model_type CANDY \
    --model_config ./config/model_config/candy_cbatch64_ctemp0.2_cs0.1_cts0.yaml \
    --data_config ./config/data_config/mouse_wheel.yaml \
    --data_folder /path/to/your/data
```

### Demo Scripts

CANDY includes demo scripts for testing and experimentation:

```bash
# Spiral simulation demo
python demo/spiral_simulation/train_simulation.py \
    --ckpt_save_dir ./demo_output \
    --data_path ./demo/spiral_simulation/spiral_data.pkl \

```

### Output

CANDY generates the following outputs in the specified save directory:
- **Checkpoints**: Trained model parameters saved as `.pth` files
- **Latent representations**: Low-dimensional embeddings for each session
- **Training logs**: Loss curves and metrics during training
- **Behavior decoding results**: Performance metrics for behavioral predictions
- **Visualizations**: Plots of latent dynamics and training progress


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

