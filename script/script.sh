#!/usr/bin/bash
#SBATCH --job-name=mouse_wheel-candy_cbatch64_ctemp0.2_cs0.1_cts0-latent_dim64-seed4
#SBATCH --output=cbatch64_ctemp0.2_cs0.1_cts0/mouse_wheel-candy_cbatch64_ctemp0.2_cs0.1_cts0-latent_dim64-seed4_job.%j.out
#SBATCH --error=cbatch64_ctemp0.2_cs0.1_cts0/mouse_wheel-candy_cbatch64_ctemp0.2_cs0.1_cts0-latent_dim64-seed4_job.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=owners
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=180GB

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

source ~/.bashrc
conda activate py3.10
python3 ../train.py --behv_sup_off --save_path ../results/cbatch64_ctemp0.2_cs0.1_cts0_behvOFF --latent_dim 64 --model_seed 4 --model_config ../config/model_config/candy_cbatch64_ctemp0.2_cs0.1_cts0.yaml --data_folder ../../shared-dynamics/ --data_config ../config/data_config/mouse_wheel.yaml --decoder_config ../config/decoder_config/linear.yaml