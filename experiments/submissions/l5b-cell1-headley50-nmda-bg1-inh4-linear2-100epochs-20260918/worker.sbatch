#!/bin/bash
#SBATCH --partition=mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/js3269/projects/l5b_functional_clustering_cell1_headley50
exec /home/js3269/enter/envs/l5b_amarel_neuron/bin/python -u scripts/run_cell1_headley50_unit.py
