#!/bin/bash
#SBATCH --partition=mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --requeue
set -euo pipefail
cd /home/js3269/projects/l5b_functional_clustering_cell1_headley50
exec /home/js3269/enter/envs/l5b_amarel_neuron/bin/python -u scripts/control_cell1_headley50_batch.py "$1"
