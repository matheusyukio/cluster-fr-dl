#!/bin/bash
#SBATCH -o /home/jonas/log_matheus.txt
#SBATCH --partition=gpu_large
#SBATCH -N 1
#SBATCH -J jmtNN
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonasm.targino@gmail.com

source /home/jonas/modulos.sh
python3 main2.py 
