#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=flowers_tl
#SBATCH --ntasks=4            # núm. de procesos/threads
#SBATCH --time=3:00:00        # ajusta si necesitas más
#SBATCH --partition=cpu       # única partición disponible
#SBATCH --output=flowers_tl.out
#SBATCH --error=flowers_tl.err

echo "=== Entrenando Transfer Learning de Flores ==="

# Activa el entorno conda
source activate tl_flowers

# Ejecuta el entrenamiento
python train_flowers_tl.py

echo "=== ENTRENAMIENTO COMPLETADO ==="
