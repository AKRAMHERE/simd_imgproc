#!/bin/bash
# OpenMP scaling: 1,2,4,8,16,32 threads on one node, threads pinned to physical cores.
#   sbatch --partition=cpu_il slurm/03_bench_scaling.sbatch
#   sbatch --partition=cpu    slurm/03_bench_scaling.sbatch
# For clean numbers use --exclusive (the node is yours; otherwise neighbours share the memory bandwidth).
#SBATCH --job-name=simdip-scale
#SBATCH --partition=cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24000mb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh

PART="${SLURM_JOB_PARTITION:-local}"
[[ -x build/simdip_bench ]] || make -j 4
R="results/${PART}"
mkdir -p "${R}"

export OMP_PROC_BIND=close      # keep thread i next to thread i-1 ...
export OMP_PLACES=cores         # ... and one thread per physical core (no hyper-thread sharing)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

{
    echo "date:       $(date -Is)"
    echo "host:       $(hostname)"
    echo "partition:  ${PART}"
    echo "cpus:       ${SLURM_CPUS_PER_TASK}"
    echo "OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES}"
    echo "cpu:        $(lscpu | grep -m1 'Model name' | sed 's/.*: *//')"
    echo "compiler:   $(g++ --version | head -1)"
    lscpu | grep -E 'Socket|Thread|Core|NUMA' || true
} > "${R}/meta_scaling.txt"
cat "${R}/meta_scaling.txt"

# --cpus-per-task must be given to srun explicitly on some Slurm versions
srun --cpu-bind=none --cpus-per-task="${SLURM_CPUS_PER_TASK}" build/simdip_bench \
     --op gray,sobel,conv --impl scalar,avx2 --sizes 2048,4096,8192 --ksizes 5,9 \
     --threads 1,2,4,8,16,32 --reps 7 --label E7 --csv "${R}/E7_scaling.csv" > "${R}/E7_scaling.txt"
srun --cpu-bind=none --cpus-per-task="${SLURM_CPUS_PER_TASK}" build/simdip_bench \
     --op copy --sizes 4096,8192 --threads 1,2,4,8,16,32 --reps 7 --label E7copy --csv "${R}/E7_copy_scaling.csv" > /dev/null
echo "done: ${R}/E7_scaling.csv"
