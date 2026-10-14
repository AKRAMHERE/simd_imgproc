#!/bin/bash
# Submit the benchmark jobs to both CPU partitions you want to compare.
#   bash slurm/submit_all.sh
# Run it from the project root on a bwUniCluster login node, AFTER 01_build_test has passed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
for P in cpu_il cpu; do
    sbatch --partition="${P}" slurm/02_bench_single.sbatch
    sbatch --partition="${P}" slurm/03_bench_scaling.sbatch
done
echo "Check progress with:  squeue --me    (do not run it in a loop; once every 30 s at most)"
