# Sourced by the job scripts: loads a GNU compiler on bwUniCluster 3.0 (Lmod).
# Check what exists with:  module avail compiler/gnu     or     module spider gnu
# Outside the cluster (no `module` command) this file does nothing.
if command -v module >/dev/null 2>&1; then
    module purge
    # Prefer an explicit version for reproducibility; fall back to the site default.
    module load compiler/gnu/14.2 2>/dev/null || module load compiler/gnu
    module list 2>&1
fi
g++ --version | head -1
