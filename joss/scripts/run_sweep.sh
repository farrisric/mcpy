#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Sweep over delta_mu_H values for all systems (surfaces + nanoparticles).
#
# Usage:
#   bash run_sweep.sh <model_path> <device>
#
# For RE-GCMC runs (requires MPI with 6 ranks for the temperature ladder):
#   bash run_sweep.sh <model_path> <device> --regcmc
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL_PATH="${1:?Usage: bash run_sweep.sh <model_path> <device> [--regcmc]}"
DEVICE="${2:?Usage: bash run_sweep.sh <model_path> <device> [--regcmc]}"
MODE="${3:-gcmc}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Delta mu_H sweep values (eV)
DELTA_MU_H_VALUES=(-1.0 -0.8 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0)

# Surface types
SURFACE_TYPES=(111 100)

# Nanoparticle sizes
NANO_SIZES=(small large)

if [ "$MODE" == "--regcmc" ]; then
    N_REPLICAS=6

    echo "=== Running RE-GCMC sweep ==="
    for surf in "${SURFACE_TYPES[@]}"; do
        for dmu in "${DELTA_MU_H_VALUES[@]}"; do
            echo "  RE-GCMC surface ${surf}, delta_mu_H = ${dmu}"
            mpirun -np "$N_REPLICAS" python "$SCRIPT_DIR/regcmc_surface.py" \
                "$surf" "$dmu" "$MODEL_PATH" "$DEVICE"
        done
    done

    for size in "${NANO_SIZES[@]}"; do
        for dmu in "${DELTA_MU_H_VALUES[@]}"; do
            echo "  RE-GCMC nano ${size}, delta_mu_H = ${dmu}"
            mpirun -np "$N_REPLICAS" python "$SCRIPT_DIR/regcmc_nano.py" \
                "$size" "$dmu" "$MODEL_PATH" "$DEVICE"
        done
    done

else
    echo "=== Running plain GCMC sweep ==="
    for surf in "${SURFACE_TYPES[@]}"; do
        for dmu in "${DELTA_MU_H_VALUES[@]}"; do
            echo "  GCMC surface ${surf}, delta_mu_H = ${dmu}"
            python "$SCRIPT_DIR/gcmc_surface.py" \
                "$surf" "$dmu" "$MODEL_PATH" "$DEVICE"
        done
    done

    for size in "${NANO_SIZES[@]}"; do
        for dmu in "${DELTA_MU_H_VALUES[@]}"; do
            echo "  GCMC nano ${size}, delta_mu_H = ${dmu}"
            python "$SCRIPT_DIR/gcmc_nano.py" \
                "$size" "$dmu" "$MODEL_PATH" "$DEVICE"
        done
    done
fi

echo "=== Sweep complete ==="
