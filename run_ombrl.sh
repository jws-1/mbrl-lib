#!/bin/bash

# Default values
n_seeds="5"
seeds=""
epsilon="0.5"
percentile="0.68"
overrides="mbpo_inv_pendulum"

print_help() {
  echo "Usage: $0 [--n_seeds N | --seeds S1,S2,...] [--epsilon VAL] [--percentile VAL] [--overrides NAME]"
  echo ""
  echo "Optional arguments:"
  echo "  --n_seeds N         Default: 5. Run with seeds from 0 to N-1"
  echo "  --seeds LIST        Comma-separated list of seeds (e.g., 0,2,5). Overrides n_seeds"
  echo "  --epsilon VAL       Default: 0.5"
  echo "  --percentile VAL    Default: 0.68"
  echo "  --overrides NAME    Default: mbpo_inv_pendulum"
  echo "  -h, --help          Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n_seeds)
      n_seeds="$2"
      shift 2
      ;;
    --seeds)
      seeds="$2"
      shift 2
      ;;
    --epsilon)
      epsilon="$2"
      shift 2
      ;;
    --percentile)
      percentile="$2"
      shift 2
      ;;
    --overrides)
      overrides="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

# Determine seed list
if [[ -n "$seeds" ]]; then
  IFS=',' read -ra seed_list <<< "$seeds"
else
  seed_list=()
  for ((i=0; i<n_seeds; i++)); do
    seed_list+=("$i")
  done
fi

# Run experiments
for seed in "${seed_list[@]}"; do
  echo "Running with seed=$seed"
  python -m mbrl.examples.main \
    algorithm=ombpo \
    overrides="$overrides" \
    debug_mode=false \
    use_wandb=true \
    seed="$seed" \
    algorithm.percentile="$percentile" \
    algorithm.epsilon="$epsilon"
  echo "Finished run with seed=$seed"
done
