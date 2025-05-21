#!/bin/bash

# Default values
n_seeds="10"
seeds=""
epsilon="0.5"
percentile="0.68"
overrides="mbpo_inv_pendulum"
action_optim_lr=0.05
action_optim_steps=200
n_seed_steps=500
parallel=false
n_processes=4  # Default max parallel processes

print_help() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Optional arguments:"
  echo "  --n_seeds N             Default: 10. Run with seeds from 0 to N-1"
  echo "  --seeds LIST            Comma-separated list of seeds (e.g., 0,2,5). Overrides n_seeds"
  echo "  --epsilon VAL           Default: 0.5"
  echo "  --percentile VAL        Default: 0.68"
  echo "  --n_seed_steps VAL      Default: 500"
  echo "  --action_optim_lr VAL   Default: 0.05"
  echo "  --action_optim_steps VAL Default: 200"
  echo "  --overrides NAME        Default: mbpo_inv_pendulum"
  echo "  --parallel              Run experiments in parallel"
  echo "  --n_processes N         Number of parallel jobs (default: 4)"
  echo "  -h, --help              Show this help message"
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
    --n_seed_steps)
      n_seed_steps="$2"
      shift 2
      ;;
    --action_optim_lr)
      action_optim_lr="$2"
      shift 2
      ;;
    --action_optim_steps)
      action_optim_steps="$2"
      shift 2
      ;;
    --overrides)
      overrides="$2"
      shift 2
      ;;
    --parallel)
      parallel=true
      shift
      ;;
    --n_processes)
      n_processes="$2"
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

# Function to run a single experiment
run_experiment() {
  local seed="$1"
  echo "Running with seed=$seed"
  python -m mbrl.examples.main \
    algorithm=ombpo \
    overrides="$overrides" \
    debug_mode=false \
    use_wandb=true \
    seed="$seed" \
    algorithm.percentile="$percentile" \
    algorithm.epsilon="$epsilon" \
    algorithm.action_optim_lr="$action_optim_lr" \
    algorithm.action_optim_steps="$action_optim_steps"
  echo "Finished run with seed=$seed"
}

# Run experiments
if [[ "$parallel" == true ]]; then
  echo "Running in parallel with up to $n_processes concurrent jobs..."
  running_jobs=0
  for seed in "${seed_list[@]}"; do
    run_experiment "$seed" &
    ((running_jobs++))

    if (( running_jobs >= n_processes )); then
      wait -n  # Wait for any one job to finish
      ((running_jobs--))
    fi
  done
  wait  # Wait for remaining jobs
  echo "All parallel runs completed."
else
  for seed in "${seed_list[@]}"; do
    run_experiment "$seed"
  done
  echo "All sequential runs completed."
fi
