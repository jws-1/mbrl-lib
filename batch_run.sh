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
n_processes=4
use_slurm=false
time="0-4:00"
algorithm="ombpo"

print_help() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Optional arguments:"
  echo "  --n_seeds N               Default: 10"
  echo "  --seeds LIST              Comma-separated list of seeds (e.g., 0,2,5)"
  echo "  --epsilon VAL             Default: 0.5 (only used for ombpo)"
  echo "  --percentile VAL          Default: 0.68 (only used for ombpo)"
  echo "  --n_seed_steps VAL        Default: 500"
  echo "  --action_optim_lr VAL     Default: 0.05 (only used for ombpo)"
  echo "  --action_optim_steps VAL  Default: 200 (only used for ombpo)"
  echo "  --overrides NAME          Default: mbpo_inv_pendulum"
  echo "  --algorithm NAME          Algorithm to use (default: ombpo)"
  echo "  --parallel                Run experiments in parallel"
  echo "  --n_processes N           Max parallel jobs (default: 4)"
  echo "  --use_slurm               Submit jobs using SLURM"
  echo "  --time VAL                SLURM job time (default: 0-4:00)"
  echo "  -h, --help                Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n_seeds) n_seeds="$2"; shift 2 ;;
    --seeds) seeds="$2"; shift 2 ;;
    --epsilon) epsilon="$2"; shift 2 ;;
    --percentile) percentile="$2"; shift 2 ;;
    --n_seed_steps) n_seed_steps="$2"; shift 2 ;;
    --action_optim_lr) action_optim_lr="$2"; shift 2 ;;
    --action_optim_steps) action_optim_steps="$2"; shift 2 ;;
    --overrides) overrides="$2"; shift 2 ;;
    --algorithm) algorithm="$2"; shift 2 ;;
    --parallel) parallel=true; shift ;;
    --n_processes) n_processes="$2"; shift 2 ;;
    --use_slurm) use_slurm=true; shift ;;
    --time) time="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
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
  cmd="python -m mbrl.examples.main \
    algorithm=$algorithm \
    overrides=$overrides \
    debug_mode=false \
    use_wandb=true \
    seed=$seed"

  if [[ "$algorithm" == "ombpo" ]]; then
    cmd+=" \
      algorithm.percentile=$percentile \
      algorithm.epsilon=$epsilon \
      algorithm.action_optim_lr=$action_optim_lr \
      algorithm.action_optim_steps=$action_optim_steps"
  fi

  eval "$cmd"
  echo "Finished run with seed=$seed"
}

# Function to submit SLURM job
submit_slurm_job() {
  local seed="$1"
  local job_script
  job_script=$(mktemp)

  extra_args=""
  if [[ "$algorithm" == "ombpo" ]]; then
    extra_args+=" \
      algorithm.percentile=$percentile \
      algorithm.epsilon=$epsilon \
      algorithm.action_optim_lr=$action_optim_lr \
      algorithm.action_optim_steps=$action_optim_steps"
  fi

  cat > "$job_script" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${algorithm}_${overrides}_${seed}
#SBATCH --output=/scratch/users/%u/${algorithm}_${overrides}_${seed}.out
#SBATCH --error=/scratch/users/%u/${algorithm}_${overrides}_${seed}_err.out
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=$time
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH --exclude=erc-hpc-comp031,erc-hpc-comp049,erc-hpc-comp033
#SBATCH --chdir=/scratch/users/${USER}/

module load cuda
singularity exec --nv /users/${USER}/mbpo-exploration/container.sif bash -c "
python3 -m mbrl.examples.main \
  algorithm=$algorithm \
  overrides=$overrides \
  debug_mode=false \
  use_wandb=true \
  seed=$seed$extra_args"
EOF

  if ! job_output=$(sbatch "$job_script"); then
    echo "Error: Failed to submit SLURM job for seed $seed." >&2
    rm "$job_script"
    exit 1
  fi

  job_id=$(echo "$job_output" | awk '{print $4}')
  echo "Submitted SLURM job for seed $seed with Job ID $job_id"
  rm "$job_script"
}

# Main execution logic
if [[ "$use_slurm" == true ]]; then
  for seed in "${seed_list[@]}"; do
    submit_slurm_job "$seed"
    sleep 1
  done
  echo "All SLURM jobs submitted."
else
  if [[ "$parallel" == true ]]; then
    echo "Running in parallel using $n_processes processes"
    export -f run_experiment
    parallel -j "$n_processes" run_experiment ::: "${seed_list[@]}"
  else
    for seed in "${seed_list[@]}"; do
      run_experiment "$seed"
    done
  fi
  echo "All local experiments completed."
fi
