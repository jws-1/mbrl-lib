#!/bin/bash

# List of seeds to run
seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
  echo "Running with seed=$seed"
  python -m mbrl.examples.main algorithm=mbpo overrides=mbpo_inv_pendulum debug_mode=true use_wandb=true seed=$seed
  echo "Finished run with seed=$seed"
done
