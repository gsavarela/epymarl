#!/bin/bash
envs=(10x10-2p-2f-coop-v2 15x15-2p-2f-coop-v2 10x10-3p-3f-coop-v2)
models=(inda2c dsta2c ntwa2c)

for e in "${envs[@]}"
do
  for m in "${models[@]}"
  do
    for i in {1..5}
    do
        python src/main.py --config="$m" --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=128 &
        echo "Running with $m and lbforaging:Foraging-$e"
        sleep 2s
    done
  done
done
