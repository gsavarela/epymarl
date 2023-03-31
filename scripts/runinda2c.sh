#!/bin/bash
envs=(mpe:SimpleSpread-v0)

for e in "${envs[@]}"
do
    for i in {1..3}
    do
        python src/main.py --config=inda2c --env-config=gymma with env_args.key="$e" hidden_dim=128 &
        echo "Running with inda2c and $e"
        sleep 2s
    done
    wait
done
