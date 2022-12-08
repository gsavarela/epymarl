#!/bin/bash
envs=(15x15-3p-5f-v1)

for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=inda2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=64 &
        echo "Running with inda2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
