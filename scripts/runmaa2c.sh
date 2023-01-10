#!/bin/bash

# envs=(15x15-4p-5f-v1 15x15-3p-5f-v1)
envs=(15x15-4p-3f-v1)
for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=maa2c_ns --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=128 &
        echo "Running with maa2c_ns and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
