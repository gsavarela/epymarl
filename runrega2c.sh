#!/bin/bash

# envs=(15x15-3p-5f-v1 2s-10x10-3p-3f-v1 10x10-2p-2f-coop-v1)
envs=(15x15-3p-5f-v1)

for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=rega2c_shl --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=128 &
        echo "Running with rega2c_shl and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
