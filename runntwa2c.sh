#!/bin/bash

# envs=(15x15-3p-5f-v1 2s-10x10-3p-3f-v1 10x10-2p-2f-coop-v1)
envs=(rware-tiny-4ag-v1)

for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=ntwa2c --env-config=gymma with env_args.key="rware:$e" env_args.time_limit=500 hidden_dim=64  &
        echo "Running ntwa2c on rware:$e"
        sleep 2s
    done
    wait
done
