#!/bin/bash
envs=(15x15-4p-5f-v1)
for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=vdn_ns --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=64 env_args.time_limit=50 &
        echo "Running with vdn_ns and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
