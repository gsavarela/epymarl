#!/bin/bash
envs=(rware-tiny-4ag-v1)

for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=vdn_ns --env-config=gymma with env_args.time_limit=500 env_args.key="$e" &
        echo "Running with vdn_ns and rware:$e"
        sleep 2s
    done
    wait
done
