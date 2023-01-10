#!/bin/bash
envs=(SimpleTag-v0)
for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=ia2c_ns --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:$e" &
        echo "Running with ia2c_ns and mpe:$e"
        sleep 2s
    done
    wait
done
