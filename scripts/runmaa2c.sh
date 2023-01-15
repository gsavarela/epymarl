#!/bin/bash
envs=(rware-tiny-4ag-v1)
for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=maa2c_ns --env-config=gymma with env_args.key="$e" &
        echo "Running with maa2c_ns and $e"
        sleep 2s
    done
    wait
done
