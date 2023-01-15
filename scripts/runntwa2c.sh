#!/bin/bash
envs=(rware-tiny-4ag-v1)
for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=ntwa2c --env-config=gymma with env_args.key="$e" &
        echo "Running with ntwa2c and $e"
        sleep 2s
    done
    wait
done
