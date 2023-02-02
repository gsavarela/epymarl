#!/bin/bash
envs=(SimpleTag-v0)
sleep 21600
for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=ntwa2c --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:$e" &
        echo "Running with ntwa2c and mpe:-$e"
        sleep 2s
    done
    wait
done
