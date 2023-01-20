#!/bin/bash

envs=(15x15-3p-5f-v1)

for e in "${envs[@]}"
do
    for i in {1..1}
    do
        python src/main.py --config=ntwa2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with ntwa2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
