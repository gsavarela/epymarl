#!/bin/bash

envs=(2s-10x10-3p-3f-v1)
for e in "${envs[@]}"
do
    for i in {1..2}
    do
        python src/main.py --config=gcna2c_ns --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with gcna2c_ns and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait

    for i in {1..3}
    do
        python src/main.py --config=dva2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with dva2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
