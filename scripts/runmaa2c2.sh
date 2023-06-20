#!/bin/bash

envs=(2s-15x15-4p-5f-v1)
for e in "${envs[@]}"
do
    for i in {1..2}
    do
        python src/main.py --config=maa2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with maa2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait

    for i in {1..3}
    do
        python src/main.py --config=maa2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with maa2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
