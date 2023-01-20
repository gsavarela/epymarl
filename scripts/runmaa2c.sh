#!/bin/bash
envs=(15x15-3p-5f-v1)

for e in "${envs[@]}"
do
    for i in {1..1}
    do
        python src/main.py --config=maa2c_ns --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with maa2c_ns and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
