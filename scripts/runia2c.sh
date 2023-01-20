#!/bin/bash
envs=(15x15-3p-5f)

for e in "${envs[@]}"
do
    for i in {1..1}
    do
        python src/main.py --config=ia2c_ns --env-config=gymma with env_args.key="lbforaging:Foraging-$e-v1"  env_args.time_limit=50 &
        echo "Running with ia2c_ns and lbforaging:Foraging-$e-v1"
        sleep 2s
    done
    wait
done
