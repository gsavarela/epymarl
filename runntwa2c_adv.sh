#!/bin/bash
envs=(15x15-4p-5f-v1)

for e in "${envs[@]}"
do
    for i in {1..3}
    do
        python src/main.py --config=ntwa2c_adv --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with ntwa2c_adv and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
