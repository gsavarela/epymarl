#!/bin/bash
# TODO: FIX PARTIALLY OBSERVABLE
# envs=(10x10-2p-2f-coop-v2 2s-10x10-3p-3f-v2 15x15-3p-5f-v2)
envs=(10x10-2p-2f-coop-v2 15x15-3p-5f-v2)

for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=maa2c_baseline --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with maa2c_baseline and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
