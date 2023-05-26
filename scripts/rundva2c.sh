#!/bin/bash
envs=(10x10-3p-3f-v1 15x15-4p-5f-v1 2s-10x10-3p-3f-v1)

# Control
for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=dva2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" &
        echo "Running with dva2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
    # Test: Zhang et al., 2018
    for i in {1..5}
    do
        python src/main.py --config=dva2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" env_args.joint_rewards=False &
        echo "Running with dva2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
