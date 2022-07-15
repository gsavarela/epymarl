#!/bin/bash

# envs=(8x8-2p-2f-coop-v1 2s-8x8-2p-2f-coop-v1 15x15-4p-3f 15x15-4p-5f-v1 2s-10x10-3p-3f-v1 10x10-3p-3f-v1)
# envs=(8x8-2p-2f-coop-v1 2s-8x8-2p-2f-coop-v1 15x15-4p-3f-v1 15x15-4p-5f-v1 15x15-3p-5f-v1 2s-10x10-3p-3f-v1 10x10-3p-3f-v1)

# control, control+coop, distributed, distributed+coop
envs=(15x15-3p-5f-v2 15x15-3p-5f-coop-v2 15x15-3p-5f-share-v2 15x15-3p-5f-coop-share-v2)
# seeds=(291174067 392184168 402285178 493194269 503295279)

for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=maa2c_ns --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=128 &
        echo "Running with maa2c_ns and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done
