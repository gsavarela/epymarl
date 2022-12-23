#!/bin/bash

# envs=(8x8-2p-2f-coop-v1 2s-8x8-2p-2f-coop-v1 15x15-4p-3f 15x15-4p-5f-v1 2s-10x10-3p-3f-v1 10x10-3p-3f-v1)
# control, control+coop, partial_obs, partial_obs+coop, coop
# envs=(15x15-3p-5f-v2 15x15-3p-5f-coop-v2 2s-15x15-3p-5f-v2 2s-15x15-3p-5f-coop-v2)
# seeds=(291174067 392184168 402285178 493194269 503295279)
# envs=(2s-10x10-3p-3f-v1 10x10-3p-3f-v1)
envs=(rware-tiny-4ag-v1)

for e in "${envs[@]}"
do
    for i in {1..5}
    do
        python src/main.py --config=ia2c_ns --env-config=gymma with env_args.key="rware:$e" env_args.time_limit=500 hidden_dim=64 &
        echo "Running ia2c_ns on rware:$e"
        sleep 2s
    done
    wait
done
