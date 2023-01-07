#!/bin/bash
# envs=(15x15-4p-3f 8x8-2p-2f-2s-c 10x10-3p-3f-2s 8x8-2p-2f-c 15x15-4p-5f 15x15-3p-5f 10x10-3p-3f)
envs=(rware-tiny-4ag-v1)

for e in "${envs[@]}"
do
   for i in {0..4}
   do
      # python src/main.py --config=iql --env-config=gymma with env_args.key="lbforaging:Foraging-$e-v2" seed=$i &
      python src/main.py --config=iql_ns --env-config=gymma with env_args.key="$e" hidden_dim=64 env_args.time_limit=500 &
      echo "Running with $1 and $e"
      sleep 2s
   done
   wait
done
