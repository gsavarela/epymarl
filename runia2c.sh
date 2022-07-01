#!/bin/bash

envs=(15x15-4p-3f 8x8-2p-2f-2s-c 10x10-3p-3f-2s 8x8-2p-2f-c 15x15-4p-5f 15x15-3p-5f 10x10-3p-3f)

for e in "${envs[@]}"
do
   for i in {291174067 392184168 402285178 493194269 503295279}
   do
      python src/main.py --config=iac --env-config=gymma with env_args.key="lbforaging:Foraging-$e-v2" seed=$i &
      echo "Running with $1 and lbforaging:Foraging-$e-v2 for seed=$i"
      sleep 2s
   done
   wait
done







