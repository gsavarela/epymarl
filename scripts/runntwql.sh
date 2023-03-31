#!/bin/bash
# envs=(15x15-3p-5f)
envs=(rware-tiny-4ag)

for e in "${envs[@]}"
do
   for i in {1..5}
   do
      # python src/main.py --config=ntwql --env-config=gymma with env_args.key="lbforaging:Foraging-$e-v1" &
      # echo "Running ntwql and lbforaging:Foraging-$e-v1"
      python src/main.py --config=ntwql --env-config=gymma with env_args.key="$e-v1" &
      echo "Running ntwql and rware:$e-v1"
      sleep 2s
   done
   wait
done
