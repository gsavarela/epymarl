#!/bin/bash
envs=(SimpleTag-v0)

for e in "${envs[@]}"
do
   for i in {0..4}
   do
      python src/main.py --config=iql_ns --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:$e" env_args.pretrained_wrapper="PretrainedTag" &
      echo "Running with iql_ns and mpe:$e"
      sleep 2s
   done
   wait
done
