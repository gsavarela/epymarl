#!/bin/bash

envs=(15x15-4p-5f-v2)

for e in "${envs[@]}"
do

    for i in {1..5}
    do
        python src/main.py --config=ntwa2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=128 &
        echo "Running with ntwa2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done

sed -i 's/4: 6/4: 3/g' src/config/algs/ntwa2c.yaml

for e in "${envs[@]}"
do

    for i in {1..5}
    do
        python src/main.py --config=ntwa2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=128 &
        echo "Running with ntwa2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done

sed -i 's/4: 3/4: 0/g' src/config/algs/ntwa2c.yaml

for e in "${envs[@]}"
do

    for i in {1..5}
    do
        python src/main.py --config=ntwa2c --env-config=gymma with env_args.key="lbforaging:Foraging-$e" hidden_dim=128 &
        echo "Running with ntwa2c and lbforaging:Foraging-$e"
        sleep 2s
    done
    wait
done

