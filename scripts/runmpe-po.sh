#!/bin/bash

# envs=(15x15-4p-5f-v1 15x15-3p-5f-v1 15x15-4p-3f-v1)
envs=(SimpleSpread-v1 SimpleTag-v1)

for e in "${envs[@]}"
do
    if [[ $e = "SimpleTag-v1" ]]; then
        arg='env_args.pretrained_wrapper="PretrainedTag"'
    else
        arg=""
    fi

    # for a in dva2c inda2c maa2c
    for a in maa2c
    do
        for i in {1..2}
        do
            python src/main.py --config="$a" --env-config=gymma with env_args.key="mpe:$e" "${arg}" &
            echo "Running with $a and mpe:$e"
            sleep 2s
        done
        wait
    done

    # for i in {1..5}
    # do
    #     python src/main.py --config=dva2c --env-config=gymma with env_args.key="mpe:$e" "${arg}" networked_policy=True &
    #     echo "Running with $a and mpe:$e"
    #     sleep 2s
    # done
    # wait
done
