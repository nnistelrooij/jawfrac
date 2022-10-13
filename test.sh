#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

MEM_LIMIT="12g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create JawFracNet-output

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        --gpus="all" \
        -v $SCRIPTPATH/test:/input/ \
        -v JawFracNet-output:/output/ \
        jawfracnet_processing

docker run --rm \
        -v JawFracNet-output:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.9-slim python3 -c "from pathlib import Path; assert Path('/output/frac.nii.gz').exists()"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm JawFracNet-output