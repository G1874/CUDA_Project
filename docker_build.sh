#!/bin/bash

echo "Building cuda-evs Docker image..."

build_args=""
for (( i=1; i<=$#; i++));
do
    param="${!i}"
    if [ "$param" == "--build-args" ]; then
        j=$((i+1))
        build_args="${!j}"
    fi
done

docker build $build_args -t cuda-evs -f .devcontainer/Dockerfile .