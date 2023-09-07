#! /bin/bash

DATE=$(date --iso-8601=seconds)

mkdir -p traces

dpu-profiling functions -f "__method_start" --f "__method_end" --probe-dpu-async-api -o traces/$DATE.json -A -- "$@"
# dpu-profiling memory-transfer -- "$@"