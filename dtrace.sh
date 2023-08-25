#! /bin/bash

DATE=$(date --iso-8601=seconds)

mkdir -p traces

dpu-profiling functions -f "_trace_rank_done" -d dpu_copy_to_mrams -d dpu_copy_from_mrams -d dpu_copy_to_wram_for_rank -o traces/$DATE.json -A -- "$@"
# dpu-profiling memory-transfer -- "$@"