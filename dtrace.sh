#! /bin/bash

DATE=$(date --iso-8601=seconds)

mkdir -p traces

dpu-profiling functions -f "__method_start" -f "__method_end" -f "__run_done" -f "__callback_done" -f "__worker_done" -d dpu_copy_to_mrams -d dpu_copy_from_mrams -d dpu_copy_to_wram_for_rank -o traces/$DATE.json -A -- "$@"
# dpu-profiling memory-transfer -- "$@"