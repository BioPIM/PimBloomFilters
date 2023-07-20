#! /bin/bash

DATE=$(date --iso-8601=seconds)

mkdir -p traces

dpu-profiling functions -f "_worker_done" -f "_trace_rank_done" -o traces/$DATE.json -A -- "$@"