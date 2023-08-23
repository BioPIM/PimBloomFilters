#! /bin/bash

REPEATS=3

for i in $(seq $REPEATS); do
    ./benchmarks/benchmark2 -k 8 -m 30 -n 10000000 -l
    ./benchmarks/benchmark2 -k 8 -m 31 -n 10000000 -l
    ./benchmarks/benchmark2 -k 8 -m 32 -n 10000000 -l
    ./benchmarks/benchmark2 -k 8 -m 33 -n 10000000 -l
    ./benchmarks/benchmark2 -k 8 -m 30 -n 100000000 -l
    ./benchmarks/benchmark2 -k 8 -m 31 -n 100000000 -l
    ./benchmarks/benchmark2 -k 8 -m 32 -n 100000000 -l
    ./benchmarks/benchmark2 -k 8 -m 33 -n 100000000 -l
done