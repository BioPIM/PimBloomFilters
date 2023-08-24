#! /bin/bash

REPEATS=5

for i in $(seq $REPEATS); do
    ./benchmarks/benchmark1 -k 8 -m 30 -n 10000000 -l -r 6
    ./benchmarks/benchmark1 -k 8 -m 31 -n 10000000 -l -r 6
    ./benchmarks/benchmark1 -k 8 -m 32 -n 10000000 -l -r 6
    ./benchmarks/benchmark1 -k 8 -m 33 -n 10000000 -l -r 6
    ./benchmarks/benchmark1 -k 8 -m 30 -n 100000000 -l -r 6
    ./benchmarks/benchmark1 -k 8 -m 31 -n 100000000 -l -r 6
    ./benchmarks/benchmark1 -k 8 -m 32 -n 100000000 -l -r 6
    ./benchmarks/benchmark1 -k 8 -m 33 -n 100000000 -l -r 6
    ./benchmarks/benchmark1 -k 8 -m 30 -n 10000000 -l -r 8
    ./benchmarks/benchmark1 -k 8 -m 31 -n 10000000 -l -r 8
    ./benchmarks/benchmark1 -k 8 -m 32 -n 10000000 -l -r 8
    ./benchmarks/benchmark1 -k 8 -m 33 -n 10000000 -l -r 8
    ./benchmarks/benchmark1 -k 8 -m 30 -n 100000000 -l -r 8
    ./benchmarks/benchmark1 -k 8 -m 31 -n 100000000 -l -r 8
    ./benchmarks/benchmark1 -k 8 -m 32 -n 100000000 -l -r 8
    ./benchmarks/benchmark1 -k 8 -m 33 -n 100000000 -l -r 8
done