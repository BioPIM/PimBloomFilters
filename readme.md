# PIM Bloom filters

## Requirements

[Spdlog logging library](https://github.com/gabime/spdlog)

## How to build

```bash
mkdir build
cd build
cmake ..
make -j
```

## How to run

```bash
./tests/unit_test1 # Unit tests for PIM filter
./tests/unit_test2 # Unit tests for standard filters (basic, cache and sync variants)
./benchmarks/benchmark1 -h # Benchmark for PIM filter
./benchmarks/benchmark2 -h # Benchmark for sync-cache filter
```

## DPU Traces

1. Run a command with `dtrace.sh`

```bash
./dtrace.sh ./benchmarks/benchmark1 -k 8 -m 30 -n 10000000 -l -r 6
```

2. Explore the `traces/xxxxx.json` file with [Perfetto UI](https://ui.perfetto.dev/)