# PIM Bloom filters

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