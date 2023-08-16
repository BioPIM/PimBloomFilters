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
./src/unit_test1 # Unit tests for PIM filter
./src/unit_test2 # Unit tests for standard filters (basic, cache and sync variants)
./src/benchmark1 -h # Benchmarks for PIM filter
./src/benchmark2 -h # Benchmarks for sync-cache filter
```