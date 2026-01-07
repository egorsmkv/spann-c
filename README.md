# SPANN C Demo

Minimal C/ASM demo of a SPANN-style disk-resident ANN search with an AVX2 L2 distance kernel.

## What it does
- Generates or loads a 5M-vector dataset on disk (`spann_5m_data.bin`).
- Builds simple centroid metadata and scans selected posting lists.
- Uses an AVX2 assembly kernel for fast L2 distance (`l2_kernel.asm`).

## Build
Requires `clang` and `nasm` with AVX2 support.

```bash
nasm -f elf64 -O3 l2_kernel.asm -o l2_kernel.o
clang -O3 -mavx2 -mfma main.c l2_kernel.o -o spann_demo -lm
```

## Run
```bash
./spann_demo
```

```
--- SPANN Optimized (AVX2) 5M Vector Demo ---
Target: 5,000,000 Vectors
Estimated Disk Usage: ~2.56 GB

Detected existing data file: spann_5m_data.bin. Loading metadata...
Metadata loaded successfully. Skipping data generation.

Executing search against 5M vectors (Disk-Resident)...

Search Results:
Nearest Neighbor ID: 47048
L2 Distance:         16.628300
Total Latency:       63.722 ms
```

## Notes
- First run will generate `spann_5m_data.bin` (~2.56 GB). Subsequent runs reuse it.
- The demo parameters are in `main.c` (dim=128, centroids=100, 50k vectors per list).

## Performance and tuning
- Reduce `num_centroids` or `vectors_per_list` to shrink disk usage and speed up the first run.
- The search scans up to 32 posting lists; change the cap in `search_spann` if you want more recall at the cost of latency.
- `epsilon_2` in `search_spann` controls the candidate threshold; larger values include more lists.
- Build with `-O3 -mavx2 -mfma` and run on a CPU that supports AVX2 for best performance.

## Code style (clang-format)
If you want to improve or standardize formatting, run `clang-format` on the C sources.

```bash
clang-format -i main.c
```

Optional: generate a starting config, then tune it to your preferences.

```bash
clang-format -style=LLVM -dump-config > .clang-format
```

## Files
- `main.c`: SPANN demo and disk-scanning search.
- `l2_kernel.asm`: AVX2 L2 distance kernel.
- `spann_5m_data.bin`: Generated dataset (large binary file).
