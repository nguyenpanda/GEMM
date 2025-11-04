# OpenMP Introduction

This repository by `nguyenpanda` contains `C/C++` code and examples for the `Parallel Computing` course taught by `Dr. Thanh-Dang Diep`, who is a lecturer at `Ho Chi Minh City University of Technology (HCMUT), VNU-HCM`.

## Build & Run

```bash
cd <dir> # e.g., performance, test, ...
make execute
```

Tip: run `make help` to see all available targets defined in each `Makefile`.

## Repository layout

```bash
repo_root
├── performance
│   ├── .env
│   ├── main.cpp
│   └── Makefile
│   ...
│   ...
├── <dir-num>
│   ├── .env
│   ├── main.cpp
│   └── Makefile
├── includes
│   ├── matrix.h
│   ├── nguyenpanda.h
│   └── omp_intro.h
├── Makefile # top-level (primary) Makefile
├── README.md
└── template
    ├── .env
    ├── main.cpp
    └── Makefile
```

- The top-level `repo_root/Makefile` is the primary entry point.

- To override `OpenMP` or `runtime` environment variables for a specific exercise, edit `.env` file.

- To add or tweak compile flags (e.g., `CXXFLAGS`, `LDFLAGS`, `INCLUDES`), edit `Makefile` in `<dir>`. Or adding via command line while compile using `EXTRA_CXXFLAGS` or `EXTRA_MACRO` (e.g., `cd performance && make execute EXECUTE=build/BM_MatAdd.exe EXTRA_CXXFLAGS="-O3 -Ofast -Wall"`)

## Directories

A quick map of what each directory (`<dir>`) demonstrates:

| **Name** | **Description** | **Note** |
|:--------:|-----------------|----------|
| include | Contains the core of this repo | The implementation is done in header file, because of `C++` template |
| performance | All benchmarks are done in this repo | Using `Google Benchmark` and contains its own  |
| test | Testing the correctness of algorithm | Using `Numpy` |
