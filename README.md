# OpenMP Introduction

This repository by `nguyenpanda` contains `C/C++` code and examples for the `Parallel Computing` course taught by `Dr. Thanh-Dang Diep`, who is a lecturer at `Ho Chi Minh City University of Technology (HCMUT), VNU-HCM`.

## Build & Run

```bash
cd <dir-num> # e.g., 01, 02, ...
make execute
```

Tip: run `make help` to see all available targets defined in each `Makefile`.

## Repository layout

```bash
repo_root
├── 01
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

- To override `OpenMP` environment variables for a specific exercise, edit `.env` file.

- To add or tweak compile flags (e.g., `CXXFLAGS`, `LDFLAGS`, `INCLUDES`), edit `Makefile` in `<dir-num>`.

## Directories

A quick map of what each directory (`<dir-num>`) demonstrates:

| **Name** | **Description** | **Note** |
|:--------:|-----------------|----------|
| 01 | Test common `OpenMP` pragmas/keywords | |
| 02 | 1D vector addition | |
| 03 | Computing `π` (pi) | |
| chapter16 | `Chapter 16: Scheduling and work distribution` from `The Art of Multiprocessor Programming` edition `2` by `Herlihy et al` | |
