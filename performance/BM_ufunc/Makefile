include ./.env
DIR = $(CURDIR)

MACRO += -DOMP_ENABLE

# Disable Auto-Vectorize
# CXXFLAGS += -fno-vectorize -fno-slp-vectorize

# Order
# MACRO += -DMATMUL_ORDER=2
# MACRO += -DPRECISION_MODE

# Allow SIMD instruction
CXXFLAGS += -mavx512f -march=native

# SIMD Report
CXXFLAGS += -Rpass=vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
# CXXFLAGS += -fno-slp-vectorize -Rpass-missed=loop-vectorize

include ../Makefile
