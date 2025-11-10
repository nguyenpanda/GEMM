include ./.env
DIR = $(CURDIR)

MACRO += -DOMP_ENABLE

# Disable Auto-Vectorize
# CXXFLAGS += -fno-vectorize -fno-slp-vectorize

# Order
# MACRO += -DPRECISION_MODE

# Allow SIMD instruction
CXXFLAGS += -mavx2 -march=native

# SIMD Report
CXXFLAGS += -Rpass=vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
# CXXFLAGS += -fno-slp-vectorize -Rpass-missed=loop-vectorize

include ../Makefile
