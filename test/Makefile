include ./.env
DIR = $(CURDIR)

# TODO FIX
ARG += --operator @

# SIMD instruction
CXXFLAGS += -mavx512f -march=native -O3

# Debug mode
# CXXFLAGS += -g 
# MACRO += -DDEBUG_MODE

# OpenMP
MACRO += -DOMP_ENABLE

# Order
MACRO += -DMATMUL_ORDER=2
# MACRO += -DPRECISION_MODE

include ../Makefile
