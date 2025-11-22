# MPI-specific settings
DIR = $(CURDIR)

# Include environment variables
include ./.env

# Enable OpenMP (only for hybrid/grid benchmarks, not baseline)
MACRO += -DOMP_ENABLE

# Enable MPI
MACRO += -DMPI_ENABLE

# Special handling for baseline: disable OpenMP
# The baseline benchmark is pure MPI only and doesn't use OpenMP
# Compiling with OpenMP enabled causes runtime errors when MPI processes
# try to initialize the OpenMP runtime but don't actually use it
BASELINE_MACRO := $(filter-out -DOMP_ENABLE,$(MACRO))
BASELINE_CXXFLAGS := $(filter-out -fopenmp,$(EXTRA_CXXFLAGS))
BASELINE_LDFLAGS := $(filter-out -lomp,$(LDFLAGS))

# MPI flags (detected from mpicxx --showme)
MPI_INCLUDE = -I/usr/include/openmpi-x86_64
MPI_LIBS = -L/usr/lib64/openmpi/lib -lmpi_cxx -lmpi

# Add MPI includes and libs
EXTRA_CXXFLAGS += $(MPI_INCLUDE)
LDFLAGS += $(MPI_LIBS)

# Disable Auto-Vectorize
# EXTRA_CXXFLAGS += -fno-vectorize -fno-slp-vectorize

# Order
# MACRO += -DMATMUL_ORDER=2
# MACRO += -DPRECISION_MODE

# Allow SIMD instruction
EXTRA_CXXFLAGS += -mavx2 -march=native

# SIMD Report
EXTRA_CXXFLAGS += -Rpass=vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
# EXTRA_CXXFLAGS += -fno-slp-vectorize -Rpass-missed=loop-vectorize

# Include parent Makefile
include ../Makefile

# Override compile rule for baseline (no OpenMP)
$(BUILD)/BM_MatMul_MPI_Baseline.exe: $(DIR)/BM_MatMul_MPI_Baseline.cpp $(ROOT)/performance/Makefile $(DIR)/Makefile
	@echo -e "$(YELLOW)Compiling baseline WITHOUT OpenMP...$(RESET)"
	$(CXX) $(WARNING) $(BASELINE_MACRO) -std=c++17 -O3 $(BASELINE_CXXFLAGS) $(MPI_INCLUDE) $(INCLUDES) -MMD -MP $< -o $@ -lbenchmark -lpthread $(MPI_LIBS)
