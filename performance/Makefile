CXX			:= clang++
EXECUTE		:= 
MPI_NP		:= 1
HOSTFILE	:=

# Makefile
ROOT 			:= $(abspath $(dir $(abspath $(dir $(DIR)))))
BUILD			:= $(DIR)/build
LOGGING			:= $(DIR)/log

# OS
OS				:= $(shell uname -s)

# OpenMP
ifeq ($(OS), Darwin) # MacOS
	INCLUDES	+= -I/usr/local/opt/libomp/include
	CXXFLAGS	+= -Xpreprocessor -fopenmp
	LDFLAGS		+= -lomp -L/usr/local/opt/libomp/lib
else # Linux distribution
	CXXFLAGS	+= -fopenmp 
	LDFLAGS		+= -lpthread
endif

# Arguments
MACRO		+= $(EXTRA_MACRO)
CXXFLAGS	+= $(EXTRA_CXXFLAGS) -std=c++23 -O3
INCLUDES	+= -I$(ROOT)/includes
WARNING		+= -Wall -Wextra
LDFLAGS		+= -lbenchmark

SRC_FILES	+= $(wildcard $(DIR)/*.cpp)
DEPENDS 	:= $(patsubst $(DIR)/%.cpp, $(BUILD)/%.d, $(SRC_FILES))
EXE_FILES	:= $(patsubst $(DIR)/%.cpp, $(BUILD)/%.exe, $(SRC_FILES))

RED			:= \033[1;91m
GREEN		:= \033[1;92m
YELLOW		:= \033[1;93m
RESET		:= \033[0m
GREEN_LINE	:= "$(GREEN)----------------------------------$(RESET)"
META_MESS	:= ************ DIR=$(DIR) ************

ifeq ($(DIR), )
$(error DIR is not defined!)
endif
$(info $(META_MESS))

-include $(DEPENDS)

# GOOGLE BENCHMARK
define EXECUTE_MACRO
@echo -e $(GREEN_LINE)
@echo -e "Executing $(GREEN)$(1)$(RESET) with MPI_NP=$(GREEN)$(MPI_NP)$(RESET)"
@echo -e $(GREEN_LINE)
$(eval BENCHMARK_OUT_FILE := $(LOGGING)/$(basename $(notdir $(1)))-$(shell date +"%y-%m-%d_%H-%M-%S").$(BENCHMARK_OUT_FORMAT))
@if [ $(MPI_NP) -gt 1 ]; then \
	if [ -n "$(HOSTFILE)" ]; then \
		mpirun -np $(MPI_NP) --hostfile $(HOSTFILE) $(abspath $(1)) --benchmark_out="$(BENCHMARK_OUT_FILE)"; \
	else \
		mpirun -np $(MPI_NP) $(abspath $(1)) --benchmark_out="$(BENCHMARK_OUT_FILE)"; \
	fi; \
else \
	$(abspath $(1)) --benchmark_out="$(BENCHMARK_OUT_FILE)"; \
fi
endef

.PHONY: help
help: ### Show this help message
	@echo -e $(HELP_MSG)
	@awk -F ':.*###' '$$0 ~ FS {printf "$(GREEN)%15s$(RESET)%s\n", $$1 ":", $$2}' \
		$(MAKEFILE_LIST) | grep -v '@awk' | sort

.PHONY: test
test: ### Use for debugging
	@echo -e $(GREEN_LINE)
	@echo -e "      ROOT:" $(ROOT)
	@echo -e "       DIR:" $(DIR)
	@echo -e "     BUILD:" $(BUILD)
	@echo -e " SRC_FILES:" $(SRC_FILES)
	@echo -e " EXE_FILES:" $(EXE_FILES)
	@echo -e "        OS:" $(OS)
	@echo -e "  INCLUDES:" $(INCLUDES)
	@echo -e "  CXXFLAGS:" $(CXXFLAGS)
	@echo -e "   LDFLAGS:" $(LDFLAGS)
	@echo -e "   DEPENDS:" $(DEPENDS)

$(BUILD)/%.exe: $(DIR)/%.cpp $(ROOT)/performance/Makefile $(DIR)/Makefile
	$(CXX) $(WARNING) $(MACRO) $(CXXFLAGS) $(INCLUDES) -MMD -MP $< -o $@ $(LDFLAGS)

.PHONY: compile
compile: _directories $(EXE_FILES) ### Compile

.PHONY: execute
execute: compile ### Execute
ifneq ($(EXECUTE), )
	$(call EXECUTE_MACRO, $(EXECUTE))
else ifeq ($(EXECUTE), all)
	$(foreach file, $(EXE_FILES), $(call EXECUTE_MACRO, $(file)))
endif

.PHONY: _directories
_directories: ### Directories
	@mkdir -p $(BUILD) $(LOGGING)

.PHONY: clear
clear: ### Clear
	@echo -e "$(RED)Are you sure you want to delete the directory $(YELLOW)'$(BUILD)'$(RED)? (y/N)$(RESET)"
	@read -p "Type y to confirm: " confirm; \
	if [ "$$confirm" = "y" ]; then \
		echo -e "$(RED)Deleting $(YELLOW)$(BUILD)$(RESET)"; \
		rm -rf $(BUILD); \
	else \
		echo -e "$(GREEN)Deletion cancelled.$(RESET)"; \
	fi

