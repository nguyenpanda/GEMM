CXX			:= clang++
EXECUTE		:= 

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
	LDFLAGS		+= -fopenmp
endif

# Arguments
MACRO		+= $(EXTRA_MACRO)
CXXFLAGS	+= $(EXTRA_CXXFLAGS) --std=c++23
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
@echo $(GREEN_LINE)
@echo "Executing $(GREEN)$(1)$(RESET)"
@echo $(GREEN_LINE)
$(eval BENCHMARK_OUT_FILE := $(LOGGING)/$(basename $(notdir $(1)))-$(shell date +"%y-%m-%d_%H-%M-%S").$(BENCHMARK_OUT_FORMAT))
$(abspath $(1)) --benchmark_out="$(BENCHMARK_OUT_FILE)"
endef

.PHONY: help
help: ### Show this help message
	@echo $(HELP_MSG)
	@awk -F ':.*###' '$$0 ~ FS {printf "$(GREEN)%15s$(RESET)%s\n", $$1 ":", $$2}' \
		$(MAKEFILE_LIST) | grep -v '@awk' | sort

.PHONY: test
test: ### Use for debugging
	@echo $(GREEN_LINE)
	@echo "      ROOT:" $(ROOT)
	@echo "       DIR:" $(DIR)
	@echo "     BUILD:" $(BUILD)
	@echo " SRC_FILES:" $(SRC_FILES)
	@echo " EXE_FILES:" $(EXE_FILES)
	@echo "        OS:" $(OS)
	@echo "  INCLUDES:" $(INCLUDES)
	@echo "  CXXFLAGS:" $(CXXFLAGS)
	@echo "   LDFLAGS:" $(LDFLAGS)
	@echo "   DEPENDS:" $(DEPENDS)

$(BUILD)/%.exe: $(DIR)/%.cpp $(ROOT)/performance/Makefile $(DIR)/Makefile
	$(CXX) $(WARNING) $(MACRO) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -MMD -MP $< -o $@

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
	@echo "$(RED)Are you sure you want to delete the directory $(YELLOW)'$(BUILD)'$(RED)? (y/N)$(RESET)"
	@read -p "Type y to confirm: " confirm; \
	if [ "$$confirm" = "y" ]; then \
		echo "$(RED)Deleting $(YELLOW)$(BUILD)$(RESET)"; \
		rm -rf $(BUILD); \
	else \
		echo "$(GREEN)Deletion cancelled.$(RESET)"; \
	fi

