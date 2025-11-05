CXX			:= clang++
EXECUTABLE	:= main.exe

# Makefile
ROOT 		:= $(abspath $(dir $(DIR)))
BUILD		:= $(DIR)/build

# OS
OS			:= $(shell uname -s)

# OpenMP
ifeq ($(OS), Darwin) # MacOS
	OMP_CONFIG	:= $(ROOT)/lib/omp-config.exe
	ifneq ("$(wildcard $(OMP_CONFIG))", "") # If exists
		INCLUDES	+= $(shell $(OMP_CONFIG) --includes)
		CXXFLAGS	+= $(shell $(OMP_CONFIG) --cflags)
		LDFLAGS		+= $(shell $(OMP_CONFIG) --ldflags)
	else
		INCLUDES	+= -I/usr/local/opt/libomp/include
		CXXFLAGS	+= -Xpreprocessor -fopenmp
		LDFLAGS		+= -lomp -L/usr/local/opt/libomp/lib
	endif
else # Linux distribution
	CXXFLAGS	+= -fopenmp
	LDFLAGS		+= 
endif

# Arguments
MACRO		+= $(EXTRA_MACRO)
INCLUDES	+= -I$(ROOT)/includes
CXXFLAGS	+= -std=c++23 $(EXTRA_CXXFLAGS)
WARNING		+= -Wall -Wextra
LDFLAGS		+= 

SRC_FILES	+= $(wildcard $(DIR)/*.cpp)
OBJ_FILES	+= $(patsubst $(DIR)/%.cpp, $(BUILD)/%.o, $(SRC_FILES))
DEPENDS 	:= $(patsubst $(DIR)/%.cpp, $(BUILD)/%.d, $(SRC_FILES))
EXE_FILES	:= $(BUILD)/$(EXECUTABLE)

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
	@echo -e " OBJ_FILES:" $(OBJ_FILES)
	@echo -e "        OS:" $(OS)
	@echo -e "  INCLUDES:" $(INCLUDES)
	@echo -e "  CXXFLAGS:" $(CXXFLAGS)
	@echo -e "   LDFLAGS:" $(LDFLAGS)
	@echo -e "   DEPENDS:" $(DEPENDS)

$(EXE_FILES): $(OBJ_FILES)
	$(CXX) $(WARNING) $(LDFLAGS) $^ -o $@

$(BUILD)/%.o: $(DIR)/%.cpp $(ROOT)/Makefile $(DIR)/Makefile
	$(CXX) $(WARNING) $(MACRO) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

.PHONY: compile
compile: _directories $(EXE_FILES) ### Compile
	
.PHONY: execute
execute: compile ### Execute
	@echo -e $(GREEN_LINE)
	@echo -e "Executing $(GREEN)$(EXE_FILES)$(RESET)"
	@echo -e $(GREEN_LINE)
	$(EXE_FILES) $(ARG)

.PHONY: _directories
_directories: ### Directories
	@mkdir -p $(BUILD)

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

