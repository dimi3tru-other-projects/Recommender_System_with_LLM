.PHONY: requirements create_environment test_environment detect_platform

################################################################################
# GLOBALS                                                                      #
################################################################################

PROJECT_NAME=Recommender System with LLM
PYTHON_BIN?=python3.10

# 1. Platform‑specific variables
ifeq ($(OS),Windows_NT)                     # Windows
PYTHON_INTERPRETER=venv/Scripts/python.exe
PROJECT_DIR=$(CURDIR)
REQUIREMENTS_FILE=requirements-windows.txt
CUDA_PRESENT=$(shell where nvcc >nul 2>&1 && echo yes || echo no)          # nvcc check
else                                       # Unix (macOS / Linux)
UNAME_S:=$(shell uname -s)
PYTHON_INTERPRETER=$(shell pwd)/venv/bin/python3
PROJECT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
REQUIREMENTS_FILE=requirements-unix.txt
ifeq ($(UNAME_S),Darwin)                   # macOS (no CUDA)
CUDA_PRESENT=no
else                                       # Linux
CUDA_PRESENT=$(shell command -v nvcc >/dev/null 2>&1 && echo yes || echo no)
endif
endif

# 2. PyTorch index selector
PYTORCH_CUDA_VERSION?=cu118                # default CUDA 11.8
ifeq ($(OS),Windows_NT)
TORCH_INDEX:=$(if $(filter yes,$(CUDA_PRESENT)),$(PYTORCH_CUDA_VERSION),cpu)
else
ifeq ($(UNAME_S),Darwin)
TORCH_INDEX:=metal
else
TORCH_INDEX:=$(if $(filter yes,$(CUDA_PRESENT)),$(PYTORCH_CUDA_VERSION),cpu)
endif
endif

# 3. Conda availability
HAS_CONDA:=$(shell conda --version >/dev/null 2>&1 && echo True || echo False)

# 4. kernel name: lowercase, underscores instead of spaces (for .ipynb files)
KERNEL_NAME=$(shell echo $(PROJECT_NAME) | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

################################################################################
# COMMANDS                                                                     #
################################################################################

## Show detected configuration
detect_platform:
	@echo "Project name: $(PROJECT_NAME)"
	@echo "OS:           $(OS) $(UNAME_S)"
	@echo "Interpreter:  $(PYTHON_INTERPRETER)"
	@echo "CUDA present: $(CUDA_PRESENT)"
	@echo "Torch index:  $(TORCH_INDEX)"
	@echo "Has conda:    $(HAS_CONDA)"
	@echo "Req file:     $(REQUIREMENTS_FILE)"
	@echo "Kernel name:  $(KERNEL_NAME)"

## Create venv and upgrade basic tools
# create_environment:
# 	@command -v python3 >/dev/null 2>&1 || (echo "Python3 not found!" && exit 1)
# 	python3 -m venv venv && $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel

create_environment:
	@command -v $(PYTHON_BIN) >/dev/null 2>&1 || (echo "$(PYTHON_BIN) not found!" && exit 1)
	$(PYTHON_BIN) -m venv venv && $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel

## Run sanity‑check script
test_environment:
	$(PYTHON_INTERPRETER) junk/test_environment.py

## Install project requirements and matching PyTorch build
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r $(REQUIREMENTS_FILE)
	@echo "Installing PyTorch from index '$(TORCH_INDEX)'..."
	$(PYTHON_INTERPRETER) -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$(TORCH_INDEX)

## Register current virtual environment as a Jupyter kernel
add_kernel:
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name=$(KERNEL_NAME) --display-name="Python (venv: $(PROJECT_NAME))"

## Remove registered Jupyter kernel for this project
remove_kernel:
	jupyter kernelspec uninstall -f $(KERNEL_NAME)

# ## Make Dataset
# data: requirements
# 	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

# ## Delete all compiled Python files
# clean:
# 	find . -type f -name "*.py[co]" -delete
# 	find . -type d -name "__pycache__" -delete

# ## Lint using flake8
# lint:
# 	flake8 src

# ## Upload Data to S3
# sync_data_to_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync data/ s3://$(BUCKET)/data/
# else
# 	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
# endif

# ## Download Data from S3
# sync_data_from_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync s3://$(BUCKET)/data/ data/
# else
# 	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
# endif

# 	python3 -m venv venv && \
# 	. venv/bin/activate && \
# 	pip install -U pip setuptools wheel
# ifeq (True,$(HAS_CONDA))
# 		@echo ">>> Detected conda, creating conda environment."
# ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
# 	conda create --name $(PROJECT_NAME) python=3
# else
# 	conda create --name $(PROJECT_NAME) python=2.7
# endif
# 		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
# else
# 	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
# 	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
# 	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
# 	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
# 	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
# endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
