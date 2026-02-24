.PHONY: setup build-train build-dev build-all clean-venv ensure-flash-attn rebuild-flash-attn verify-flash-attn submit-flash-attn-rebuild

# Number of cores to use for compilation
# Default for dedicated Blackwell build nodes; override when needed.
CORES ?= 8
# Keep train-env builds on an ABI with available torch/ray wheels on GPU nodes.
PYTHON_VERSION ?= 3.13
UV ?= uv
VENV_PYTHON ?= .venv/bin/python
FLASH_ATTN_PACKAGE ?= flash-attn
# Blackwell default (SM120). Override when targeting different GPU architectures.
FLASH_ATTN_CUDA_ARCHS ?= 120

# Syncs only the base dependencies
setup:
	$(UV) sync --python $(PYTHON_VERSION)

# Syncs the training environment (compiles flash-attn)
build-train:
	MAX_JOBS=$(CORES) $(UV) sync --python $(PYTHON_VERSION) --extra train
	$(MAKE) ensure-flash-attn

# Syncs dev dependencies
build-dev:
	$(UV) sync --python $(PYTHON_VERSION) --extra dev

# Syncs absolutely everything
build-all:
	MAX_JOBS=$(CORES) $(UV) sync --python $(PYTHON_VERSION) --all-extras
	$(MAKE) ensure-flash-attn

# Ensures flash-attn is import-compatible with the current torch/cuda stack.
ensure-flash-attn:
	@$(VENV_PYTHON) -c "import flash_attn; from flash_attn import flash_attn_interface" >/dev/null 2>&1 || $(MAKE) rebuild-flash-attn
	$(MAKE) verify-flash-attn

# Forces a source build against the torch version already installed in .venv.
rebuild-flash-attn:
	@echo "Rebuilding flash-attn from source against current torch..."
	$(UV) pip install --python $(VENV_PYTHON) "setuptools>=80.0" "wheel>=0.46.0" "packaging>=24.0" "ninja>=1.13.0"
	$(UV) pip uninstall --python $(VENV_PYTHON) $(FLASH_ATTN_PACKAGE) || true
	MAX_JOBS=$(CORES) TORCH_CUDA_ARCH_LIST=$(FLASH_ATTN_CUDA_ARCHS) FLASH_ATTN_CUDA_ARCHS=$(FLASH_ATTN_CUDA_ARCHS) FLASH_ATTENTION_FORCE_BUILD=1 \
	$(UV) pip install --python $(VENV_PYTHON) --no-build-isolation --no-cache --no-binary $(FLASH_ATTN_PACKAGE) --reinstall-package $(FLASH_ATTN_PACKAGE) $(FLASH_ATTN_PACKAGE)

verify-flash-attn:
	@$(VENV_PYTHON) -c "import flash_attn; from flash_attn import flash_attn_interface; print('flash-attn import OK')"

# Submits a constrained Slurm rebuild job (GPU partition by default).
submit-flash-attn-rebuild:
	bash scripts/run_flash_attn_rebuild.sh

# Wipes the environment clean so you can start fresh
clean-venv:
	rm -rf .venv
	$(UV) venv --python $(PYTHON_VERSION)
