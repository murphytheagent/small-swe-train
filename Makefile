.PHONY: setup build-train build-dev build-all clean-venv

# Number of cores to use for compilation
CORES ?= 16

# Syncs only the base dependencies
setup:
	uv sync

# Syncs the training environment (compiles flash-attn)
build-train:
	export MAX_JOBS=$(CORES) && uv sync --extra train

# Syncs dev dependencies
build-dev:
	uv sync --extra dev

# Syncs absolutely everything
build-all:
	export MAX_JOBS=$(CORES) && uv sync --all-extras

# Wipes the environment clean so you can start fresh
clean-venv:
	rm -rf .venv
	uv venv