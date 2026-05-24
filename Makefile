EXECUTOR_PORT    ?= 8001
EXECUTOR_WORKERS ?= $(shell nproc)

.PHONY: executor-up executor-down

executor-up:  ## Start the FastAPI code executor server in the background
	EXECUTOR_WORKERS=$(EXECUTOR_WORKERS) \
	python -m opd.envs.code_executor_server \
	    --host 0.0.0.0 --port $(EXECUTOR_PORT) --workers $(EXECUTOR_WORKERS) &
	@echo "Executor started on http://localhost:$(EXECUTOR_PORT)"
	@echo "Set:  export CODE_EXECUTOR_URL=http://localhost:$(EXECUTOR_PORT)"

executor-down:  ## Stop the FastAPI code executor server
	@pkill -f "opd.envs.code_executor_server" && echo "Executor stopped" || echo "Executor was not running"
