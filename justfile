# Lint using ruff (use `just format` to do formatting)
lint:
	ruff format --check
	ruff check

# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Install Python dependencies
requirements:
	uv sync

# Format source code with ruff
format:
	ruff check --fix
	ruff format

# Check repo with mypy
type:
	uv run mypy $(PROJECT_NAME)

# Run tests
test:
	uv run pytest tests

# Run marimo server
marimo:
	uv run marimo edit ./notebooks/


# Set up Python interpreter environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	
