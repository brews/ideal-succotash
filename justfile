# format python files
format:
	uv run ruff format

# lint python files, fixing what can be fixed
lint:
	uv run ruff check --fix

# run tests
test:
	uv run pytest -v --color=yes

# run tests with coverage
test-cov:
	uv run pytest -vv --color=yes --cov ideal_succotash

# run format, linting, testing checks
validate: format lint test
