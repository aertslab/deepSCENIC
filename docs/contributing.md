# Contributing

## Development Setup

```bash
git clone https://github.com/aertslab/deepSCENIC.git
cd deepSCENIC

# Using hatch (recommended)
pip install hatch
hatch env create  # Creates default environment with dev dependencies

# Or using pip directly
pip install -e ".[dev,test,doc]"
pre-commit install
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check src/
ruff format src/
```

To run these automatically on every commit, install the pre-commit hooks once:

```bash
pre-commit install
```

## Running Tests

```bash
# Using hatch
hatch test           # current Python version
hatch test --all     # all supported Python versions (as in CI)

# Or using pytest directly
pytest tests/
```

## Documentation

Build and preview docs locally:

```bash
# Using hatch (recommended)
hatch run docs:build   # Build docs
hatch run docs:open    # Open in browser
hatch run docs:clean   # Clean build artifacts

# Or using sphinx directly
sphinx-build -M html docs docs/_build -W
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Docstring Style

Use numpy-style docstrings:

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """
    Short description.

    Longer description if needed.

    Parameters
    ----------
    param1
        Description of param1
    param2
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> my_function(1, "test")
    True
    """
```

## Adding New Features

When adding new functionality:

1. **Add to appropriate module**: `pp` for preprocessing, `tl` for tools, `pl` for plotting
2. **Write docstrings**: Use numpy-style with Parameters, Returns, and Examples
3. **Export in `__init__.py`**: Add to the module's `__all__` list
4. **Add tests**: Create tests in `tests/` directory
5. **Update docs**: Add to relevant API page in `docs/api/`

## Reporting Issues

When reporting bugs, please include:

- deepSCENIC version (`ds.__version__`)
- Python version
- Operating system
- Minimal reproducible example
- Full error traceback

## Questions?

Open an issue on [GitHub](https://github.com/aertslab/deepSCENIC/issues).