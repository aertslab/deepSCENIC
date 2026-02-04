# Data Validation

Schema validation utilities for deepSCENIC MuData objects.

## Validation Functions

```{eval-rst}
.. autofunction:: deepscenic.validate_schema
.. autofunction:: deepscenic.is_valid_schema
```

## Exceptions

```{eval-rst}
.. autoexception:: deepscenic.SchemaError
.. autoclass:: deepscenic.SchemaWarning
```

## Usage

```python
import deepscenic as ds

# Validate a MuData object (logs warnings)
ds.validate_schema(mdata, strict=False)

# Strict validation (raises SchemaError on issues)
ds.validate_schema(mdata, strict=True)

# Check validity without side effects
is_valid = ds.is_valid_schema(mdata)
```
