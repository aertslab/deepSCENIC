# Input/Output (`ds.io`)

Functions for reading and writing deepSCENIC data.

## Reading Data

```{eval-rst}
.. autofunction:: deepscenic.io.read
.. autofunction:: deepscenic.io.read_legacy
```

## Writing Data

```{eval-rst}
.. autofunction:: deepscenic.io.write
```

## Convenience Functions

These functions are also available at the package level:

```python
import deepscenic as ds

# Equivalent calls:
mdata = ds.read("data.h5mu")
mdata = ds.io.read("data.h5mu")
```
