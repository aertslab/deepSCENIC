# Input/Output

Functions for reading and writing deepSCENIC data.

## Reading Data

```{eval-rst}
.. autofunction:: deepscenic.read
.. autofunction:: deepscenic.read_bed
```

## Writing Data

```{eval-rst}
.. autofunction:: deepscenic.write
```

## Usage

```python
import deepscenic as ds

# Read a MuData file
mdata = ds.read("data.h5mu")

# Read regions from BED file
regions = ds.read_bed("peaks.bed")

# Write processed data
ds.write(mdata, "processed.h5mu")
```
