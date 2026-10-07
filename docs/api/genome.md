# Genome

Reference genome handling for sequence extraction. Register a genome once and functions that need DNA sequence, such as {func}`deepscenic.tl.in_silico_mutagenesis`, use it by default.

## Genome Registration

```{eval-rst}
.. autofunction:: deepscenic.register_genome
.. autofunction:: deepscenic.get_genome
.. autofunction:: deepscenic.clear_genome
```

## Classes

```{eval-rst}
.. autoclass:: deepscenic.Genome
   :no-members:

.. autoclass:: deepscenic.GenomeIntervalDataset
   :no-members:
```

## Usage

```python
import deepscenic as ds

# Register the reference genome used for sequence extraction
ds.register_genome("genome.fa")

# Retrieve the registered genome
genome = ds.get_genome()
```
