# Tools (`ds.tl`)

Functions for training, inference, perturbation simulation, and GRN extraction.

## Training

```{eval-rst}
.. autofunction:: deepscenic.tl.train
```

## Model

```{eval-rst}
.. autoclass:: deepscenic.tl.DeepSCENICModel
   :no-members:

.. autofunction:: deepscenic.tl.load_model
```

## Inference

```{eval-rst}
.. autofunction:: deepscenic.tl.to_latent
```

## Perturbation Simulation

```{eval-rst}
.. autofunction:: deepscenic.tl.simulate_perturbation
.. autofunction:: deepscenic.tl.simulate_multi_perturbation
```

## GRN Extraction

```{eval-rst}
.. autofunction:: deepscenic.tl.extract_grn
.. autofunction:: deepscenic.tl.extract_e1_matrix
.. autofunction:: deepscenic.tl.extract_e2_matrix
.. autofunction:: deepscenic.tl.get_tf_targets
.. autofunction:: deepscenic.tl.get_gene_regulators
```
