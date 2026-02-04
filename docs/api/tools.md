# Tools (`ds.tl`)

Functions for training, inference, perturbation simulation, and GRN extraction.

## Training

### Phase 1: Full Model Training

```{eval-rst}
.. autofunction:: deepscenic.tl.train
```

### Phases 2-3: E2 Finetuning

```{eval-rst}
.. autofunction:: deepscenic.tl.finetune_e2
```

## Model

```{eval-rst}
.. autoclass:: deepscenic.tl.DeepSCENICModel
   :no-members:

.. autoclass:: deepscenic.tl.TrainingState
   :no-members:
```

### Loading Models

```{eval-rst}
.. autofunction:: deepscenic.tl.load_model
.. autofunction:: deepscenic.tl.load_legacy_model
.. autofunction:: deepscenic.tl.load_legacy_data
```

## Configuration

```{eval-rst}
.. autoclass:: deepscenic.tl.ModelConfig
   :no-members:

.. autoclass:: deepscenic.tl.TrainingHistory
   :no-members:

.. autoclass:: deepscenic.tl.EarlyStopping
   :no-members:
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
