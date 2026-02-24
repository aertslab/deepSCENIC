# Tools (`ds.tl`)

Functions for training, inference, perturbation simulation, and GRN extraction.

## Training

### Phase 1: Full Model Training

```{eval-rst}
.. autofunction:: deepscenic.tl.train
```

### Phases 2-3: R2G Finetuning

```{eval-rst}
.. autofunction:: deepscenic.tl.finetune_r2g
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

### Basic Extraction

```{eval-rst}
.. autofunction:: deepscenic.tl.extract_grn
.. autofunction:: deepscenic.tl.extract_tf2r_matrix
.. autofunction:: deepscenic.tl.extract_r2g_matrix
```

### Query Functions

```{eval-rst}
.. autofunction:: deepscenic.tl.get_tf_targets
.. autofunction:: deepscenic.tl.get_gene_regulators
```

### Cell-Type Aware Extraction

For cell-type specific GRN analysis, use these functions in order:

```{eval-rst}
.. autofunction:: deepscenic.tl.identify_active_enhancers
.. autofunction:: deepscenic.tl.compute_tf_activity_scores
.. autofunction:: deepscenic.tl.identify_key_tfs
.. autofunction:: deepscenic.tl.build_grn_for_tfs
```
