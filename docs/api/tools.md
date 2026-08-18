# Tools (`ds.tl`)

Functions for training, inference, perturbation simulation, sequence interpretation, and GRN extraction.

## Training

### Phase 1: Full Model Training

```{eval-rst}
.. autofunction:: deepscenic.tl.train
```

### Phase 2: R2G Finetuning

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

## Sequence Interpretation

```{eval-rst}
.. autofunction:: deepscenic.tl.in_silico_mutagenesis
.. autoclass:: deepscenic.tl.ISMResult
   :no-members:
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
.. autofunction:: deepscenic.tl.get_region_info
```

### Cell-Type Aware Extraction

For cell-type specific GRN analysis, use these functions in order:

```{eval-rst}
.. autofunction:: deepscenic.tl.identify_active_enhancers
.. autofunction:: deepscenic.tl.compute_celltype_tf2r
.. autofunction:: deepscenic.tl.compute_celltype_r2g
.. autofunction:: deepscenic.tl.compute_celltype_tf2g
.. autofunction:: deepscenic.tl.compute_tf_activity_scores
.. autofunction:: deepscenic.tl.build_grn_for_tfs
```
