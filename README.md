# deepSCENIC
<img src="https://github.com/aertslab/deepSCENIC/blob/main/docs/deepSCENIC.png" width=100% height=100%>

<!---
![architecture](https://github.com/aertslab/deepSCENIC/blob/main/docs/deepSCENIC.png)
-->

### deepSCENIC workflow
1. [format_data.ipynb](notebooks/format_data.ipynb): notebook to format data for training deepSCENIC model
2. For training deepSCENIC with Enformer embeddings, we need first to generate the embeddings on scATAC-seq peaks with: \
3. [vsc_script.slurm](vsc_script.slurm): train deepSCENIC model. You can monitor the training opening `tensorboard` on the log folder specified in the slurm script.\
`tensorboard --logdir <LOG_DIR>/logs --port=6006`
4. [extract_GRN.ipynb](notebooks/extract_GRN.ipynb): Extract eGRN (enhancer gene regulatory network) from deepSCENIC
6. [explain_sequences.ipynb](notebooks/explain_sequences.ipynb): run sequence explanation (DeepExplainer, GradientsxInputs, ISM,...)
