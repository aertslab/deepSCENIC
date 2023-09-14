# deepSCENIC
<img src="https://github.com/aertslab/deepSCENIC/blob/main/docs/deepSCENIC.png" width=100% height=100%>

<!---
![architecture](https://github.com/aertslab/deepSCENIC/blob/main/docs/deepSCENIC.png)
-->

### deepSCENIC workflow
1. [format_data.ipynb](notebooks/format_data.ipynb): notebook to format data for training deepSCENIC model
For training deepSCENIC with Enformer embedding, we nee first to generate the embeddings on scATAC-seq peaks with:
2. [Enformer_embedding.ipynb](notebooks/Enformer_embeddings.ipynb): notebook to extract Enformer embeddings of genomic regions
Or we can pretrain a CNN (based on [Sei](https://github.com/FunctionLab/sei-framework)) to predict scATA-seq
2. [tf2rNet_pretrain.ipynb](notebooks/tf2rNet_pretraining.ipynb): notebook to pretrain tf2rNet model
3. [vsc_script.slurm](vsc_script.slurm): train deepSCENIC model
4. [extract_GRN.ipynb](notebooks/extract_GRN.ipynb): Extract eGRN (enhancer gene regulatory network) from deepSCENIC
5. [insilico_perturbations.ipynb](notebooks/insilico_perturbations.ipynb): Perform in silico perturbation of TFs
6. [explain_sequences.ipynb](notebooks/explain_sequences.ipynb): run sequence explanation (DeepExplainer, GradientsxInputs, ISM,...)