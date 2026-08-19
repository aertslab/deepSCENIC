# Changelog

All notable changes to deepSCENIC will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New scanpy-style API (`ds.pp`, `ds.tl`, `ds.pl` modules)
- Comprehensive documentation with 6 tutorial notebooks
- MuData-based data structures for multimodal data
- `ds.datasets` module for fetching TF lists and gene annotations
- Schema validation with `ds.data.validate_schema()`
- Sphinx documentation with sphinx-book-theme
- ReadTheDocs integration with hatch/uv

### Changed
- Refactored from legacy single-file structure to modular package
- Updated to PyTorch 2.0+ compatibility
- Improved docstrings with numpy-style format
- Package structure follows scverse conventions

### Deprecated
- Legacy `deepSCENIC` class (use `ds.tl.train()` instead)
- Old file-based data loading (use `ds.read()` with MuData)

## [0.1.0] - 2024-XX-XX

Initial release with core functionality:

### Added
- TF2rNet (Enformer + MotifNet) for sequence-based TF binding prediction
- VAE architecture for multimodal learning (RNA + ATAC)
- Perturbation simulation for TF knockdown/overexpression
- GRN extraction and visualization tools
- Basic preprocessing functions
