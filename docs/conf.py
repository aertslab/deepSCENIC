# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

# Add source to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------
info = metadata("deepscenic")
project = info["Name"]
# Handle cases where author-email might not be set
_author_email = info.get("Author-email")
if _author_email:
    author = _author_email.split("<")[0].strip()
else:
    # Fall back to author name from metadata
    _author = info.get("Author")
    author = _author if _author else "deepSCENIC Team"
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]

# Get repository URL from metadata
_repo_url = "https://github.com/aertslab/deepSCENIC"

# -- General configuration ---------------------------------------------------
extensions = [
    # Sphinx core
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    # Type hints
    "sphinx_autodoc_typehints",
    # Markdown and notebooks (scverse convention: myst_nb, not nbsphinx)
    "myst_nb",
    # Copybutton for code blocks
    "sphinx_copybutton",
    # Tabs for code examples
    "sphinx_tabs.tabs",
    # Bibliography
    "sphinxcontrib.bibtex",
    # Open Graph meta tags
    "sphinxext.opengraph",
]

# Autosummary settings
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Napoleon settings (numpy docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True

# MyST settings (scverse convention)
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 6

# myst_nb settings (replaces nbsphinx in scverse convention)
nb_execution_mode = "off"  # Don't execute notebooks during build
nb_output_stderr = "remove"

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "mudata": ("https://mudata.readthedocs.io/en/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Bibliography
bibtex_bibfiles = ["references.bib"]

# -- Options for HTML output (scverse convention: sphinx_book_theme) ---------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "deepSCENIC.png"

html_theme_options = {
    "repository_url": _repo_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
}

# Source suffix - myst_nb handles both .md and .ipynb
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Templates
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Suppress warnings for missing references in external packages
nitpicky = True
nitpick_ignore = [
    ("py:class", "optional"),
    # scipy sparse types (intersphinx doesn't resolve short names)
    ("py:class", "scipy.sparse._csr.csr_matrix"),
    ("py:class", "scipy.sparse.csr_matrix"),
    ("py:class", "csr_matrix"),
    ("py:class", "sparse matrix"),
    # mudata (not in standard intersphinx)
    ("py:class", "mudata.MuData"),
    ("py:class", "MuData"),
    # Short type aliases (docstrings use short names, not fully qualified)
    # These could be fixed by using full paths in docstrings, but for now we ignore them
    ("py:class", "Figure"),  # matplotlib.figure.Figure
    ("py:class", "Axes"),  # matplotlib.axes.Axes
    ("py:class", "AnnData"),  # anndata.AnnData
    ("py:class", "DataFrame"),  # pandas.DataFrame
    ("py:class", "pd.DataFrame"),
    ("py:class", "np.ndarray"),  # numpy.ndarray
    ("py:class", "ndarray"),
    ("py:class", "Tensor"),  # torch.Tensor
    ("py:class", "nn.Module"),  # torch.nn.Module
    ("py:class", "path"),  # pathlib.Path or os.path
    ("py:class", "Path"),
    # Internal private classes
    ("py:class", "deepscenic.models._vae.DeepSCENICVAE"),
    ("py:class", "deepscenic.models._tf2rnet.MotifNet"),
    ("py:class", "deepscenic.tl._training_state.TrainingConfig"),
    ("py:class", "DeepSCENICVAE"),
    ("py:class", "MotifNet"),
    ("py:class", "TrainingConfig"),
]

# Ignore regex patterns for malformed type hints from docstrings
nitpick_ignore_regex = [
    # Default values parsed as types
    (r"py:class", r"default=.*"),
    # Literal string values
    (r"py:class", r"'.*'"),
    # Set/dict literals
    (r"py:class", r"\{.*"),
    (r"py:class", r".*\}"),
]
