"""Neural network architectures for deepSCENIC."""

from ._decoder import GenerativeNet, GenerativeNetATAC
from ._encoder import InferenceNet
from ._layers import GaussianSampler, PositiveLinear
from ._ppi import NoPPI, PPIgnn
from ._tf2rnet import MotifNet
from ._vae import DeepSCENICVAE, VAEOutput

__all__ = [
    # Layers
    "PositiveLinear",
    "GaussianSampler",
    # Encoder/Decoder
    "InferenceNet",
    "GenerativeNet",
    "GenerativeNetATAC",
    # TF2rNet
    "MotifNet",
    # PPI
    "PPIgnn",
    "NoPPI",
    # Main model
    "DeepSCENICVAE",
    "VAEOutput",
]
