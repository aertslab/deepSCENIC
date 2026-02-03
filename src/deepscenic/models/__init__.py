"""Neural network architectures for deepSCENIC."""

from ._decoder import GenerativeNet
from ._encoder import InferenceNet
from ._layers import GaussianSampler, PositiveLinear
from ._motifnet import MotifNet
from ._vae import DeepSCENICVAE, VAEOutput

__all__ = [
    # Layers
    "PositiveLinear",
    "GaussianSampler",
    # Encoder/Decoder
    "InferenceNet",
    "GenerativeNet",
    # Sequence head (DNA → TF binding)
    "MotifNet",
    # Main model
    "DeepSCENICVAE",
    "VAEOutput",
]
