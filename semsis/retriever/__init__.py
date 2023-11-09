from os import PathLike
from typing import Type

import yaml

from semsis import registry

register, get_cls = registry.setup("retriever")

from .base import Retriever
from .faiss_cpu import RetrieverFaissCPU
from .faiss_gpu import RetrieverFaissGPU


def load_backend_from_config(cfg_path: PathLike) -> Type[Retriever]:
    """Load the backend retriever type from the configuration file.

    Args:
        cfg_path (os.PathLike): Path to the configuration file.

    Returns:
        Type[Retriever]: The backend retriever type.
    """
    with open(cfg_path, mode="r") as f:
        cfg = yaml.safe_load(f)
    return get_cls(cfg["backend"])


__all__ = [
    "Retriever",
    "RetrieverFaissCPU",
    "RetrieverFaissGPU",
    "register",
    "get_cls",
    "load_backend_from_config",
]
