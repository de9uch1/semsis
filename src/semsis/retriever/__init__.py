import yaml

from semsis import registry
from semsis.typing import StrPath

register, get_cls = registry.setup("retriever")

from .base import Retriever, Metric
from .faiss_cpu import RetrieverFaissCPU
from .faiss_gpu import RetrieverFaissGPU


def load_backend_from_config(cfg_path: StrPath) -> type[Retriever]:
    """Load the backend retriever type from the configuration file.

    Args:
        cfg_path (StrPath): Path to the configuration file.

    Returns:
        type[Retriever]: The backend retriever type.
    """
    with open(cfg_path, mode="r") as f:
        cfg = yaml.safe_load(f)
    return get_cls(cfg["backend"])


__all__ = [
    "Metric",
    "Retriever",
    "RetrieverFaissCPU",
    "RetrieverFaissGPU",
    "register",
    "get_cls",
    "load_backend_from_config",
]
