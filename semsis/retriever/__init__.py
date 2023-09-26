from typing import Type

from .base import REGISTRY, Retriever, register
from .faiss import RetrieverFaiss
from .faiss_gpu import RetrieverFaissGPU


def load_retriever(name: str) -> Type[Retriever]:
    return REGISTRY[name]


__all__ = [
    "Retriever",
    "RetrieverFaiss",
    "RetrieverFaissGPU",
    "register",
    "load_retriever",
]
