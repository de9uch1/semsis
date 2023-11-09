import abc
from dataclasses import asdict, dataclass
from os import PathLike
from typing import Any, Optional, Tuple, Type, TypeVar

import numpy as np
import yaml

from semsis import retriever

T = TypeVar("T")


class Retriever(abc.ABC):
    """Base class of retriever classes.

    Args:
        index (Any): Index object.
        cfg (Retriever.Config): Configuration dataclass.
    """

    def __init__(self, index: Any, cfg: "Config") -> None:
        self.index = index
        self.cfg = cfg

    @dataclass
    class Config:
        """Configuration of the retriever.

        - dim (int): Size of the dimension.
        - backend (str): Backend of the search engine.
        - metric (str): Distance function.
        """

        dim: int
        backend: str = "faiss-cpu"
        metric: str = "l2"

        def save(self, path: PathLike) -> None:
            """Save the configuration.

            Args:
                path (os.PathLike): File path.
            """
            with open(path, mode="w") as f:
                yaml.dump(asdict(self), f, indent=True)

        @classmethod
        def load(cls, path: PathLike):
            """Load the configuration.

            Args:
                path (os.PathLike): File path.

            Returns:
                Retriver.Config: This configuration object.
            """
            with open(path, mode="r") as f:
                config = yaml.safe_load(f)
            return cls(**config)

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the size of the index."""

    @classmethod
    @abc.abstractmethod
    def build(cls: Type[T], cfg: "Config") -> T:
        """Build this class from the given configuration.


        Args:
            cfg (Retriever.Config): Configuration.

        Returns:
            Retriever: This class with the constucted index object.
        """

    def to_gpu_train(self) -> None:
        """Transfers the index to GPUs for training."""

    def to_gpu_add(self) -> None:
        """Transfers the index to GPUs for adding vectors."""

    def to_gpu_search(self) -> None:
        """Transfers the index to GPUs for searching."""

    def to_cpu(self) -> None:
        """Transfers the index to CPUs."""

    def set_nprobe(self, nprobe: int) -> None:
        """Set nprobe parameter for IVF-family indexes.

        Args:
            nprobe (int): Number of nearest neighbor clusters that are
              probed in search time.
        """

    def set_efsearch(self, efsearch: int) -> None:
        """Set efSearch parameter for HNSW indexes.

        Args:
            efsearch (int): The depth of exploration of the search.
        """

    @abc.abstractmethod
    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize the input vectors for a backend library and the specified metric.

        Args:
            vectors (np.ndarray): Input vectors.

        Returns:
            np.ndarray: Normalized vectors.
        """

    @abc.abstractmethod
    def train(self, vectors: np.ndarray) -> None:
        """Train the index for some approximate nearest neighbor search algorithms.

        Args:
            vectors (np.ndarray): Training vectors.
        """

    @abc.abstractmethod
    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """Add key vectors to the index.

        Args:
            vectors (np.ndarray): Key vectors to be added.
            ids (np.ndarray, optional): Value indices.
        """

    @abc.abstractmethod
    def search(self, querys: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Search the k nearest neighbor vectors of the querys.

        Args:
            querys (np.ndarray): Query vectors.
            k (int): Top-k.

        Returns:
            distances (np.ndarray): Distances between the querys and the k nearest
               neighbor vectors.
            indices (np.ndarray): Indices of the k nearest neighbor vectors.
        """

    @classmethod
    def load(cls: Type[T], index_path: PathLike, cfg_path: PathLike) -> T:
        """Load the index and its configuration.

        Args:
            index_path (os.PathLike): Index file path.
            cfg_path (os.PathLike): Configuration file path.

        Returns:
            Retriever: This class.
        """
        cfg = cls.Config.load(cfg_path)
        index = cls.load_index(index_path)
        return cls(index, cfg)

    def save(self, index_path: PathLike, cfg_path: PathLike) -> None:
        """Save the index and its configuration.

        Args:
            index_path (os.PathLike): Index file path to save.
            cfg_path (os.PathLike): Configuration file path to save.
        """
        self.cfg.save(cfg_path)
        self.save_index(index_path)

    @classmethod
    @abc.abstractmethod
    def load_index(cls, path: PathLike) -> Any:
        """Load the index.

        Args:
            path (os.PathLike): Index file path.

        Returns:
            Any: Index object.
        """

    @abc.abstractmethod
    def save_index(self, path: PathLike) -> None:
        """Save the index.

        Args:
            path (os.PathLike): Index file path to save.
        """

    @classmethod
    def get_cls(cls, name: str) -> Type["Retriever"]:
        """Return the retriever class from the given name.

        Args:
            name (str): Registered name.

        Returns:
            Type[Retriever]: Retriever class.
        """
        return retriever.get_cls(name)
