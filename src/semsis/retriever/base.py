from __future__ import annotations

import abc
import enum
from dataclasses import asdict, dataclass
from typing import Any, Optional, TypeVar

import yaml

from semsis import retriever
from semsis.typing import NDArrayF32, NDArrayFloat, NDArrayI64, StrPath

T = TypeVar("T")


class Metric(str, enum.Enum):
    l2 = "l2"
    ip = "ip"
    cos = "cos"


@dataclass
class RetrieverParam:
    """Configuration of the retriever.

    - hnsw_nlinks (int): [HNSW] Number of links for each node.
        If this value is greater than 0, HNSW will be used.
    - ivf_nlists (int): [IVF] Number of centroids.
    - pq_nblocks (int): [PQ] Number of sub-vectors to be splitted.
    - pq_nbits (int): [PQ] Size of codebooks for each sub-space.
        Usually 8 bit is employed; thus, each codebook has 256 codes.
    - opq (bool): [OPQ] Use OPQ pre-transformation which minimizes the quantization error.
    - pca (bool): [PCA] Use PCA dimension reduction.
    - pca_dim (int): [PCA] Dimension size which is reduced by PCA.
    - fp16 (bool): Use FP16. (GPU only)
    """

    hnsw_nlinks: int = 0
    ivf_nlists: int = 0
    pq_nblocks: int = 0
    pq_nbits: int = 8
    opq: bool = False
    pca: bool = False
    pca_dim: int = 0
    fp16: bool = False

    def __post_init__(self):
        self.hnsw = self.hnsw_nlinks > 0
        self.ivf = self.ivf_nlists > 0
        self.pq = self.pq_nblocks > 0
        if self.opq and self.pca:
            raise ValueError("`opq` and `pca` cannot be set True at the same time.")
        self.transform = self.opq or self.pca


class Retriever(abc.ABC):
    """Base class of retriever classes.

    Args:
        index (Any): Index object.
        cfg (Retriever.Config): Configuration dataclass.
    """

    def __init__(self, index: Any, cfg: Config) -> None:
        self.index = index
        self.cfg = cfg

    @dataclass
    class Config:
        """Configuration of the retriever.

        - dim (int): Size of the dimension.
        - backend (str): Backend of the search engine.
        - metric (Metric): Distance function.
        """

        dim: int = -1
        backend: str = "faiss-cpu"
        metric: Metric = Metric.l2

        def save(self, path: StrPath) -> None:
            """Save the configuration.

            Args:
                path (StrPath): File path.
            """
            with open(path, mode="w") as f:
                yaml.dump(
                    {
                        k: v.value if isinstance(v, Metric) else v
                        for k, v in asdict(self).items()
                    },
                    f,
                    indent=True,
                )

        @classmethod
        def load(cls, path: StrPath):
            """Load the configuration.

            Args:
                path (StrPath): File path.

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
    def build(cls: type[T], cfg: Config) -> T:
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
    def normalize(self, vectors: NDArrayFloat) -> NDArrayF32:
        """Normalize the input vectors for a backend library and the specified metric.

        Args:
            vectors (NDArrayFloat): Input vectors.

        Returns:
            NDArrayF32: Normalized vectors.
        """

    @abc.abstractmethod
    def train(self, vectors: NDArrayFloat) -> None:
        """Train the index for some approximate nearest neighbor search algorithms.

        Args:
            vectors (NDArrayFloat): Training vectors.
        """

    @abc.abstractmethod
    def add(self, vectors: NDArrayFloat, ids: Optional[NDArrayI64] = None) -> None:
        """Add key vectors to the index.

        Args:
            vectors (NDArrayFloat): Key vectors to be added.
            ids (NDArrayI64, optional): Value indices.
        """

    @abc.abstractmethod
    def search(self, querys: NDArrayFloat, k: int = 1) -> tuple[NDArrayF32, NDArrayI64]:
        """Search the k nearest neighbor vectors of the querys.

        Args:
            querys (NDArrayFloat): Query vectors.
            k (int): Top-k.

        Returns:
            distances (NDArrayF32): Distances between the querys and the k nearest
               neighbor vectors.
            indices (NDArrayI64): Indices of the k nearest neighbor vectors.
        """

    @classmethod
    def load(cls: type[T], index_path: StrPath, cfg_path: StrPath) -> T:
        """Load the index and its configuration.

        Args:
            index_path (StrPath): Index file path.
            cfg_path (StrPath): Configuration file path.

        Returns:
            Retriever: This class.
        """
        cfg = cls.Config.load(cfg_path)
        index = cls.load_index(index_path)
        return cls(index, cfg)

    def save(self, index_path: StrPath, cfg_path: StrPath) -> None:
        """Save the index and its configuration.

        Args:
            index_path (StrPath): Index file path to save.
            cfg_path (StrPath): Configuration file path to save.
        """
        self.cfg.save(cfg_path)
        self.save_index(index_path)

    @classmethod
    @abc.abstractmethod
    def load_index(cls, path: StrPath) -> Any:
        """Load the index.

        Args:
            path (StrPath): Index file path.

        Returns:
            Any: Index object.
        """

    @abc.abstractmethod
    def save_index(self, path: StrPath) -> None:
        """Save the index.

        Args:
            path (StrPath): Index file path to save.
        """

    @classmethod
    def get_cls(cls, name: str) -> type[Retriever]:
        """Return the retriever class from the given name.

        Args:
            name (str): Registered name.

        Returns:
            type[Retriever]: Retriever class.
        """
        return retriever.get_cls(name)
