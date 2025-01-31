from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from semsis.retriever import Retriever, register
from semsis.retriever.base import Metric, RetrieverParam
from semsis.typing import NDArrayF32, NDArrayFloat, NDArrayI64, StrPath

FaissMetricType = int


def faiss_index_builder(
    cfg: RetrieverFaissCPU.Config, dim: int, metric: FaissMetricType
) -> faiss.Index:
    """Build a faiss index from the given configuration.

    Args:
        cfg (RetrieverFaissCPU.Config):
        dim (int):
        metric: FaissMetricType

    Returns:
        faiss.Index: Faiss index.
    """
    if cfg.ivf:
        if cfg.hnsw:
            cq = faiss.IndexHNSWFlat(dim, cfg.hnsw_nlinks, metric)
        else:
            cq = faiss.IndexFlat(dim, metric)

        if cfg.pq:
            return faiss.IndexIVFPQ(
                cq, dim, cfg.ivf_nlists, cfg.pq_nblocks, cfg.pq_nbits, metric
            )
        else:
            return faiss.IndexIVFFlat(cq, dim, cfg.ivf_nlists, metric)
    else:
        if cfg.pq:
            if cfg.hnsw:
                return faiss.IndexHNSWPQ(dim, cfg.pq_nblocks, cfg.hnsw_nlinks)
            return faiss.IndexPQ(dim, cfg.pq_nblocks, cfg.pq_nbits, metric)
        else:
            if cfg.hnsw:
                return faiss.IndexHNSWFlat(dim, cfg.hnsw_nlinks, metric)
            return faiss.IndexFlat(dim, metric)


@register("faiss-cpu")
class RetrieverFaissCPU(Retriever):
    """Faiss CPU retriever class.

    Args:
        index (faiss.Index): Index object.
        cfg (RetrieverFaissCPU.Config): Configuration dataclass.
    """

    METRICS_MAP: dict[Metric, FaissMetricType] = {
        Metric.l2: faiss.METRIC_L2,
        Metric.ip: faiss.METRIC_INNER_PRODUCT,
        Metric.cos: faiss.METRIC_INNER_PRODUCT,
    }

    @dataclass
    class Config(RetrieverParam, Retriever.Config):
        """Configuration of the retriever.

        - dim (int): Size of the dimension.
        - backend (str): Backend of the search engine.
        - metric (Metric): Distance function.
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

        def __post_init__(self):
            super().__post_init__()
            if self.pca and self.pca_dim <= 0:
                self.pca_dim = self.dim

    index: faiss.Index
    cfg: Config

    def __len__(self) -> int:
        """Return the size of the index."""
        return self.index.ntotal

    @classmethod
    def build(cls, cfg: Config) -> RetrieverFaissCPU:
        """Build this class from the given configuration.

        Args:
            cfg (RetrieverFaissCPU.Config): Configuration.

        Returns:
            RetrieverFaissCPU: This class with the constucted index object.
        """
        metric = cls.METRICS_MAP[cfg.metric]
        dim = cfg.pca_dim if cfg.pca else cfg.dim

        index = faiss_index_builder(cfg, dim, metric)
        if not cfg.ivf:
            index = faiss.IndexIDMap(index)

        if cfg.opq:
            vtrans = faiss.OPQMatrix(cfg.dim, M=cfg.pq_nblocks)
            index = faiss.IndexPreTransform(vtrans, index)
        elif cfg.pca:
            vtrans = faiss.PCAMatrix(cfg.dim, d_out=cfg.pca_dim)
            index = faiss.IndexPreTransform(vtrans, index)
        return cls(index, cfg)

    def set_nprobe(self, nprobe: int) -> None:
        """Set nprobe parameter for IVF-family indexes.

        Args:
            nprobe (int): Number of nearest neighbor clusters that are
              probed in search time.
        """
        if self.cfg.ivf:
            faiss.extract_index_ivf(self.index).nprobe = nprobe

    def set_efsearch(self, efsearch: int) -> None:
        """Set efSearch parameter for HNSW indexes.

        Args:
            efsearch (int): The depth of exploration of the search.
        """
        if self.cfg.hnsw:
            faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", efsearch)

    def normalize(self, vectors: NDArrayFloat) -> NDArrayF32:
        """Normalize the input vectors for a backend library and the specified metric.

        Args:
            vectors (NDArrayFloat): Input vectors.

        Returns:
            NDArrayF32: Normalized vectors.
        """
        if not np.issubdtype(vectors.dtype, np.float32):
            vectors = np.array(vectors, dtype=np.float32)
        if self.cfg.metric == Metric.cos:
            vectors /= np.fmax(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-9)
        return vectors

    def train(self, vectors: NDArrayFloat) -> None:
        """Train the index for some approximate nearest neighbor search algorithms.

        Args:
            vectors (NDArrayFloat): Training vectors.
        """
        vectors = self.normalize(vectors)
        self.index.train(vectors)

    def add(self, vectors: NDArrayFloat, ids: Optional[NDArrayI64] = None) -> None:
        """Add key vectors to the index.

        Args:
            vectors (NDArrayFloat): Key vectors to be added.
            ids (NDArrayI64, optional): Value indices.
        """
        vectors = self.normalize(vectors)
        if ids is None:
            ids = np.arange(len(self), len(self) + len(vectors))
        return self.index.add_with_ids(vectors, ids)

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
        querys = self.normalize(querys)
        return self.index.search(querys, k=k)

    @classmethod
    def load_index(cls, path: StrPath) -> faiss.Index:
        """Load the index.

        Args:
            path (StrPath): Index file path.

        Returns:
            faiss.Index: Index object.
        """
        return faiss.read_index(str(path))

    def save_index(self, path: StrPath) -> None:
        """Save the index.

        Args:
            path (StrPath): Index file path to save.
        """
        return faiss.write_index(self.index, str(path))
