import logging
from dataclasses import dataclass
from typing import Optional, Union

import faiss
import numpy as np
import torch

from semsis.retriever import register
from semsis.retriever.faiss_cpu import RetrieverFaissCPU

logger = logging.getLogger(__name__)

GpuIndex = (
    faiss.GpuIndex if hasattr(faiss, "GpuIndex") else faiss.Index
)  # avoid error on faiss-cpu
ShardedGpuIndex = Union[GpuIndex, faiss.IndexReplicas]


def faiss_index_to_gpu(
    index: faiss.Index, num_gpus: int = -1, precompute: bool = False, fp16: bool = False
) -> ShardedGpuIndex:
    """Transfers the index from CPU to GPU.

    Args:
        index (faiss.Index): Faiss index.
        num_gpus (int): Number of GPUs to use. `-1` uses all devices.
        precompute (bool): Uses the precompute table for IVF-family.
        fp16 (bool): Use fp16.

    Returns:
        faiss.GpuIndex | faiss.IndexReplicas: Faiss index.
    """
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = fp16
    co.useFloat16CoarseQuantizer = fp16
    co.indicesOptions = faiss.INDICES_CPU
    co.shard = True
    if precompute:
        co.usePrecomputed = precompute

    return faiss.index_cpu_to_all_gpus(index, co, ngpu=num_gpus)


def faiss_index_to_cpu(index: ShardedGpuIndex) -> faiss.Index:
    """Transfers the index from GPU to CPU.

    Args:
        index (faiss.GpuIndex | faiss.IndexReplicas): faiss index.

    Returns:
        faiss.Index: faiss index.
    """
    return faiss.index_gpu_to_cpu(index)


@register("faiss-gpu")
class RetrieverFaissGPU(RetrieverFaissCPU):
    """Faiss GPU retriever class.

    This class extend the faiss behavior for efficiency.

    Args:
        index (faiss.Index): Index object.
        cfg (RetrieverFaissGPU.Config): Configuration dataclass.
    """

    cfg: "RetrieverFaissGPU.Config"

    def __init__(self, index: faiss.Index, cfg: "RetrieverFaissGPU.Config") -> None:
        super().__init__(index, cfg)
        self.A: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.gpu_ivf_index: Optional[ShardedGpuIndex] = None

    @dataclass
    class Config(RetrieverFaissCPU.Config):
        """Configuration of the retriever.

        - dim (int): Size of the dimension.
        - backend (str): Backend of the search engine.
        - metric (str): Distance function.
        - hnsw_nlinks (int): [HNSW] Number of links for each node.
            If this value is greater than 0, HNSW will be used.
        - ivf_nlists (int): [IVF] Number of centroids.
        - pq_nblocks (int): [PQ] Number of sub-vectors to be splitted.
        - pq_nbits (int): [PQ] Size of codebooks for each sub-space.
            Usually 8 bit is employed; thus, each codebook has 256 codes.
        - opq (bool): [OPQ] Use OPQ pre-transformation which minimizes the quantization error.
        - pca (bool): [PCA] Use PCA dimension reduction.
        - pca_dim (int): [PCA] Dimension size which is reduced by PCA.
        - fp16 (bool): Use FP16.
        """

        fp16: bool = False

    def to_gpu_train(self) -> None:
        """Transfers the faiss index to GPUs for training.

        This speeds up training of IVF clustering and the PQ codebook:

        * IVF learns the nlists centroids by k-means.

        * PQ can be used for two components: OPQ training and vector quantization.
          - OPQ training: OPQ rotates the input vectors as a pre-transform and the
              rotation matrix is trained to minimize the PQ reconstruction error. If the
              ProductQuantizer object is given in OPQ training, it is finally set
              `ProductQuantizer::Train_hot_start` flag that initializes the codebook by
              the state in the last iteration of OPQ training.
          - Vector Quantization: It learns the PQ codes that is actually added to the
              index. Note that IVFPQ learns codes from residual vectors from each
              centroid, which are different from OPQ.
        """
        if self.cfg.ivf:
            # IVF* learns the nlists centroids from d-dimensional vectors.
            # clustering_index passignments by computing distance between `nlists x d`
            # and `ntrain x d` that would be large, so it is sharded.
            ivf = faiss.extract_index_ivf(self.index)
            clustering_index = faiss_index_to_gpu(
                faiss.IndexFlat(ivf.d, self.METRICS_MAP[self.cfg.metric]),
                fp16=self.cfg.fp16,
            )
            ivf.clustering_index = clustering_index

        if self.cfg.pq:
            index = (
                self.index.index
                if isinstance(self.index, faiss.IndexPreTransform)
                else self.index
            )
            pq: faiss.ProductQuantizer = faiss.downcast_index(index).pq

            # PQ splits input vectors into dsub-dimensional sub-vectors and assigns
            # quantization codes in each sub-space. In addition, PQ is trained from
            # sampled vectors and all training vectors are not used. Therefore, a single
            # GPU is used for PQ assignment because the GPU memory footprint is small
            # (`ksub x dsub`, where ksub is the codebook size and typically =256).
            pq.assign_index = faiss_index_to_gpu(
                faiss.IndexFlatL2(pq.dsub), num_gpus=1, fp16=self.cfg.fp16
            )
            if self.cfg.opq:
                opq: faiss.OPQMatrix = faiss.downcast_VectorTransform(
                    self.index.chain.at(0)
                )
                if self.cfg.ivf:
                    # IVFPQ learns PQ from the different vectors, so this PQ is not
                    # shared with IVFPQ.
                    opq_pq = faiss.ProductQuantizer(opq.d_out, opq.M, pq.nbits)
                    opq_pq.assign_index = faiss_index_to_gpu(
                        faiss.IndexFlatL2(opq_pq.dsub), num_gpus=1, fp16=self.cfg.fp16
                    )
                else:
                    # Otherwise, PQ codes are initialized by the last state of OPQ
                    # training and PQ training starts from that initialized codebook.
                    opq_pq = pq
                opq.pq = opq_pq

    def to_gpu_add(self) -> None:
        """Transfers the faiss index to GPUs for adding."""

        if self.cfg.transform:
            vt: faiss.LinearTransform = faiss.downcast_VectorTransform(
                faiss.downcast_index(self.index).chain.at(0)
            )
            self.A = torch.from_numpy(
                faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in).T
            )
            self.b = torch.from_numpy(faiss.vector_to_array(vt.b))

            self.A = self.A.cuda()
            self.b = self.b.cuda()
            if self.cfg.fp16:
                self.A = self.A.half()
                self.b = self.b.half()

        if self.cfg.ivf:
            ivf = faiss.extract_index_ivf(self.index)
            if self.cfg.hnsw:
                # Temporarily replace the HNSW coarse quantizer with the flat index.
                hnsw_cq = faiss.downcast_index(ivf.quantizer)
                ivf.quantizer = faiss.downcast_index(hnsw_cq.storage)
                self.gpu_ivf_index = faiss_index_to_gpu(ivf, fp16=self.cfg.fp16)

                # After transfering the IVF index with the flat coarse quantizer,
                # the coarse quantizer of the master IVF index is restored to HNSW.
                ivf.quantizer = hnsw_cq
            else:
                self.gpu_ivf_index = faiss_index_to_gpu(ivf, fp16=self.cfg.fp16)
            self.gpu_ivf_index.reset()

    def to_gpu_search(self) -> None:
        """Transfers the faiss index to GPUs for searching."""
        if self.cfg.ivf and self.cfg.hnsw:
            ivf_index: faiss.IndexIVF = faiss.extract_index_ivf(self.index)
            ivf_index.quantizer = faiss.downcast_index(
                faiss.downcast_index(ivf_index.quantizer).storage
            )
        self.index = faiss_index_to_gpu(self.index, fp16=self.cfg.fp16)
        logger.info(f"The retriever index is on the GPU.")

    def to_cpu(self) -> None:
        """Transfers the faiss index to CPUs."""
        self.index = faiss_index_to_cpu(self.index)

    def rotate(self, x: torch.Tensor, shard_size: int = 2**20) -> torch.Tensor:
        """Rotate the input vectors instead of faiss.IndexPreTransform.

        Args:
            x (torch.Tensor): Input vectors of shpae `(n, D)`
            shard_size (int): Number of rotated vectors at once.
              The default size is 2**20 (Each shard would take 2 GiB when D=256).

        Returns:
            torch.Tensor: Pre-transformed vectors of shape `(n, D)`.
        """
        if self.A is None:
            return x

        x = x.type(self.A.dtype)
        x_device = x.device
        A_device = self.A.device

        # Compute rotation of `x[i:j]` while `i < n`.
        results = []
        n = x.size(0)
        if shard_size <= 0:
            shard_size = n
        i = 0
        while i < n:
            j = min(i + shard_size, n)
            xs = x[i:j]
            xs = xs.to(A_device)
            xs @= self.A
            if self.b is not None and self.b.numel() > 0:
                xs += self.b
            results.append(xs.to(x_device))
            i = j
        return torch.cat(results, dim=0)

    def add_gpu_ivf_index(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Adds vectors to the index with the full GPU IVF index.

        This method runs as follows:
        1. Transfers a trained IVF index to GPU devices. The index has only centroid
          information and does not have any vectors.
        2. Adds the input vectors to the temporary GPU index.
        3. Copies a replica of the GPU index to CPU.
        4. Copies the added vectors with their IVF lists from 3. to the real index.
        5. Empties the storage of the GPU index. Here, the GPU index has only centroids.

        Args:
            vectors (ndarray): Key vectors to be added.
            ids (np.ndarray): Value indices.
        """
        ivf: faiss.IndexIVF = faiss.extract_index_ivf(self.index)
        self.gpu_ivf_index.add_with_ids(vectors, ids)
        cpu_ivf_index: faiss.IndexIVF = faiss_index_to_cpu(self.gpu_ivf_index)
        assert cpu_ivf_index.ntotal == vectors.shape[0]
        ivf.merge_from(cpu_ivf_index, 0)
        self.gpu_ivf_index.reset()

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """Add key vectors to the index.

        Args:
            vectors (np.ndarray): Key vectors to be added.
            ids (np.ndarray, optional): Value indices.
        """
        vectors = self.normalize(vectors)
        if self.cfg.transform:
            vectors = self.rotate(torch.from_numpy(vectors)).numpy().astype(np.float32)

        if ids is None:
            ids = np.arange(len(self), len(self) + len(vectors))

        index = self.index
        if self.cfg.transform:
            index = faiss.downcast_index(index.index)

        if self.cfg.ivf and self.gpu_ivf_index is not None:
            self.add_gpu_ivf_index(vectors, ids)
        else:
            index.add_with_ids(vectors, ids)

        if self.cfg.transform:
            self.index.ntotal = index.ntotal
