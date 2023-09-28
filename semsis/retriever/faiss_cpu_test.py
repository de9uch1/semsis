from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pytest

from semsis.retriever.faiss_cpu import RetrieverFaissCPU, faiss_index_builder

N = 3
D = 8
M = 4
nbits = 4
nlists = 4


@pytest.fixture(scope="module")
def v() -> np.ndarray:
    np.random.seed(0)
    return np.random.rand(N, D).astype(np.float32)


@pytest.mark.parametrize("hnsw_nlinks", [0, 4])
@pytest.mark.parametrize("pq_nblocks", [0, M])
@pytest.mark.parametrize("ivf_nlists", [0, nlists])
def test_faiss_index_builder(hnsw_nlinks: int, pq_nblocks: int, ivf_nlists: int):
    cfg = RetrieverFaissCPU.Config(
        D,
        hnsw_nlinks=hnsw_nlinks,
        ivf_nlists=ivf_nlists,
        pq_nblocks=pq_nblocks,
        pq_nbits=nbits,
    )
    index = faiss_index_builder(cfg, D, RetrieverFaissCPU.METRICS_MAP[cfg.metric])

    if ivf_nlists > 0:
        if pq_nblocks > 0:
            assert isinstance(index, faiss.IndexIVFPQ)
        else:
            assert isinstance(index, faiss.IndexIVFFlat)

        cq = faiss.downcast_index(index.quantizer)
        if hnsw_nlinks > 0:
            assert isinstance(cq, faiss.IndexHNSWFlat)
        else:
            assert isinstance(cq, faiss.IndexFlat)
    else:
        if pq_nblocks > 0:
            if hnsw_nlinks > 0:
                assert isinstance(index, faiss.IndexHNSWPQ)
            else:
                assert isinstance(index, faiss.IndexPQ)
        else:
            if hnsw_nlinks > 0:
                assert isinstance(index, faiss.IndexHNSWFlat)
            else:
                assert isinstance(index, faiss.IndexFlat)


class TestRetrieverFaissCPU:
    @pytest.fixture
    def retriever(self):
        cfg = RetrieverFaissCPU.Config(D)
        index = faiss.IndexIDMap(faiss.IndexFlatL2(D))
        return RetrieverFaissCPU(index, cfg)

    def test__len__(self, retriever: RetrieverFaissCPU, v: np.ndarray):
        idx = np.arange(N)
        assert len(retriever) == 0
        retriever.index.add_with_ids(v, idx)
        assert len(retriever) == N
        retriever.index.add_with_ids(v, idx)
        assert len(retriever) == N * 2

    @pytest.mark.parametrize("metric", RetrieverFaissCPU.METRICS_MAP.keys())
    @pytest.mark.parametrize("opq", [False, True])
    @pytest.mark.parametrize("pca", [False, True])
    @pytest.mark.parametrize("pca_dim", [0, 4])
    def test_build(self, metric: str, opq: bool, pca: bool, pca_dim: int):
        if opq and pca:
            with pytest.raises(ValueError):
                RetrieverFaissCPU.Config(
                    D, metric=metric, opq=opq, pca=pca, pca_dim=pca_dim
                )
        else:
            cfg = RetrieverFaissCPU.Config(
                D, metric=metric, opq=opq, pca=pca, pca_dim=pca_dim
            )
            index = RetrieverFaissCPU.build(cfg).index

            if opq:
                assert isinstance(index, faiss.IndexPreTransform)
                idmap_index = faiss.downcast_index(index.index)
                assert isinstance(idmap_index, faiss.IndexIDMap)
                assert isinstance(
                    faiss.downcast_index(idmap_index.index), faiss.IndexFlat
                )
                assert isinstance(
                    faiss.downcast_VectorTransform(index.chain.at(0)), faiss.OPQMatrix
                )
            elif pca:
                assert cfg.pca_dim == D if pca_dim <= 0 else pca_dim
                assert isinstance(index, faiss.IndexPreTransform)
                idmap_index = faiss.downcast_index(index.index)
                assert isinstance(idmap_index, faiss.IndexIDMap)
                assert isinstance(
                    faiss.downcast_index(idmap_index.index), faiss.IndexFlat
                )
                assert isinstance(
                    faiss.downcast_VectorTransform(index.chain.at(0)), faiss.PCAMatrix
                )
            else:
                assert isinstance(index, faiss.IndexIDMap)
                assert isinstance(faiss.downcast_index(index.index), faiss.IndexFlat)

    @pytest.mark.parametrize("hnsw_nlinks", [0, 4])
    @pytest.mark.parametrize("ivf_nlists", [0, nlists])
    @pytest.mark.parametrize("opq", [False, True])
    def test_set_nprobe(self, hnsw_nlinks: int, ivf_nlists: int, opq: bool):
        retriever = RetrieverFaissCPU.build(
            RetrieverFaissCPU.Config(
                D, hnsw_nlinks=hnsw_nlinks, ivf_nlists=ivf_nlists, opq=opq
            )
        )
        index = retriever.index
        if ivf_nlists > 0:
            assert faiss.extract_index_ivf(index).nprobe == 1
            retriever.set_nprobe(8)
            assert faiss.extract_index_ivf(index).nprobe == 8
        else:
            # Do nothing
            retriever.set_nprobe(8)

    @pytest.mark.parametrize("hnsw_nlinks", [0, 4])
    @pytest.mark.parametrize("ivf_nlists", [0, nlists])
    @pytest.mark.parametrize("opq", [False, True])
    def test_set_efsearch(self, hnsw_nlinks: int, ivf_nlists: int, opq: bool):
        retriever = RetrieverFaissCPU.build(
            RetrieverFaissCPU.Config(
                D, hnsw_nlinks=hnsw_nlinks, ivf_nlists=ivf_nlists, opq=opq
            )
        )
        index = retriever.index
        if hnsw_nlinks > 0:
            if ivf_nlists > 0:
                hnsw = faiss.downcast_index(
                    faiss.extract_index_ivf(index).quantizer
                ).hnsw
            else:
                if opq:
                    idmap = faiss.downcast_index(index.index)
                else:
                    idmap = index
                hnsw = faiss.downcast_index(idmap.index).hnsw
            assert hnsw.efSearch == 16
            retriever.set_efsearch(64)
            assert hnsw.efSearch == 64
        else:
            # Do nothing
            retriever.set_efsearch(64)

    @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_normalize(self, metric: str, dtype, v: np.ndarray):
        cfg = RetrieverFaissCPU.Config(D, metric=metric)
        if metric == "l2":
            index = faiss.IndexFlatL2(D)
        else:
            index = faiss.IndexFlatIP(D)
        retriever = RetrieverFaissCPU(index, cfg)
        v = v.astype(dtype)
        nv = retriever.normalize(v)
        assert np.issubdtype(nv.dtype, np.float32)
        if metric == "cos":
            assert np.allclose((nv**2).sum(-1), np.ones(N))
        else:
            assert np.allclose(nv, v)

    @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
    def test_train(self, metric: str, v: np.ndarray):
        cfg = RetrieverFaissCPU.Config(D, metric=metric)
        if metric == "l2":
            index = faiss.IndexFlatL2(D)
        else:
            index = faiss.IndexFlatIP(D)
        retriever = RetrieverFaissCPU(index, cfg)
        retriever.train(v)
        assert retriever.index.is_trained

    @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
    @pytest.mark.parametrize("ids", [None, np.arange(N).astype(np.int64) + 1])
    def test_add(self, metric: str, ids: Optional[np.ndarray], v: np.ndarray):
        cfg = RetrieverFaissCPU.Config(D, metric=metric)
        if metric == "l2":
            index = faiss.IndexFlatL2(D)
        else:
            index = faiss.IndexFlatIP(D)
        index = faiss.IndexIDMap(index)
        retriever = RetrieverFaissCPU(index, cfg)
        retriever.add(v, ids=ids)
        dist, idxs = retriever.index.search(v, k=1)
        if metric == "l2":
            assert np.allclose(dist, np.zeros_like(dist))
            expected_ids = np.arange(N)
            if ids is not None:
                expected_ids += 1
            assert np.array_equal(idxs, expected_ids[:, None])
        else:
            dtable = (v[:, None] * v[None, :]).sum(-1)
            argmax_ids = np.argmax(dtable, axis=-1)
            expected_dist = dtable[np.arange(N), argmax_ids]
            if ids is None:
                expected_ids = argmax_ids
            else:
                expected_ids = argmax_ids + 1
            assert np.array_equal(idxs, expected_ids[:, None])
            assert np.allclose(dist, expected_dist[:, None])

    @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
    def test_search(self, metric: str, v: np.ndarray):
        cfg = RetrieverFaissCPU.Config(D, metric=metric)
        if metric == "l2":
            index = faiss.IndexFlatL2(D)
        else:
            index = faiss.IndexFlatIP(D)
        index = faiss.IndexIDMap(index)
        retriever = RetrieverFaissCPU(index, cfg)
        retriever.add(v)
        dist, ids = retriever.search(v, k=1)
        if metric == "l2":
            assert np.allclose(dist, np.zeros_like(dist))
        else:
            dtable = (v[:, None] * v[None, :]).sum(-1)
            expected_ids = np.argmax(dtable, axis=-1)
            expected_dist = dtable[np.arange(N), expected_ids]
            assert np.array_equal(ids, expected_ids[:, None])
            assert np.allclose(dist, expected_dist[:, None])

    def test_io(self, tmp_path: Path, retriever: RetrieverFaissCPU, v: np.ndarray):
        idx_path = tmp_path / "test.idx"
        cfg_path = tmp_path / "test.cfg"
        v = np.random.rand(N, D).astype(np.float32)
        retriever.add(v)
        retriever.save(idx_path, cfg_path)
        new_retriever = RetrieverFaissCPU.load(idx_path, cfg_path)
        assert len(new_retriever) == len(retriever)
