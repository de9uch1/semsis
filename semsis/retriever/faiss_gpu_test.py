import faiss
import pytest
import torch

from semsis.retriever.faiss_gpu import faiss_index_to_cpu, faiss_index_to_gpu

if (
    not hasattr(faiss, "GpuIndex")
    or not torch.cuda.is_available()
    or faiss.get_num_gpus() <= 0
):
    pytest.skip("Skipping faiss-gpu tests", allow_module_level=True)

D = 8
N = 10


@pytest.fixture
def flat_index():
    return faiss.IndexFlatL2(D)


# TODO: check the precompute and fp16 options
@pytest.mark.parametrize("num_gpus", [-1, 0, 1, 2])
def test_faiss_index_to_gpu(flat_index: faiss.Index, num_gpus: int, fp16: bool):
    gpu_index = faiss_index_to_gpu(flat_index)
    ngpus = faiss.get_num_gpus() if num_gpus == -1 else num_gpus
    if ngpus == 1:
        assert isinstance(gpu_index, faiss.GpuIndex)
        assert gpu_index.getDevice() == 0
    else:
        assert isinstance(gpu_index, faiss.IndexReplicas)
        nshards = gpu_index.count()
        for i in range(nshards):
            gpu_index_i = faiss.downcast_index(gpu_index.at(i))
            assert isinstance(gpu_index_i, faiss.GpuIndex)
            assert gpu_index_i.getDevice() == i


def test_faiss_index_to_cpu(flat_index: faiss.Index):
    gpu_index = faiss_index_to_gpu(flat_index, 1)
    assert isinstance(gpu_index, faiss.GpuIndex)
    cpu_index = faiss_index_to_cpu(gpu_index)
    assert not isinstance(cpu_index, faiss.GpuIndex)
    assert not isinstance(cpu_index, faiss.Index)


# TODO: add tests for RetrieverFaissGPU
