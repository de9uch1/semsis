import h5py
import numpy as np
import pytest
from numpy.typing import DTypeLike

from semsis.kvstore import KVStore

D = 8
N = 16


class TestKVStore:
    @pytest.fixture
    def f(self, tmp_path_factory: pytest.TempPathFactory):
        tmpdir = tmp_path_factory.mktemp("kvstore")
        return h5py.File(str(tmpdir / "tmp.hdf5"), mode="w")

    def test__init__(self, f: h5py.File):
        kvstore = KVStore(f)
        assert kvstore.f is f

    def test__len__(self, f: h5py.File):
        kvstore = KVStore(f)
        kvstore.new(D)
        assert len(kvstore) == 0

    def test_filename(self, f: h5py.File):
        kvstore = KVStore(f)
        assert kvstore.filename == f.filename

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_dtype(self, f: h5py.File, dtype):
        kvstore = KVStore(f)
        kvstore.new(D, dtype=dtype)
        assert np.issubdtype(kvstore.dtype, dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_new(self, f: h5py.File, dtype: DTypeLike):
        kvstore = KVStore(f)
        kvstore.new(D, dtype=dtype)
        k = kvstore.key
        v = kvstore.value
        assert isinstance(k, h5py.Dataset)
        assert isinstance(v, h5py.Dataset)
        assert np.issubdtype(k.dtype, dtype)
        assert np.issubdtype(v.dtype, np.int64)
        assert list(k.shape) == [0, D]
        assert list(v.shape) == [0]
        with pytest.raises(ValueError):
            kvstore.new(D)

    def test_add(self, f: h5py.File):
        keys = np.random.rand(N, D)
        values = np.ones(N).astype(np.int64)

        kvstore = KVStore(f)
        kvstore.new(D)
        kvstore.add(keys)
        assert len(kvstore) == N
        assert np.allclose(kvstore.key[:], keys)
        assert np.array_equal(kvstore.value, np.arange(N).astype(np.int64))
        kvstore.add(keys, values)
        assert len(kvstore) == N * 2
        assert np.allclose(kvstore.key[:], np.concatenate([keys, keys], axis=0))
        assert np.array_equal(
            kvstore.value[:], np.concatenate([np.arange(N), values], axis=0)
        )

    def test_open(self, tmp_path):
        path = str(tmp_path / "tmp.hdf5")
        with KVStore.open(path, mode="w") as kvstore:
            assert isinstance(kvstore.f, h5py.File)
            assert kvstore.f.name is not None
        assert kvstore.f.name is None

    def test_close(self, f: h5py.File):
        kvstore = KVStore(f)
        assert kvstore.f.name is not None
        kvstore.close()
        assert kvstore.f.name is None
