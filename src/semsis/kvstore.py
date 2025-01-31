from __future__ import annotations

import contextlib
from typing import Generator, Optional

import h5py
import numpy as np
from numpy.typing import DTypeLike

from semsis.typing import NDArrayFloat, NDArrayI64, StrPath


class KVStore:
    """Key--value store class.

    Args:
        f (h5py.File): HDF5 file object.
    """

    def __init__(self, f: h5py.File) -> None:
        self.f = f

    @property
    def filename(self) -> str:
        """Returns the file name on the disk."""
        return self.f.filename

    @property
    def key(self) -> h5py.Dataset:
        """Return the key array."""
        return self.f["key"]

    @property
    def value(self) -> h5py.Dataset:
        """Return the value array."""
        return self.f["value"]

    def __len__(self) -> int:
        """Return the length of the key array."""
        return self.key.shape[0]

    @property
    def dtype(self) -> DTypeLike:
        """Return the dtype of the key array."""
        return self.key.dtype

    def add(self, keys: NDArrayFloat, values: Optional[NDArrayI64] = None) -> None:
        """Add the given key vectors into the key array.

        Args:
            keys (NDArrayFloat): The key vectors of shape `(n, dim)`.
            values (NDArrayI64, optional): The value IDs of shape `(n, dim)`.
              If values are not given, value IDs will be assigned incrementally.
        """
        ntotal = len(self)
        n = len(keys)
        if values is None:
            values = np.arange(ntotal, ntotal + n)
        self.key.resize(ntotal + n, axis=0)
        self.value.resize(ntotal + n, axis=0)
        self.key[ntotal:] = keys
        self.value[ntotal:] = values

    def new(self, dim: int, dtype: DTypeLike = np.float32) -> None:
        """Create the new arrays.

        Args:
            dim (int): The dimension size.
            dtype (DTypeLike): Dtype of the key array.
        """
        self.f.create_dataset("key", shape=(0, dim), dtype=dtype, maxshape=(None, dim))
        self.f.create_dataset("value", shape=(0,), dtype=np.int64, maxshape=(None,))

    @classmethod
    @contextlib.contextmanager
    def open(cls, path: StrPath, mode: str = "r") -> Generator[KVStore, None, None]:
        """Open a binary file of this kvstore.

        Args:
            path (StrPath): A path to the file.
            mode (str): Mode of this file objects.
              See https://docs.h5py.org/en/stable/high/file.html

        Yields:
            KVStore: This class.
        """
        f = h5py.File(path, mode=mode)
        self = cls(f)
        try:
            yield self
        finally:
            self.close()

    def close(self) -> None:
        """Close the file stream."""
        self.f.flush()
        self.f.close()
