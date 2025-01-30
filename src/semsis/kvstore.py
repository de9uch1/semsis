import contextlib
from os import PathLike
from typing import Any, Generator, Optional

import h5py
import numpy as np
from numpy.typing import DTypeLike


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

    def add(self, keys: np.ndarray, values: Optional[np.ndarray] = None) -> None:
        """Add the given key vectors into the key array.

        Args:
            keys (np.ndarray): The key vectors of shape `(n, dim)`.
            values (np.ndarray, optional): The value IDs of shape `(n, dim)`.
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
            dtype (DtypeLike): Dtype of the key array.
        """
        self.f.create_dataset("key", shape=(0, dim), dtype=dtype, maxshape=(None, dim))
        self.f.create_dataset("value", shape=(0,), dtype=np.int64, maxshape=(None,))

    @classmethod
    @contextlib.contextmanager
    def open(cls, path: PathLike, mode: str = "r") -> Generator["KVStore", Any, None]:
        """Open a binary file of this kvstore.

        Args:
            path (os.PathLike): A path to the file.
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
