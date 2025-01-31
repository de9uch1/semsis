import os
from typing import Union

import numpy as np
from numpy.typing import NDArray

StrPath = Union[str, os.PathLike[str]]
NDArrayI64 = NDArray[np.int64]
NDArrayF32 = NDArray[np.float32]
NDArrayF16 = NDArray[np.float16]
NDArrayFloat = Union[NDArrayF32, NDArrayF16]
