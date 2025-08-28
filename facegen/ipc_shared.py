# facegen/ipc_shared.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from multiprocessing import shared_memory as shm

@dataclass(frozen=True)
class ShmSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

class ShmArray:
    """Tiny shared array wrapper (create or attach). Methods ≤6 lines."""
    def __init__(self, *args, spec: ShmSpec, create: bool = True, **kwargs):
        nbytes = int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize
        self._shm = shm.SharedMemory(name=spec.name, create=create, size=nbytes if create else 0)
        self.a = np.ndarray(spec.shape, dtype=spec.dtype, buffer=self._shm.buf)

    def write(self, *args, src: np.ndarray, **kwargs) -> None:
        self.a[...] = np.asarray(src, dtype=self.a.dtype, order="C")

    def close(self, *args, **kwargs) -> None:
        self._shm.close()

    def unlink(self, *args, **kwargs) -> None:
        self._shm.unlink()
