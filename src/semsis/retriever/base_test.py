import pickle
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pytest
import yaml

from semsis.retriever import load_backend_from_config, register
from semsis.retriever.base import Retriever

D = 8


class RetrieverMock(Retriever):
    def __len__(self) -> int:
        ...

    @classmethod
    def build(cls, cfg: "Retriever.Config"):
        ...

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        ...

    def train(self, vectors: np.ndarray) -> None:
        ...

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        ...

    def search(self, querys: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @classmethod
    def load_index(cls, path: PathLike) -> Any:
        with open(path, mode="rb") as f:
            return pickle.load(f)

    def save_index(self, path: PathLike) -> None:
        with open(path, mode="wb") as f:
            pickle.dump(self.index, f)


class TestRetriever:
    @pytest.fixture
    def index(self):
        return object

    def test__init__(self, index):
        cfg = RetrieverMock.Config(D)
        retriever = RetrieverMock(index, cfg)
        assert retriever.index == index
        assert retriever.cfg == cfg

    class TestConfig:
        @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
        def test_save(self, tmp_path: Path, metric: str):
            cfg_path = tmp_path / "cfg.yaml"
            cfg = RetrieverMock.Config(D, metric=metric)
            cfg.save(cfg_path)
            with open(cfg_path, mode="r") as f:
                new_cfg = yaml.safe_load(f)
            assert asdict(cfg) == new_cfg

        @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
        def test_load(self, tmp_path: Path, metric: str):
            cfg_path = tmp_path / "cfg.yaml"
            cfg = RetrieverMock.Config(D, metric=metric)
            cfg.save(cfg_path)
            new_cfg = RetrieverMock.Config.load(cfg_path)
            assert cfg == new_cfg

    @pytest.mark.parametrize("metric", ["l2", "ip", "cos"])
    def test_io(self, tmp_path: Path, index, metric: str):
        idx_path = tmp_path / "test.idx"
        cfg_path = tmp_path / "test.cfg"
        cfg = RetrieverMock.Config(D, metric=metric)
        retriever = RetrieverMock(index, cfg)
        retriever.save(idx_path, cfg_path)
        new_retriever = RetrieverMock.load(idx_path, cfg_path)
        assert new_retriever.index == retriever.index
        assert new_retriever.cfg == retriever.cfg


def test_load_backend_from_config(tmp_path: Path):
    backend = "mock"

    @register(backend)
    class MockClass(RetrieverMock):
        @dataclass
        class Config(RetrieverMock.Config):
            foo: str = "foo"
            bar: int = 3

    cfg = MockClass.Config(D, backend)
    cfg_path = tmp_path / "cfg.yaml"
    cfg.save(cfg_path)
    assert issubclass(load_backend_from_config(cfg_path), MockClass)
