#!/usr/bin/env python3
import contextlib
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import simple_parsing
import torch
from simple_parsing import field
from tqdm import tqdm

from semsis.kvstore import KVStore
from semsis.retriever import Metric, Retriever
from semsis.retriever.base import RetrieverParam
from semsis.typing import NDArrayFloat
from semsis.utils import Stopwatch

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stderr,
)
logger = logging.getLogger("semsis.cli.build_retriever")


@dataclass
class Config:
    """Configuraiton for build_retriever"""

    # Path to an index file.
    index_path: str = field()
    # Path to an index configuration file.
    config_path: str = field()
    # Path to key--value store files.
    kvstore: list[str] = field(nargs="+")

    # Path to a trained index file.
    # If this option is not specified, the index path will be set.
    trained_index_path: Optional[str] = None
    # Only use CPU.
    cpu: bool = False
    # Backend of the search engine.
    backend: str = "faiss-cpu"
    # Distance function.
    metric: Metric = Metric.l2
    # The number of data to be loaded at a time.
    chunk_size: int = 1000000
    # The number of training data of the index.
    train_size: int = 1000000
    # Append entries to the tail.
    append_sequential: bool = False
    # Save checkpoint after adding each key--value set.
    save_checkpoint: bool = False


def parse_args() -> tuple[Config, RetrieverParam]:
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Config, "common")
    parser.add_arguments(RetrieverParam, "retriever_param")
    args = parser.parse_args()
    return args.common, args.retriever_param


def train_retriever(
    args: Config,
    retriever_param: RetrieverParam,
    training_vectors: NDArrayFloat,
    use_gpu: bool = False,
) -> Retriever:
    train_size, dim = training_vectors.shape
    retriever_type = Retriever.get_cls(args.backend)
    dim = training_vectors.shape[1]

    cfg = retriever_type.Config(
        dim=dim,
        backend=args.backend,
        metric=args.metric,
        **{
            k: v
            for k, v in asdict(retriever_param).items()
            if hasattr(retriever_type.Config, k)
        },
    )
    logger.info(cfg)
    logger.info(f"Input vector: {cfg.dim} dimension")

    trained_index_path = args.trained_index_path or args.index_path
    if os.path.exists(trained_index_path):
        trained_retriever = retriever_type.load(trained_index_path, args.config_path)
        if trained_retriever.cfg == cfg:
            logger.info(f"Load trained index: {trained_index_path}")
            return trained_retriever
        raise FileExistsError(trained_index_path)

    retriever = retriever_type.build(cfg)
    if use_gpu:
        retriever.to_gpu_train()

    timer = Stopwatch()
    logger.info(f"Train a retriever from {train_size:,} datapoints.")
    with timer.measure():
        retriever.train(training_vectors)
    logger.info(f"Training done in {timer.total:.1f} seconds")
    retriever.save(trained_index_path, args.config_path)
    return retriever_type.load(trained_index_path, args.config_path)


def main(args: Config, retriever_param: RetrieverParam) -> None:
    logger.info(args)

    timer = Stopwatch()
    use_gpu = torch.cuda.is_available() and not args.cpu
    chunk_size = args.chunk_size
    with contextlib.ExitStack() as stack:
        kvstores = [
            stack.enter_context(KVStore.open(fname, mode="r")) for fname in args.kvstore
        ]
        training_vectors: NDArrayFloat = np.concatenate(
            [
                kvstore.key[: min(args.train_size // len(kvstores), len(kvstore))]
                for kvstore in kvstores
            ]
        )
        retriever = train_retriever(
            args, retriever_param, training_vectors, use_gpu=use_gpu
        )
        if use_gpu:
            retriever.to_gpu_add()

        logger.info(f"Build a retriever in {args.index_path}")
        for kvstore in kvstores:
            logger.info(f"Add {len(kvstore):,} vectors from {kvstore.filename}")
            offset = len(retriever) if args.append_sequential else 0
            with timer.measure():
                for i in tqdm(
                    range(math.ceil(len(kvstore) / chunk_size)),
                    desc="Building a retriever",
                ):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(kvstore))
                    num_added = end_idx - start_idx
                    logger.info(f"Add vectors: {num_added:,} / {len(kvstore):,}")
                    retriever.add(
                        kvstore.key[start_idx:end_idx],
                        kvstore.value[start_idx:end_idx] + offset,
                    )
                    logger.info(f"Retriever index size: {len(retriever):,}")

            if args.save_checkpoint:
                retriever.save(args.index_path, args.config_path)

        if not args.save_checkpoint:
            retriever.save(args.index_path, args.config_path)

        logger.info(f"Added {len(retriever):,} datapoints")
        logger.info(f"Retriever index size: {len(retriever):,}")
    logger.info(f"Built the retriever in {timer.total:.1f} seconds")


def cli_main() -> None:
    args, retriever_param = parse_args()
    main(args, retriever_param)


if __name__ == "__main__":
    cli_main()
