#!/usr/bin/env python3
import contextlib
import logging
import math
import os
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from semsis.kvstore import KVStore
from semsis.retriever import Retriever
from semsis.utils import Stopwatch

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger("semsis.cli.build_retriever")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--kvstore", metavar="FILE", nargs="+", required=True,
                        help="Path to key--value store files.")
    parser.add_argument("--index-path", metavar="FILE", required=True,
                        help="Path to an index file.")
    parser.add_argument("--trained-index-path", metavar="FILE",
                        help="Path to a trained index file. If this option is not specified, "
                        "the final index path will be set.")
    parser.add_argument("--config-path", metavar="FILE", required=True,
                        help="Path to a configuration file.")
    parser.add_argument("--cpu", action="store_true",
                        help="Only use CPU.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")
    parser.add_argument("--backend", metavar="NAME", type=str, default="faiss-cpu",
                        help="Backend of the search engine.")
    parser.add_argument("--metric", metavar="TYPE", default="l2", choices=["l2", "ip", "cos"],
                        help="Distance function.")
    parser.add_argument("--chunk-size", metavar="N", type=int, default=1000000,
                        help="The number of data to be loaded at a time.")
    parser.add_argument("--train-size", metavar="N", type=int, default=1000000,
                        help="The number of training data.")
    parser.add_argument("--hnsw-nlinks", metavar="N", type=int, default=0,
                        help="[HNSW] The number of links per node.")
    parser.add_argument("--ivf-nlists", metavar="N", type=int, default=0,
                        help="[IVF] The number of centroids")
    parser.add_argument("--pq-nblocks", metavar="N", type=int, default=0,
                        help="[PQ] The number of sub-vectors")
    parser.add_argument("--opq", action="store_true",
                        help="Use OPQ to minimize the quantization error.")
    parser.add_argument("--pca", action="store_true",
                        help="Use PCA to reduce the dimension size.")
    parser.add_argument("--pca-dim", metavar="N", type=int, default=-1,
                        help="The dimension size which is reduced by PCA.")
    parser.add_argument("--append-sequential", action="store_true",
                        help="Append entries from the tail.")
    parser.add_argument("--save-checkpoint", action="store_true",
                        help="Save checkpoint after adding each key--value set.")
    # fmt: on
    return parser.parse_args()


def train_retriever(
    args: Namespace, training_vectors: np.ndarray, use_gpu: bool = False
) -> Retriever:
    train_size, dim = training_vectors.shape
    retriever_type = Retriever.get_cls(args.backend)
    dim = training_vectors.shape[1]
    cfg_dc = retriever_type.Config
    cfg_dict = {"dim": dim, "backend": args.backend, "metric": args.metric}

    def set_if_hasattr(name: str):
        if hasattr(cfg_dc, name):
            cfg_dict[name] = getattr(args, name)

    set_if_hasattr("hnsw_nlinks")
    set_if_hasattr("ivf_nlists")
    set_if_hasattr("pq_nblocks")
    set_if_hasattr("opq")
    set_if_hasattr("pca")
    set_if_hasattr("pca_dim")
    set_if_hasattr("fp16")
    cfg = retriever_type.Config(**cfg_dict)
    logger.info(cfg)
    logger.info(f"Input vector: {cfg.dim} dimension")

    if getattr(args, "trained_index_path", None):
        trained_index_path = args.trained_index_path
    else:
        trained_index_path = args.index_path

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


def main(args: Namespace) -> None:
    logger.info(args)

    timer = Stopwatch()
    use_gpu = torch.cuda.is_available() and not args.cpu
    chunk_size = args.chunk_size
    with contextlib.ExitStack() as stack:
        kvstores = [
            stack.enter_context(KVStore.open(fname, mode="r")) for fname in args.kvstore
        ]
        training_vectors = np.concatenate(
            [
                kvstore.key[: min(args.train_size // len(kvstores), len(kvstore))]
                for kvstore in kvstores
            ]
        )
        retriever = train_retriever(args, training_vectors, use_gpu=use_gpu)
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
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
