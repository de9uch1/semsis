#!/usr/bin/env python3
import concurrent.futures
import ctypes.util
import logging
import math
import signal
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from os import PathLike
from typing import Generator, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import BatchEncoding

from semsis.encoder import SentenceEncoder
from semsis.encoder.tokenizer import Tokenizer
from semsis.kvstore import KVStore
from semsis.registry import get_registry
from semsis.utils import Stopwatch

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger("semsis.cli.store_kv")


@dataclass
class Batch:
    """Mini-batch class.

    inputs (BatchEncoding): Model inputs.
    ids (np.ndarray): Sample IDs.
    """

    inputs: BatchEncoding
    ids: np.ndarray


@dataclass
class Dataset:
    """Dataset class.

    sequences (List[List[int]]): Token ID sequences.
    lengths (np.ndarray): Lengths of each sequence.
    """

    sequences: List[List[int]]
    lengths: np.ndarray

    def __len__(self):
        return len(self.lengths)

    def yield_batches(
        self, tokenizer: Tokenizer, batch_size: int
    ) -> Generator[Batch, None, None]:
        """Yields mini-batches.

        Args:
            tokenizer (Tokenizer): A tokenizer.
            batch_size (int): Batch size.

        Yields:
            Batch: A mini-batch.
        """
        sort_order = self.lengths.argsort()[::-1]
        num_data = len(self)
        for n in tqdm(range(math.ceil(len(self) / batch_size))):
            b, e = n * batch_size, min((n + 1) * batch_size, num_data)
            indices = sort_order[b:e]
            batch = tokenizer.collate([self.sequences[i] for i in indices])
            yield Batch(batch, indices)


def set_pdeathsig(sig: int) -> None:
    """Sets pdeathsig by prctl."""
    libc_path = ctypes.util.find_library("c")
    if libc_path is not None:
        libc = ctypes.cdll.LoadLibrary(libc_path)
        if hasattr(libc, "prctl"):
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, sig)


def prepare_dataset(
    file: PathLike,
    tokenizer: Tokenizer,
    num_workers: int = 1,
    chunk_size: int = 100000,
) -> Dataset:
    """Prepare a dataset.

    Args:
        file (os.PathLike): Input file.
        tokenizer (Tokenizer): Tokenizer.
        num_workers (int, optional): Number of workers.
        chunk_size (int): Size of data processed by each process at a time.

    Returns:
        Dataset: A dataset dataclass.
    """
    with open(file, mode="r") as f:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=set_pdeathsig,
            initargs=(signal.SIGINT,),
        ) as executor:
            sequences = list(
                tqdm(
                    executor.map(tokenizer.tokenize, f, chunksize=chunk_size),
                    desc="Preprocess the data",
                    mininterval=1,
                )
            )

    return Dataset(sequences, np.array([len(seq) for seq in sequences]))


def parse_args() -> Namespace:
    """Parses the command line arguments.

    Returns:
        Namespace: Command line arguments.
    """
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--input", type=str, required=True,
                        help="Input file.")
    parser.add_argument("--output", type=str, default="kv.bin",
                        help="Path to the key--value store.")
    parser.add_argument("--model", type=str, default="sentence-transformers/LaBSE",
                        help="Model name")
    parser.add_argument("--representation", type=str, default="sbert",
                        choices=get_registry("sentence_encoder").keys(),
                        help="Sentence representation type.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of workers to preprocess the data.")
    parser.add_argument("--chunk-size", type=int, default=100000,
                        help="Chunk size for multi-processing.")
    # fmt: on
    return parser.parse_args()


def main(args: Namespace) -> None:
    logger.info(args)

    timer = Stopwatch()
    encoder = SentenceEncoder.build(args.model, args.representation)
    tokenizer = encoder.tokenizer

    logger.info(f"Start preprocessing the data")
    with timer.measure():
        dataset = prepare_dataset(args.input, tokenizer, args.workers, args.chunk_size)
    logger.info(f"Dataset size: {len(dataset):,}")
    logger.info(f"Preprocessed the data in {timer.total:.1f} seconds.")

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        if args.fp16:
            encoder = encoder.half()

    logger.info(f"Start storing the keys and values.")
    with KVStore.open(args.output, mode="w") as kvstore:
        kvstore.new(encoder.get_embed_dim(), np.float16 if args.fp16 else np.float32)

        timer.reset()
        with timer.measure():
            for batch in dataset.yield_batches(tokenizer, args.batch_size):
                sentence_vectors = (
                    encoder(batch.inputs.to(encoder.device)).cpu().numpy()
                )
                kvstore.add(sentence_vectors, batch.ids)
    logger.info(f"Stored the keys and values in {timer.total:.1f} seconds.")


def cli_main() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
