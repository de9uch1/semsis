#!/usr/bin/env python3
import concurrent.futures
import ctypes.util
import logging
import math
import signal
import sys
from dataclasses import dataclass
from typing import Generator

import numpy as np
import simple_parsing
import torch
from simple_parsing.helpers.fields import choice
from tqdm import tqdm
from transformers import BatchEncoding

from semsis.encoder import SentenceEncoder
from semsis.encoder.tokenizer import Tokenizer
from semsis.kvstore import KVStore
from semsis.registry import get_registry
from semsis.typing import NDArrayFloat, NDArrayI64, StrPath
from semsis.utils import Stopwatch

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stderr,
)
logger = logging.getLogger("semsis.cli.store_kv")


@dataclass
class Batch:
    """Mini-batch class.

    inputs (BatchEncoding): Model inputs.
    ids (NDArrayI64): Sample IDs.
    """

    inputs: BatchEncoding
    ids: NDArrayI64


@dataclass
class Dataset:
    """Dataset class.

    sequences (list[list[int]]): Token ID sequences.
    lengths (NDArrayI64): Lengths of each sequence.
    """

    sequences: list[list[int]]
    lengths: NDArrayI64

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


def read_lines(file: StrPath, prefix_string: str = "") -> Generator[str, None, None]:
    """Read lines.

    Args:
        file (StrPath): Input file.
        prefix_string (str): Prefix string to be added for each line.
    """
    with open(file, mode="r") as f:
        for line in f:
            yield prefix_string + line.strip()


def prepare_dataset(
    file: StrPath,
    tokenizer: Tokenizer,
    num_workers: int = 1,
    chunk_size: int = 100000,
    prefix_string: str = "",
) -> Dataset:
    """Prepare a dataset.

    Args:
        file (StrPath): Input file.
        tokenizer (Tokenizer): Tokenizer.
        num_workers (int, optional): Number of workers.
        chunk_size (int): Size of data processed by each process at a time.
        prefix_string (str): Prefix string.

    Returns:
        Dataset: A dataset dataclass.
    """
    sequences: list[list[int]] = []
    if tokenizer.is_fast:
        chunk = []
        for line in tqdm(
            read_lines(file, prefix_string=prefix_string),
            desc="Preprocess the data",
            mininterval=1,
        ):
            chunk.append(line)
            if len(chunk) >= chunk_size:
                sequences += tokenizer.tokenize_batch(chunk)
                chunk = []
        if len(chunk) > 0:
            sequences += tokenizer.tokenize_batch(chunk)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=set_pdeathsig,
            initargs=(signal.SIGINT,),
        ) as executor:
            for tokenized_chunk in tqdm(
                executor.map(
                    tokenizer.tokenize,
                    read_lines(file, prefix_string=prefix_string),
                    chunksize=chunk_size,
                ),
                desc="Preprocess the data",
                mininterval=1,
            ):
                sequences.append(tokenized_chunk)

    return Dataset(sequences, np.array([len(seq) for seq in sequences]))


@dataclass
class Config:
    """Configuration for store_kv"""

    # Path to an input file.
    input: str
    # Path to the key-value store file.
    output: str = "kv.bin"
    # Model name.
    model: str = "sentence-transformers/LaBSE"
    # Type of sentence representation.
    representation: str = choice(
        *get_registry("sentence_encoder").keys(), default="sbert"
    )
    # Use fp16.
    fp16: bool = False
    # Batch size.
    batch_size: int = 128
    # Number of workers for preprocessing the data.
    workers: int = 16
    # Chunk size for multi-processing.
    chunk_size: int = 100000

    # Prefix string.
    # This option is useful for `intfloat/e5-large`.
    prefix_string: str = ""


def main(args: Config) -> None:
    logger.info(args)

    timer = Stopwatch()
    encoder = SentenceEncoder.build(args.model, args.representation)
    tokenizer = encoder.tokenizer

    logger.info("Start preprocessing the data")
    with timer.measure():
        dataset = prepare_dataset(
            args.input,
            tokenizer,
            args.workers,
            args.chunk_size,
            prefix_string=args.prefix_string,
        )
    logger.info(f"Dataset size: {len(dataset):,}")
    logger.info(f"Preprocessed the data in {timer.total:.1f} seconds.")

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        if args.fp16:
            encoder = encoder.half()

    logger.info("Start storing the keys and values.")
    with KVStore.open(args.output, mode="w") as kvstore:
        kvstore.new(encoder.get_embed_dim(), np.float16 if args.fp16 else np.float32)

        timer.reset()
        with timer.measure():
            for batch in dataset.yield_batches(tokenizer, args.batch_size):
                sentence_vectors: NDArrayFloat = (
                    encoder(batch.inputs.to(encoder.device)).cpu().numpy()
                )
                kvstore.add(sentence_vectors, batch.ids)
    logger.info(f"Stored the keys and values in {timer.total:.1f} seconds.")


def cli_main() -> None:
    args = simple_parsing.parse(Config)
    main(args)


if __name__ == "__main__":
    cli_main()
