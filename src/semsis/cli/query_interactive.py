#!/usr/bin/env python3
import argparse
import enum
import fileinput
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Generator

import simple_parsing
import torch
from simple_parsing.helpers.fields import choice, field

from semsis.encoder import SentenceEncoder
from semsis.registry import get_registry
from semsis.retriever import load_backend_from_config
from semsis.typing import StrPath
from semsis.utils import Stopwatch

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stderr,
)
logger = logging.getLogger("semsis.cli.query_interactive")


def buffer_lines(
    input: StrPath, buffer_size: int = 1, prefix_string: str = ""
) -> Generator[list[str], None, None]:
    buf: list[str] = []
    with fileinput.input(
        [input], mode="r", openhook=fileinput.hook_encoded("utf-8")
    ) as f:
        for line in f:
            buf.append(prefix_string + line.strip())
            if len(buf) >= buffer_size:
                yield buf
                buf = []
        if len(buf) > 0:
            yield buf


class Format(str, enum.Enum):
    plain = "plain"
    json = "json"


@dataclass
class Config:
    """Configuration for query_interactive"""

    # Path to an index file.
    index_path: str
    # Path to an index configuration file.
    config_path: str

    # Path to an input file. If not specified, read from the standard input.
    input: str = "-"
    # Path to an output file.
    output: argparse.FileType("w") = field(default="-")
    # Output format.
    format: Format = Format.plain
    # Model name.
    model: str = "sentence-transformers/LaBSE"
    # Type of sentence representation.
    representation: str = choice(
        *get_registry("sentence_encoder").keys(), default="sbert"
    )

    # Use fp16.
    fp16: bool = False
    # Buffer size to query at a time.
    buffer_size: int = 128
    # Transfer the encoder to GPUs.
    gpu_encode: bool = False
    # Transfer the retriever to GPUs.
    gpu_retrieve: bool = False
    # Number of trials to measure the search time.
    ntrials: int = 1
    # Search the top-k nearest neighbor.
    topk: int = 1
    # Show the search time in milli seconds instead of seconds.
    msec: bool = False
    # Set the efSearch parameter for the HNSW indexes.
    # This corresponds to the beam width at the search time.
    efsearch: int = 16
    # Set the nprobe parameter for the IVF indexes.
    # This corresponds to the number of neighboring clusters to be searched.
    nprobe: int = 8
    # Prefix string.
    # This option is useful for `intfloat/e5-large`.
    prefix_string: str = ""


def main(args: Config) -> None:
    logger.info(args)

    encoder = SentenceEncoder.build(args.model, args.representation)
    if torch.cuda.is_available() and args.gpu_encode:
        if args.fp16:
            encoder = encoder.half()
        encoder = encoder.cuda()
        logger.info("The encoder is on the GPU.")

    retriever_type = load_backend_from_config(args.config_path)
    retriever = retriever_type.load(args.index_path, args.config_path)
    retriever.set_efsearch(args.efsearch)
    retriever.set_nprobe(args.nprobe)
    if torch.cuda.is_available() and args.gpu_retrieve:
        retriever.to_gpu_search()
    logger.info(f"Retriever configuration: {retriever.cfg}")
    logger.info(f"Retriever index size: {len(retriever):,}")

    def _print(string: str):
        print(string, file=args.output)

    timers = defaultdict(Stopwatch)
    ntrials = args.ntrials
    start_id = 0
    nqueryed = 0
    acctimes = defaultdict(float)
    unit = "msec" if args.msec else "sec"
    for lines in buffer_lines(
        args.input, buffer_size=args.buffer_size, prefix_string=args.prefix_string
    ):
        timers["encode"].reset()
        timers["retrieve"].reset()

        with timers["encode"].measure():
            querys = encoder.encode(lines).cpu().numpy()
        with timers["retrieve"].measure():
            dists, idxs = retriever.search(querys, k=args.topk)
        for _ in range(ntrials - 1):
            with timers["encode"].measure():
                querys = encoder.encode(lines).cpu().numpy()
            with timers["retrieve"].measure():
                dists, idxs = retriever.search(querys, k=args.topk)

        for i, line in enumerate(lines):
            uid = start_id + i

            # python<=3.9 does not support the match statement.
            if args.format == Format.plain:
                dist_str = " ".join([f"{x:.3f}" for x in dists[i].tolist()])
                idx_str = " ".join([str(x) for x in idxs[i].tolist()])
                _print(f"Q-{uid}\t{line}")
                _print(f"D-{uid}\t{dist_str}")
                _print(f"I-{uid}\t{idx_str}")
            elif args.format == Format.json:
                res = {
                    "i": uid,
                    "query": line,
                    "results": [
                        {"rank": rank, "distance": distance, "idx": idx}
                        for rank, (distance, idx) in enumerate(
                            zip(dists[i].tolist(), idxs[i].tolist())
                        )
                    ],
                }
                _print(json.dumps(res, ensure_ascii=False))

        times = {k: timer.total for k, timer in timers.items()}
        times["search"] = sum([timer.total for timer in timers.values()])
        if args.msec:
            times = {k: t * 1000 for k, t in times.items()}
        for name, t in times.items():
            acctimes[name] += t

        nqueryed = start_id + len(lines)
        for name, c in [("encode", "E"), ("retrieve", "R"), ("search", "S")]:
            t = times[name]
            at = times[name] / ntrials
            logger.info(f"T{c}-{start_id}:{nqueryed}\t{t:.1f} {unit}")
            logger.info(f"AT{c}-{start_id}:{nqueryed}\t{at:.1f} {unit}")

        start_id = nqueryed

    batch_avgtimes = defaultdict(float)
    single_avgtimes = defaultdict(float)
    for k, v in acctimes.items():
        batch_avgtimes[k] = v / ntrials
        single_avgtimes[k] = v / (nqueryed * ntrials)

    for k in acctimes.keys():
        logger.info(f"Total {k} time: {acctimes[k]:.1f} {unit}")
        logger.info(f"Average {k} time per batch: {batch_avgtimes[k]:.1f} {unit}")
        logger.info(
            f"Average {k} time per single query: {single_avgtimes[k]:.1f} {unit}"
        )


def cli_main() -> None:
    args = simple_parsing.parse(Config)
    main(args)


if __name__ == "__main__":
    cli_main()
