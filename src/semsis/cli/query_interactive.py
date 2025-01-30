#!/usr/bin/env python3
import fileinput
import logging
import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from os import PathLike
from typing import Generator, List

import torch

from semsis.encoder import SentenceEncoder
from semsis.registry import get_registry
from semsis.retriever import load_backend_from_config
from semsis.utils import Stopwatch

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger("semsis.cli.query_interactive")


def buffer_lines(
    input: PathLike, buffer_size: int = 1
) -> Generator[List[str], None, None]:
    buf: List[str] = []
    with fileinput.input(
        [input], mode="r", openhook=fileinput.hook_encoded("utf-8")
    ) as f:
        for line in f:
            buf.append(line.strip())
            if len(buf) >= buffer_size:
                yield buf
                buf = []
        if len(buf) > 0:
            yield buf


def parse_args() -> Namespace:
    """Parses the command line arguments.

    Returns:
        Namespace: Command line arguments.
    """
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--input", type=str, default="-",
                        help="Input file.")
    parser.add_argument("--index-path", metavar="FILE",
                        help="Path to an index file.")
    parser.add_argument("--config-path", metavar="FILE", required=True,
                        help="Path to a configuration file.")
    parser.add_argument("--model", type=str, default="sentence-transformers/LaBSE",
                        help="Model name")
    parser.add_argument("--representation", type=str, default="sbert",
                        choices=get_registry("sentence_encoder").keys(),
                        help="Sentence representation type.")
    parser.add_argument("--gpu-encode", action="store_true",
                        help="Transfer the encoder to GPUs.")
    parser.add_argument("--gpu-retrieve", action="store_true",
                        help="Transfer the retriever to GPUs.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")
    parser.add_argument("--ntrials", type=int, default=1,
                        help="Number of trials to measure the search time.")
    parser.add_argument("--topk", type=int, default=1,
                        help="Search the top-k nearest neighbor.")
    parser.add_argument("--buffer-size", type=int, default=1,
                        help="Buffer size to query at a time.")
    parser.add_argument("--msec", action="store_true",
                        help="Show the search time in milli seconds instead of seconds.")
    parser.add_argument("--efsearch", type=int, default=16,
                        help="Set the efSearch parameter for the HNSW indexes. "
                        "This corresponds to the beam width at the search time.")
    parser.add_argument("--nprobe", type=int, default=8,
                        help="Set the nprobe parameter for the IVF indexes. "
                        "This corresponds to the number of neighboring clusters to be searched.")
    # fmt: on
    return parser.parse_args()


def main(args: Namespace) -> None:
    logger.info(args)

    encoder = SentenceEncoder.build(args.model, args.representation)
    if torch.cuda.is_available() and args.gpu_encode:
        encoder = encoder.cuda()
        if args.fp16:
            encoder = encoder.half()
        logger.info(f"The encoder is on the GPU.")

    retriever_type = load_backend_from_config(args.config_path)
    retriever = retriever_type.load(args.index_path, args.config_path)
    retriever.set_efsearch(args.efsearch)
    retriever.set_nprobe(args.nprobe)
    if torch.cuda.is_available() and args.gpu_retrieve:
        retriever.to_gpu_search()
    logger.info(f"Retriever configuration: {retriever.cfg}")
    logger.info(f"Retriever index size: {len(retriever):,}")

    timers = defaultdict(Stopwatch)
    ntrials = args.ntrials
    start_id = 0
    nqueryed = 0
    acctimes = defaultdict(float)
    unit = "msec" if args.msec else "sec"
    for lines in buffer_lines(args.input, buffer_size=args.buffer_size):
        timers["encode"].reset()
        timers["retrieve"].reset()
        for _ in range(ntrials):
            with timers["encode"].measure():
                querys = encoder.encode(lines).cpu().numpy()
            with timers["retrieve"].measure():
                dists, idxs = retriever.search(querys, k=args.topk)

        for i, line in enumerate(lines):
            uid = start_id + i
            dist_str = " ".join([f"{x:.3f}" for x in dists[i].tolist()])
            idx_str = " ".join([str(x) for x in idxs[i].tolist()])
            print(f"Q-{uid}\t{line}")
            print(f"D-{uid}\t{dist_str}")
            print(f"I-{uid}\t{idx_str}")

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
            print(f"T{c}-{start_id}:{nqueryed}\t{t:.1f} {unit}")
            print(f"AT{c}-{start_id}:{nqueryed}\t{at:.1f} {unit}")

        start_id = nqueryed

    batch_avgtimes = defaultdict(float)
    single_avgtimes = defaultdict(float)
    for k, v in acctimes.items():
        batch_avgtimes[k] = v / ntrials
        single_avgtimes[k] = v / (nqueryed * ntrials)

    for k in acctimes.keys():
        print(f"Total {k} time: {acctimes[k]:.1f} {unit}")
        print(f"Average {k} time per batch: {batch_avgtimes[k]:.1f} {unit}")
        print(f"Average {k} time per single query: {single_avgtimes[k]:.1f} {unit}")


def cli_main() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
