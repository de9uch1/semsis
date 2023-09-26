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
from semsis.retriever import load_retriever
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
    parser.add_argument("--representation", type=str, default="sbert", choices=["avg", "cls", "sbert"],
                        help="Sentence representation type.")
    parser.add_argument("--backend", metavar="NAME", type=str, default="faiss",
                        help="Backend of the search engine.")
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
                        help="Number of trials to measure the search time.")
    parser.add_argument("--msec", action="store_true",
                        help="Show the search time in milli seconds instead of seconds.")
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

    retriever_type = load_retriever(args.backend)
    retriever = retriever_type.load(args.index_path, args.config_path)
    if torch.cuda.is_available() and args.gpu_retrieve:
        retriever.to_gpu_search()
    logger.info(f"Retriever configuration: {retriever.cfg}")
    logger.info(f"Retriever index size: {len(retriever):,}")

    encode_timer, retrieve_timer = Stopwatch(), Stopwatch()
    ntrials = args.ntrials
    start_id = 0
    nqueryed = 0
    acctimes = defaultdict(float)
    time_unit = "msec" if args.msec else "sec"
    for lines in buffer_lines(args.input, buffer_size=args.buffer_size):
        encode_timer.reset()
        retrieve_timer.reset()
        for _ in range(ntrials):
            with encode_timer.measure():
                querys = encoder.encode(lines).cpu().numpy()
            with retrieve_timer.measure():
                dists, idxs = retriever.search(querys, k=args.topk)

        for i in range(len(lines)):
            uid = start_id + i
            dist_str = " ".join([f"{x:.3f}" for x in dists[i].tolist()])
            idx_str = " ".join([str(x) for x in idxs[i].tolist()])
            print(f"Distance-{uid}\t{dist_str}")
            print(f"Result-{uid}\t{idx_str}")

        encode_time = encode_timer.total
        retrieve_time = retrieve_timer.total
        search_time = encode_time + retrieve_time
        if args.msec:
            encode_time *= 1000
            retrieve_time *= 1000
            search_time *= 1000

        nqueryed = start_id + len(lines)
        print(f"EncodeTime-{start_id}:{nqueryed - 1}\t{encode_time:.1f} {time_unit}")
        print(
            f"AverageEncodeTime-{start_id}:{nqueryed - 1}\t{encode_time / ntrials:.1f} {time_unit}"
        )
        print(
            f"RetrieveTime-{start_id}:{nqueryed - 1}\t{retrieve_time:.1f} {time_unit}"
        )
        print(
            f"AverageRetrieveTime-{start_id}:{nqueryed - 1}\t{retrieve_time / ntrials:.1f} {time_unit}"
        )
        print(f"SearchTime-{start_id}:{nqueryed - 1}\t{search_time:.1f} {time_unit}")
        print(
            f"AverageSearchTime-{start_id}:{nqueryed - 1}\t{search_time / ntrials:.1f} {time_unit}"
        )

        start_id = nqueryed
        acctimes["encode"] += encode_time
        acctimes["retrieve"] += retrieve_time
        acctimes["search"] += search_time

    avgtimes = defaultdict(float)
    for k, v in acctimes.items():
        avgtimes[k] = v / (nqueryed * ntrials)

    for k in acctimes.keys():
        print(f"Total {k} time: {acctimes[k]:.1f} {time_unit}")
        print(f"Average {k} time: {avgtimes[k]:.1f} {time_unit}")


def cli_main() -> None:
    args = parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
