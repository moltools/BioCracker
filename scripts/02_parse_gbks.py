"""Parse regions from GenBank files and annotate them using gene and domain models."""

import argparse
import os
import json
import glob
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Generator

from biocracker.utils.logging import setup_logging, add_file_handler
from biocracker.io.readers import load_regions
from biocracker.io.options import AntiSmashOptions
from biocracker.inference.registry import register_domain_model, register_gene_model
from biocracker.inference.model_paras import ParasModel
from biocracker.inference.model_pfam import PfamModel
from biocracker.pipelines.annotate_region import annotate_region


log = logging.getLogger(__name__)


_WORKER_OPTIONS = None
_WORKER_READY = False


def cli() -> argparse.Namespace:
    """
    Command line interface for parsing and annotating GenBank files.
    
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbks", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="output directory")
    parser.add_argument("--paras", type=str, required=False, help="path to all-substrates PARAS model file")
    parser.add_argument("--cache", type=str, required=False, help="cache directory")
    parser.add_argument("--hmms", type=str, required=False, help="directory with HMMs for gene models")
    parser.add_argument("--workers", type=int, default=1, help="number of parallel workers to use")
    return parser.parse_args()


def iter_gbks(folder: str) -> Generator[str, None, None]:
    """
    Iterate over all GenBank files in a folder.
    
    :param folder: folder path
    :yield: paths to GenBank files
    """
    with os.scandir(folder) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".gbk"):
                yield e.path


def _init_worker(
    cache_dir: str,
    paras_model_path: str | None,
    hmms_dir: str | None
) -> None:
    """
    Initialize a worker process by setting up its own cache and registering models.

    :param cache_dir: base cache directory
    :param paras_model_path: path to the PARAS model file (or None)
    :param hmms_dir: directory containing HMM files for gene models (or None)
    """
    global _WORKER_OPTIONS, _WORKER_READY

    per_worker_cache = os.path.join(cache_dir, f"worker_{os.getpid()}")
    os.makedirs(per_worker_cache, exist_ok=True)

    pm = ParasModel(threshold=0.1, keep_top=3, cache_dir=per_worker_cache, model_path=paras_model_path)
    register_domain_model(pm)

    if hmms_dir:
        hmm_files = glob.glob(os.path.join(hmms_dir, "*.hmm"))
        for hmm_file in hmm_files:
            label = Path(os.path.basename(hmm_file)).stem
            register_gene_model(PfamModel(hmm_path=hmm_file, label=label))

    _WORKER_OPTIONS = AntiSmashOptions(readout_level="cand_cluster")
    _WORKER_READY = True


def _process_one_gbk(gbk_file: str) -> list[str]:
    """
    Process a single GenBank file: load regions, annotate them, and return JSON lines.

    :param gbk_file: path to the GenBank file
    :return: list of JSON strings representing annotated regions
    """
    if not _WORKER_READY:
        raise RuntimeError("worker not initialized; call _init_worker first")
    
    regions = load_regions(gbk_file, _WORKER_OPTIONS)

    lines: list[str] = []
    for region in regions:
        annotate_region(region)
        lines.append(json.dumps(region.to_dict()))

    return lines


def main() -> None:
    """
    Main function to parse and annotate GenBank files.
    """
    t0 = time.time()
    args = cli()

    os.makedirs(args.out, exist_ok=True)

    setup_logging(level="INFO")
    add_file_handler(os.path.join(args.out, "parse_gbks.log"), level="INFO")

    out_jsonl = os.path.join(args.out, "regions.jsonl")

    # If cache dir is not given, set output dir as cache
    if args.cache is None:
        args.cache = args.out

    os.makedirs(args.cache, exist_ok=True)

    log.info(f"workers: {args.workers}")
    log.info(f"gbk dir: {args.gbks}")
    log.info(f"out: {args.out}")
    log.info(f"cache: {args.cache}")
    log.info(f"paras: {args.paras}")
    log.info(f"hmms: {args.hmms}")
    log.info(f"out jsonl: {out_jsonl}")

    completed_files = 0

    # Tune chunks by letting executor pull tasks naturally; keep a bounded number in flight
    max_in_flight = args.workers * 5

    gbk_iter = iter_gbks(args.gbks)

    with open(out_jsonl, "w") as out_f, ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.cache, args.paras, args.hmms)
    ) as ex:
        futures = set()

        # Prime the queue
        try:
            for _ in range(max_in_flight):
                gbk_file = next(gbk_iter)
                futures.add(ex.submit(_process_one_gbk, gbk_file))
        except StopIteration:
            pass

        while futures:
            # Write each result as it completes, and add a new task if available
            for fut in as_completed(futures):
                futures.remove(fut)
                lines = fut.result()
                if lines:
                    out_f.write("\n".join(lines) + "\n")
                completed_files += 1

                # Submit next
                try:
                    gbk = next(gbk_iter)
                    futures.add(ex.submit(_process_one_gbk, gbk))
                except StopIteration:
                    pass

                if completed_files % 100 == 0:
                    log.info(f"completed {completed_files} GenBank files")

                break  # exit for-loop to re-evaluate futures

    t1 = time.time()
    log.info(f"process files total: {completed_files}")
    log.info(f"total time: {t1 - t0:.1f} seconds ({(t1 - t0) / completed_files:.2f} sec/file)")


if __name__ == "__main__":
    main()
