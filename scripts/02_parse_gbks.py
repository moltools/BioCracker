"""Parse regions from GenBank files and annotate them using gene and domain models."""

import argparse
import os
import json
import glob
import time
import logging
from pathlib import Path

from biocracker.utils.logging import setup_logging, add_file_handler
from biocracker.io.readers import load_regions
from biocracker.io.options import AntiSmashOptions
from biocracker.inference.registry import GENE_MODELS, DOMAIN_MODELS, register_domain_model, register_gene_model
from biocracker.inference.model_paras import ParasModel
from biocracker.inference.model_pfam import PfamModel
from biocracker.pipelines.annotate_region import annotate_region


log = logging.getLogger(__name__)


def cli() -> argparse.Namespace:
    """
    Command line interface for parsing and annotating GenBank files.
    
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbks", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="output directory")
    parser.add_argument("--cache", type=str, required=True, help="cache directory")
    parser.add_argument("--hmms", type=str, required=False, help="directory with HMMs for gene models")
    return parser.parse_args()


def main() -> None:
    """
    Main function to parse and annotate GenBank files.
    """
    t0 = time.time()

    args = cli()
    os.makedirs(args.out, exist_ok=True)

    setup_logging(level="INFO")
    add_file_handler(os.path.join(args.out, "parse_gbks.log"), level="INFO")

    register_domain_model(ParasModel(cache_dir=args.cache, threshold=0.1, keep_top=3))

    if args.hmms:
        # Find all .hmm files in the provided directory; use filename (without extension) as label
        hmm_files = glob.glob(os.path.join(args.hmms, "*.hmm"))
        for hmm_file in hmm_files:
            label = Path(os.path.basename(hmm_file)).stem
            register_gene_model(PfamModel(hmm_path=hmm_file, label=label))

    log.info(f"registered domain models: {list(DOMAIN_MODELS)}")
    log.info(f"registered gene models: {list(GENE_MODELS)}")

    options = AntiSmashOptions(readout_level="cand_cluster")

    gbk_iter = glob.iglob(f"{args.gbks}/*.gbk")

    out_jsonl = os.path.join(args.out, "regions.jsonl")
    with open(out_jsonl, "w") as out_f:
        for gbk_file in gbk_iter:
            regions = load_regions(gbk_file, options)
            for region in regions:
                annotate_region(region)
                out_f.write(json.dumps(region.to_dict()) + "\n")

    te = time.time()
    elapsed = te - t0
    elapsed_mins = elapsed / 60.0
    elapsed_hrs = elapsed_mins / 60.0
    log.info(f"total time elapsed: {elapsed:.2f} seconds")
    log.info(f"total time elapsed: {elapsed_mins:.2f} minutes")
    log.info(f"total time elapsed: {elapsed_hrs:.2f} hours")


if __name__ == "__main__":
    main()
