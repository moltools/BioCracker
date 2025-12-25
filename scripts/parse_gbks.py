"""Parse regions from GenBank files and annotate them using gene and domain models."""

import argparse
import os
import json
import glob

from biocracker.utils.logging import setup_logging, add_file_handler
from biocracker.io.readers import load_regions
from biocracker.io.options import AntiSmashOptions
from biocracker.inference.registry import register_domain_model
from biocracker.inference.model_paras import ParasModel
from biocracker.pipelines.annotate_region import annotate_region


def cli() -> argparse.Namespace:
    """
    Command line interface for parsing and annotating GenBank files.
    
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbks", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="output directory")
    return parser.parse_args()


def main() -> None:
    """
    Main function to parse and annotate GenBank files.
    """
    args = cli()
    os.makedirs(args.out, exist_ok=True)

    setup_logging(level="INFO")
    add_file_handler(os.path.join(args.out, "parse_gbks.log"), level="INFO")

    register_domain_model(ParasModel(cache_dir=None, threshold=0.1, keep_top=3))

    options = AntiSmashOptions(readout_level="cand_cluster")

    gbk_iter = glob.iglob(f"{args.gbks}/*.gbk")

    out_jsonl = os.path.join(args.out, "regions.jsonl")
    with open(out_jsonl, "w") as out_f:
        for gbk_file in gbk_iter:
            regions = load_regions(gbk_file, options)
            for region in regions:
                annotate_region(region)
                out_f.write(json.dumps(region.to_dict()) + "\n")


if __name__ == "__main__":
    main()
