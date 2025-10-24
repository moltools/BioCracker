#!/usr/bin/env python3

"""Parse antiSMASH GenBank files and extract linear readout information."""

import argparse
import logging

from biocracker.antismash import parse_region_gbk_file
from biocracker.config import LOGGER_LEVEL, LOGGER_NAME
from biocracker.readout import linear_readouts

# Setup logging
logger = logging.getLogger(LOGGER_NAME)
logging.basicConfig(level=LOGGER_LEVEL)


def cli() -> argparse.Namespace:
    """
    Parse command line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbk", type=str, required=True, help="Path to the antiSMASH GenBank file")
    parser.add_argument(
        "--toplevel",
        type=str,
        choices=["cand_cluster", "region"],
        default="cand_cluster",
        help="Top level feature to parse (default: cand_cluster)",
    )
    parser.add_argument(
        "--readlevel",
        type=str,
        choices=["rec", "gene"],
        default="rec",
        help='Level of readout, either "rec" for region/cluster level or "gene" for gene level (default: rec)',
    )
    parser.add_argument("--thresh", type=float, default=0.1, help="Threshold for substrate prediction (default: 0.1)")
    return parser.parse_args()


def main() -> None:
    """
    Main function to parse the antiSMASH GenBank file.
    """
    args = cli()
    gbk_path = args.gbk
    target_name = args.toplevel
    targets = parse_region_gbk_file(gbk_path, top_level=target_name)
    logger.info(f" > Parsed {len(targets)} {target_name}(s) from {gbk_path}")
    for target in targets:
        for readout in linear_readouts(target, level=args.readlevel):
            name = readout["rec"].name if args.readlevel == "gene" else readout["rec"].record_id
            logger.info(f"   > Readout {name}: {readout['readout']}")


if __name__ == "__main__":
    main()
