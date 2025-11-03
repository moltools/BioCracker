#!/usr/bin/env python3

"""Parse antiSMASH GenBank files and extract linear readout information."""

import argparse
import logging
import os

from biocracker.antismash import parse_region_gbk_file
from biocracker.config import LOGGER_LEVEL, LOGGER_NAME
from biocracker.readout import NRPSModuleReadout, PKSModuleReadout, linear_readouts
from biocracker.text_mining import get_default_tokenspecs, mine_virtual_tokens

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
    parser.add_argument("--outfile", type=str, default=None, help="Path to log output file (default: None)")
    return parser.parse_args()


def main() -> None:
    """
    Main function to parse the antiSMASH GenBank file.
    """
    args = cli()

    if args.outfile is not None:
        # Delete if file exists and create new log file
        if os.path.exists(args.outfile):
            os.remove(args.outfile)

        file_handler = logging.FileHandler(args.outfile)
        file_handler.setLevel(LOGGER_LEVEL)
        logger.addHandler(file_handler)

    gbk_path = args.gbk
    target_name = args.toplevel
    targets = parse_region_gbk_file(gbk_path, top_level=target_name)
    logger.info(f" > Parsed {len(targets)} {target_name}(s) from {gbk_path}")
    for target in targets:
        for readout in linear_readouts(target, level=args.readlevel):
            name = readout["rec"].name if args.readlevel == "gene" else readout["rec"].record_id
            for module_idx, module in enumerate(readout["readout"], start=1):
                if isinstance(module, PKSModuleReadout):
                    logger.info(
                        f"   > {name} Module {module_idx:0>2} ({module['role']}): "
                        f"{module['at_source']}_{module['module_type']}"
                    )
                elif isinstance(module, NRPSModuleReadout):
                    substrate = module.get("substrate_name", "Unknown")
                    score = module.get("score", 0.0)
                    logger.info(f"   > {name} Module {module_idx:0>2} ({module['role']}): {substrate} ({score})")
                else:
                    logger.warning(f"   > {name} Module {module_idx:0>2}: Unknown module type")

        tokenspecs = get_default_tokenspecs()
        for mined_tokenspec in mine_virtual_tokens(target, tokenspecs):
            token_name = mined_tokenspec.get("token", "Unknown")
            token_score = mined_tokenspec.get("score", 0.0)
            logger.info(f"   > Mined tokenspec: {token_name} ({token_score})")


if __name__ == "__main__":
    main()
