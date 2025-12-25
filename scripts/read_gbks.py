"""Parse linear readouts from parsed GenBank files."""

import argparse
import os

from biocracker.utils.logging import setup_logging, add_file_handler
from biocracker.utils.json import iter_json
from biocracker.model.region import Region
from biocracker.query.modules import linear_readout


def cli() -> argparse.Namespace:
    """
    Command line interface for parsing linear readouts from GenBank files.
    
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="output directory")
    return parser.parse_args()


def main() -> None:
    """
    Main function to parse linear readouts from GenBank files.
    """
    args = cli()
    os.makedirs(args.out, exist_ok=True)

    setup_logging(level="INFO")
    add_file_handler(os.path.join(args.out, "read_gbks.log"), level="INFO")

    for region_record in iter_json(args.jsonl, jsonl=True):
        region = Region.from_dict(region_record)
        readout = linear_readout(region)
        print(readout)
    

if __name__ == "__main__":
    main()
