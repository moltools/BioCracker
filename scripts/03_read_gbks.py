"""Parse linear readouts from parsed GenBank files."""

import argparse
import json
import os
import logging

from biocracker.utils.logging import setup_logging, add_file_handler
from biocracker.utils.json import iter_json
from biocracker.model.region import Region
from biocracker.query.modules import LinearReadout, linear_readout


log = logging.getLogger(__name__)


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

    readouts: list[LinearReadout] = []
    for region_record in iter_json(args.jsonl, jsonl=True):
        region = Region.from_dict(region_record)
        readout = linear_readout(region)
        readouts.append(readout)

    log.info(f"parsed {len(readouts)} linear readouts in total") 

    # Sort on readout ID
    readouts.sort(key=lambda r: r.id)

    # Only keep readouts with >= 2 modules
    readouts = [r for r in readouts if len(r.modules) >= 2]
    log.info(f"parsed {len(readouts)} linear readouts with >= 2 modules")

    # Report on how many readouts contain each found modifier
    modifier_counts: dict[str, int] = {}
    for readout in readouts:
        for modifier_name in set(readout.modifiers):
            if modifier_name not in modifier_counts:
                modifier_counts[modifier_name] = 0
            modifier_counts[modifier_name] += 1
    log.info(f"modifier presence across all {len(readouts)} readouts:")
    for modifier_name, count in modifier_counts.items():
        log.info(f"\t{modifier_name}: {count}")

    # Write all readouts to output JSONL
    out_jsonl = os.path.join(args.out, "linear_readouts.jsonl")
    with open(out_jsonl, "w") as out_f:
        for readout in readouts:
            out_f.write(json.dumps(readout.to_dict()) + "\n")
    

if __name__ == "__main__":
    main()
