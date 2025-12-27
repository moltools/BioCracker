"""Parse linear readouts from parsed GenBank files."""

import argparse
import json
import os

from biocracker.utils.logging import setup_logging, add_file_handler
from biocracker.utils.json import iter_json
from biocracker.model.region import Region
from biocracker.query.modules import LinearReadout, PKSModule, NRPSModule, linear_readout


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

    print(f"Parsed {len(readouts)} linear readouts in total") 

    # Sort on readout ID
    readouts.sort(key=lambda r: r.id)

    # Only keep readouts with >= 2 modules
    readouts = [r for r in readouts if len(r.modules) >= 2]
    print(f"Parsed {len(readouts)} linear readouts with >= 2 modules")

    # Get specific readout
    readout_ids = ["BGC0000054", "BGC0000055", "BGC0000336"]
    specific_readouts = [r for r in readouts if r.id in readout_ids]
    for specific_readout in specific_readouts:
        print(f"Specific readout {specific_readout.id}: {specific_readout}")
        for module in specific_readout.biosynthetic_order():
            print(f"\t{module.substrate if isinstance(module, PKSModule) else module.substrate.name}", f"{module.role}", sep="\t")

    # Write all readouts to output JSONL
    out_jsonl = os.path.join(args.out, "linear_readouts.jsonl")
    with open(out_jsonl, "w") as out_f:
        for readout in readouts:
            out_f.write(json.dumps(readout.to_dict()) + "\n")
    

if __name__ == "__main__":
    main()
