#!/usr/bin/env python3

"""Parse antiSMASH GenBank files and extract relevant features."""

import argparse
import logging

from biocracker.antismash import parse_region_gbk_file
from biocracker.config import LOGGER_NAME, LOGGER_LEVEL
from biocracker.parasect import predict_amp_domain_substrate


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
    return parser.parse_args()


def main() -> None:
    """
    Main function to parse the antiSMASH GenBank file.
    """
    args = cli()
    gbk_path = args.gbk
    regions = parse_region_gbk_file(gbk_path)
    logger.info(f" > Parsed {len(regions)} region(s) from {gbk_path}")

    for region in regions:
        region_accession = region.accession if region.accession is not None else 0
        logger.info(f" > (Region at {region.start} - {region.end}) {region.record_id}.region{region_accession:03d} : {region.product_tags}")
        len_start = max([len(str(gene.start)) for gene in region.genes])
        len_end = max([len(str(gene.end)) for gene in region.genes])
        for gene in region.genes:
            logger.info(f"   > (Gene at {gene.start:>{len_start}} - {gene.end:>{len_end}} on strand {gene.strand:>2}) {gene.name} : {gene.product}")
            for domain in gene.domains:
                domain_len_start = max([len(str(d.start)) for d in gene.domains])
                domain_len_end = max([len(str(d.end)) for d in gene.domains])
                if domain.kind == "AMP-binding":
                    domain_preds = predict_amp_domain_substrate(domain)
                    repr_domain_preds = f"(preds: {domain_preds})"
                else:
                    repr_domain_preds = ""
                logger.info(f"     > (Domain at {domain.start:>{domain_len_start}} - {domain.end:>{domain_len_end}}) {domain.kind} {repr_domain_preds}")

if __name__ == "__main__":
    main()
