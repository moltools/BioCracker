#!/usr/bin/env python3

"""Parse antiSMASH GenBank files and extract relevant features."""

import argparse
import logging

from biocracker.antismash import parse_region_gbk_file
from biocracker.config import LOGGER_LEVEL, LOGGER_NAME
from biocracker.paras import predict_amp_domain_substrate

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
    parser.add_argument("--thresh", type=float, default=0.1, help="Threshold for substrate prediction (default: 0.1)")
    return parser.parse_args()


def main() -> None:
    """
    Main function to parse the antiSMASH GenBank file.
    """
    args = cli()
    gbk_path = args.gbk
    target_name = "cand_cluster"
    targets = parse_region_gbk_file(gbk_path, top_level=target_name)
    logger.info(f" > Parsed {len(targets)} {target_name}(s) from {gbk_path}")

    for target in targets:
        region_accession = target.accession if target.accession is not None else 0
        logger.info(
            f" "
            f"> ({target_name} at {target.start} - {target.end}) "
            f"{target.record_id}.{target_name}{region_accession:03d} : {target.product_tags}"
        )
        len_start = max([len(str(gene.start)) for gene in target.genes])
        len_end = max([len(str(gene.end)) for gene in target.genes])
        for gene in target.genes:
            logger.info(
                f"   "
                f"> (gene at {gene.start:>{len_start}} - {gene.end:>{len_end}} on strand {gene.strand:>2}) "
                f"{gene.name} : {gene.product}"
            )
            for domain in gene.domains:
                domain_len_start = max([len(str(d.start)) for d in gene.domains])
                domain_len_end = max([len(str(d.end)) for d in gene.domains])
                if domain.kind == "AMP-binding":
                    domain_preds = predict_amp_domain_substrate(domain, pred_threshold=args.thresh)
                    repr_domain_preds = f"(preds: {[(name, score) for name, _, score in domain_preds]})"
                else:
                    repr_domain_preds = ""
                logger.info(
                    f"     "
                    f"> (domain at {domain.start:>{domain_len_start}} - {domain.end:>{domain_len_end}}) "
                    f"{domain.kind} {repr_domain_preds}"
                )


if __name__ == "__main__":
    main()
