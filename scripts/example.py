"""Parses antiSMASH GBK file with BioCracker."""

import argparse
import logging
import glob
import os
from pathlib import Path

from biocracker.utils.logging import setup_logging
from biocracker.io.readers import load_regions
from biocracker.io.options import AntiSmashOptions
from biocracker.inference.registry import DOMAIN_MODELS, GENE_MODELS, register_domain_model, register_gene_model
from biocracker.inference.model_paras import ParasModel
from biocracker.inference.model_pfam import PfamModel
from biocracker.pipelines.annotate_region import annotate_region
from biocracker.query.modules import PKSModule, NRPSModule, linear_readout


log = logging.getLogger(__name__)


def cli() -> argparse.Namespace:
    """
    Command line interface for example script.

    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbk", type=str, required=True, help="input antiSMASH GBK region file")
    parser.add_argument("--cache", type=str, required=True, help="path to cache directory")
    parser.add_argument("--hmms", type=str, required=False, help="path to custom HMM directory")
    parser.add_argument("--by-orf", action="store_true", help="group modules by their originating gene (ORF)")
    return parser.parse_args()


def pprint_module(module: PKSModule | NRPSModule) -> str:
    """
    Pretty print a module.
    
    :param module: module to print
    :return: string representation of the module
    """
    if isinstance(module, PKSModule):
        return f"PKS Module: extender_unit={module.substrate.extender_unit.value}, AT_loading_mode={module.anatomy.AT_loading_mode}"
    elif isinstance(module, NRPSModule):
        return f"NRPS Module: substrate={module.substrate.name}"
    else:
        return "Unknown Module"


def main() -> None:
    args = cli()

    setup_logging(logging.INFO)

    # Register domain and gene models
    register_domain_model(ParasModel(threshold=0.1, keep_top=3, cache_dir=args.cache))

    if args.hmms:
        hmm_files = glob.glob(os.path.join(args.hmms, "*.hmm"))
        for hmm_file in hmm_files:
            label = Path(os.path.basename(hmm_file)).stem
            register_gene_model(PfamModel(hmm_path=hmm_file, label=label))

    log.info(f"registered domain models: {list(DOMAIN_MODELS)}")
    log.info(f"registered gene models: {list(GENE_MODELS)}")

    # Configure options for loading regions
    options = AntiSmashOptions(readout_level="cand_cluster")

    # Load regions from GBK file; and annotate regions
    regions = load_regions(args.gbk, options)
    for region in regions:
        annotate_region(region)

    # Output some information about the parsed regions
    for region in regions:
        print("region id:", region.id, sep="\t")
        print("file name:", region.file_name, sep="\t")
        readout = linear_readout(region)
        log.info(readout)
        log.info(readout.modifiers)
        result = readout.biosynthetic_order(by_orf=args.by_orf)
        if args.by_orf:
            for gene_id, modules in result:
                for mi, module in enumerate(modules):
                    log.info(f"{gene_id}\t{mi+1}\t{pprint_module(module)}")
        else:
            for mi, module in enumerate(result):
                log.info(f"{mi+1}\t{pprint_module(module)}")


if __name__ == "__main__":
    main()
