"""Module for annotating genomic regions."""

import logging

from biocracker.utils.logging import Ctx
from biocracker.inference.registry import get_gene_models, get_domain_models
from biocracker.model.region import Region


log = logging.getLogger(__name__)


def annotate_region(region: Region) -> None:
    """
    Annotate all domains in all genes of the given region using registered models.
    
    :param region: the genomic region to annotate
    """
    log.debug(Ctx(region=region.id).prefix() + f"annotating {len(region.genes)} genes")

    for gene in region.iter_genes():
        gctx = Ctx(region=region.id, gene=gene.id)
        log.debug(gctx.prefix() + "annotating gene")

        # Gene inference
        for m in get_gene_models():
            mctx = Ctx(region=region.id, gene=gene.id, model=m.name)
            log.debug(mctx.prefix() + "running gene inference")

            results = m.predict(gene)
            for r in results:
                gene.annotations.add(r)

            log.debug(mctx.prefix() + f"added {len(results)} results")

        # Domain inference
        for domain in gene.iter_domains():
            dctx = Ctx(region=region.id, gene=gene.id, domain=domain.id)

            for m in get_domain_models():
                mctx = Ctx(region=region.id, gene=gene.id, domain=domain.id, model=m.name)
                log.debug(mctx.prefix() + f"running domain inference ({domain.type})")

                results = m.predict(domain)
                for r in results:
                    domain.annotations.add(r)

                log.debug(mctx.prefix() + f"added {len(results)} results")
