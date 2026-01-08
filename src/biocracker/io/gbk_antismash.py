"""Module for parsing GenBank files, including generic and antiSMASH-specific formats."""

import logging
from pathlib import Path
from typing import Any, Iterable

from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from biocracker.io.options import AntiSmashOptions, AntismashReadoutLevel
from biocracker.model.region import Region
from biocracker.model.gene import Gene, Strand
from biocracker.model.domain import Domain


log = logging.getLogger(__name__)


def _iter_regions(record: SeqRecord, readout_level: AntismashReadoutLevel) -> list[SeqFeature]:
    """
    Iterate over region features in a Biopython SeqRecord.
    
    :param record: Biopython SeqRecord object
    :return: list of region SeqFeature objects
    """
    return [f for f in record.features if f.type == readout_level]


def _iter_cds(record: SeqRecord, gene_identifiers: list[str] | None = None) -> list[SeqFeature]:
    """
    Iterate over CDS features in a Biopython SeqRecord.
    
    :param record: Biopython SeqRecord object
    :param gene_identifiers: list of gene feature types to look for
    :return: list of CDS SeqFeature objects
    .. note:: antiSMASH gene features are usually called 'CDS'
    """
    if gene_identifiers is None:
        gene_identifiers = ["CDS"]

    return [f for f in record.features if f.type in gene_identifiers]


def _iter_domains(record: SeqRecord, domain_identifiers: list[str] | None = None) -> list[SeqFeature]:
    """
    Iterate over domain features in a Biopython SeqRecord.
    
    :param record: Biopython SeqRecord object
    :param domain_identifiers: list of domain feature types to look for
    :return: list of domain SeqFeature objects
    .. note:: antiSMASH domain features are usually called 'aSDomain'
    """
    if domain_identifiers is None:
        domain_identifiers = ["aSDomain"]

    return [f for f in record.features if f.type in domain_identifiers]


def _start_end(feat: SeqFeature) -> tuple[Strand, int, int]:
    """
    Return (strand, start, end) as a tuple from a SeqFeature.
    
    :param feat: Biopython SeqFeature object
    :return: tuple of (strand, start, end)
    :raises ValueError: if strand is not 1 or -1
    .. note:: location is 0-based, end-exclusive
    """
    loc: FeatureLocation = feat.location

    match loc.strand:
        case 1:
            strand = Strand.FORWARD
        case -1:
            strand = Strand.REVERSE
        case _:
            raise ValueError(f"unexpected strand value: {loc.strand}")

    return strand, int(loc.start), int(loc.end)


def _in_bounds(child: SeqFeature, parent: SeqFeature) -> bool:
    """
    Check if a child feature is within the bounds of a parent feature.
    
    :param child: child SeqFeature object
    :param parent: parent SeqFeature object
    :return: True if child is within parent bounds, else False
    """
    _, cs, ce = _start_end(child)
    _, ps, pe = _start_end(parent)

    return ps <= cs and ce <= pe


def _q1(feat: SeqFeature, keys: Iterable[str]) -> str | None:
    """
    Return the first available qualifier value for any of the given keys, else None.

    :param feat: Biopython SeqFeature object
    :param keys: iterable of qualifier keys to check
    :return: qualifier value or None
    """
    for k in keys:
        vals = feat.qualifiers.get(k)
        if vals:
            return vals[0]
    
    return None


def _gene_name(feat: SeqFeature) -> str:
    """
    Get a gene name from a SeqFeature, or generate a default if none found.

    :param feat: Biopython SeqFeature object
    :return: gene name
    """
    name = _q1(feat, ("locus_tag", "gene", "protein_id", "Name"))

    if name:
        return name
    
    strand, s, e = _start_end(feat)

    return f"CDS_{s}_{e}_{strand.value}"


def _gene_rec_from_feat(feat: SeqFeature) -> Gene:
    """
    Create a Gene instance from a SeqFeature.

    :param feat: Biopython SeqFeature object
    :return: Gene instance
    """
    strand, s, e = _start_end(feat)
    name = _gene_name(feat)
    aa_seq = _q1(feat, ("translation",))

    return Gene(
        id=name,
        start=s,
        end=e,
        strand=strand,
        sequence=aa_seq,
    )


def _domain_rec_from_feat(feat: SeqFeature) -> Domain:
    """
    Create a Domain instance from a SeqFeature.

    :param feat: Biopython SeqFeature object
    :return: Domain instance
    """
    _, s, e = _start_end(feat)
    name = _q1(feat, ("label", "product", "note"))
    kind = _q1(feat, ("aSDomain", "domain", "label"))
    aa_seq = _q1(feat, ("translation",))

    return Domain(
        id=name,
        type=kind,
        start=s,
        end=e,
        sequence=aa_seq,
        raw_qualifiers={k: v for k, v in feat.qualifiers.items()},
    )


def collect_antismash_regions(
    input_file_path: Path | str,
    record: SeqRecord,
    options: AntiSmashOptions
) -> list[Region]:
    """
    Collect antiSMASH regions from a GenBank record.

    :param input_file_path: path to the GenBank file
    :param record: GenBank record
    :param options: parsing options
    :return: list of parsed regions
    """
    # Get the file name without extension
    if isinstance(input_file_path, str):
        input_file_path = Path(input_file_path)
    file_name = input_file_path.stem

    regions = _iter_regions(record, readout_level=options.readout_level)
    cds_list = _iter_cds(record, gene_identifiers=options.gene_identifiers)
    dom_list = _iter_domains(record, domain_identifiers=options.domain_identifiers)

    region_recs: list[Region] = []

    for reg in regions:
        _, rs, re = _start_end(reg)

        # Search for wanted qualifiers
        found_qualifiers: dict[str, Any] = {}
        for wanted in options.wanted_qualifiers:
            qualifiers = reg.qualifiers.get(wanted, []) or []
            found_qualifiers[wanted] = qualifiers

        # Genes inside region, sorted by coordinates
        gene_feats = [g for g in cds_list if _in_bounds(g, reg)]
        gene_feats.sort(key=lambda gf: (int(gf.location.start), int(gf.location.end)))

        # Make gene recs
        genes: list[Gene] = []
        for gf in gene_feats:
            g = _gene_rec_from_feat(gf)

            # Domains inside this gene, sorted by genomic start
            gene_doms = [df for df in dom_list if _in_bounds(df, gf)]
            gene_doms.sort(key=lambda df: (int(df.location.start), int(df.location.end)))
            dom_recs = [_domain_rec_from_feat(dd) for dd in gene_doms]

            g.domains = dom_recs
            genes.append(g)

        region_recs.append(Region(
            id=record.id,
            file_name=file_name,
            start=rs,
            end=re,
            qualifiers=found_qualifiers,
            genes=genes,
        ))

    return region_recs


def parse_antismash_gbk(path: Path, options: AntiSmashOptions) -> list:
    """
    Parse an antiSMASH-specific GenBank file.
    
    :param path: path to the antiSMASH GenBank file
    :return: list of parsed records
    """
    log.info(f"parsing antiSMASH GenBank file: {path}")

    out: list[Region] = []

    with open(path) as handle:
        for record in SeqIO.parse(handle, "genbank"):
            out.extend(collect_antismash_regions(path, record, options))

    return out
    