"""Module contains methods for parsing antismash output gbk files."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Any, Literal

from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord


def _q1(feat: SeqFeature, keys: Iterable[str]) -> str | None:
    """
    Return the first available qualifier value for any of `keys`, else None.

    :param feat: Biopython SeqFeature object
    :param keys: Iterable of qualifier keys to check
    :return: the first found qualifier value or None
    """
    for k in keys:
        vals = feat.qualifiers.get(k)
        if vals:
            return vals[0]

    return None


def _start_end(feat: SeqFeature) -> tuple[int, int, int]:
    """
    Return (strand, start, end) as ints (0-based, end-exclusive like Biopython).

    :param feat: Biopython SeqFeature object
    :return: tuple of (strand, start, end)
    """
    loc: FeatureLocation = feat.location
    strand = int(loc.strand) if loc.strand in (1, -1) else 1

    return strand, int(loc.start), int(loc.end)


def _gene_name(feat: SeqFeature) -> str:
    """
    Get gene name from feature qualifiers, prioritizing common antiSMASH/CDS name fields.

    :param feat: Biopython SeqFeature object
    :return: gene name as string
    .. note:: fallback to a deterministic coordinate-based label if no name found
    """
    name = _q1(feat, ("locus_tag", "gene", "protein_id", "Name"))
    if name:
        return name

    strand, s, e = _start_end(feat)

    return f"CDS_{s}_{e}_{'rev' if strand == -1 else 'fwd'}"


@dataclass
class DomainRec:
    """
    Data class representing a domain record parsed from antiSMASH output.

    :param start: domain start position
    :param end: domain end position
    :param kind: domain kind (e.g. "AMP-binding", "PKS_KS", etc.)
    :param aa_seq: domain-level translation (AA)
    :param name: optional label if present
    :param raw_qualifiers: raw qualifiers dictionary
    """

    start: int
    end: int
    kind: str | None
    aa_seq: str | None
    name: str | None = None
    raw_qualifiers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert DomainRec instance to dictionary.

        :return: dictionary representation of DomainRec
        """
        d = asdict(self)
        return d


@dataclass
class GeneRec:
    """
    Data class representing a gene record parsed from antiSMASH output.

    :param name: gene name
    :param strand: gene strand
    :param start: gene start position
    :param end: gene end position
    :param protein_seq: CDS-level AA translation
    :param product: optional product description
    :param note: optional note
    :param description: optional description
    :param locus_tag: optional locus tag
    :param gene_symbol: optional gene symbol ('gene' qualifier)
    :param protein_id: optional protein ID
    :param ec_number: optional first EC number if present
    :param raw_qualifiers: raw qualifiers dictionary
    :param domains: list of DomainRec instances
    """

    name: str
    strand: int
    start: int
    end: int
    protein_seq: str | None
    product: str | None = None
    note: str | None = None
    description: str | None = None
    locus_tag: str | None = None
    gene_symbol: str | None = None
    protein_id: str | None = None
    ec_number: str | None = None

    # antiSMASH often tucks many details into qualifiers; keep them
    raw_qualifiers: dict[str, Any] = field(default_factory=dict)

    domains: list["DomainRec"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert GeneRec instance to dictionary.

        :return: dictionary representation of GeneRec
        """
        d = asdict(self)
        d["domains"] = [dom.to_dict() for dom in self.domains]
        return d


@dataclass
class CandidateClusterRec:
    """
    Data class representing a candidate cluster record parsed from antiSMASH output.

    :param record_id: cluster record ID
    :param accession: cluster number/accession in gbk file
    :param start: cluster start position
    :param end: cluster end position
    :param product_tags: list of product tags
    :param genes: list of GeneRec instances
    """

    record_id: str
    accession: int | None
    start: int
    end: int
    product_tags: list[str]
    genes: list[GeneRec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert CandidateClusterRec instance to dictionary.

        :return: dictionary representation of CandidateClusterRec
        """
        d = asdict(self)
        d["genes"] = [g.to_dict() for g in self.genes]
        return d


@dataclass
class RegionRec:
    """
    Data class representing a region record parsed from antiSMASH output.

    :param record_id: region record ID
    :param accession: region number/accession in gbk file
    :param start: region start position
    :param end: region end position
    :param product_tags: list of product tags
    :param genes: list of GeneRec instances
    """

    record_id: str
    accession: int | None
    start: int
    end: int
    product_tags: list[str]
    genes: list[GeneRec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert RegionRec instance to dictionary.

        :return: dictionary representation of RegionRec
        """
        d = asdict(self)
        d["genes"] = [g.to_dict() for g in self.genes]
        return d


def _iter_regions(record: SeqRecord) -> list[SeqFeature]:
    """
    Iterate over region features in a Biopython SeqRecord.

    :param record: Biopython SeqRecord object
    :return: list of region SeqFeature objects
    """
    return [f for f in record.features if f.type == "region"]


def _iter_candidate_clusters(record: SeqRecord) -> list[SeqFeature]:
    """
    Iterate over candidate cluster features in a Biopython SeqRecord.

    :param record: Biopython SeqRecord object
    :return: list of candidate cluster SeqFeature objects
    """
    return [f for f in record.features if f.type == "cand_cluster"]


def _iter_cds(record: SeqRecord) -> list[SeqFeature]:
    """
    Iterate over CDS features in a Biopython SeqRecord.

    :param record: Biopython SeqRecord object
    :return: list of CDS SeqFeature objects
    """
    return [f for f in record.features if f.type == "CDS"]


def _iter_domains(record: SeqRecord) -> list[SeqFeature]:
    """
    Iterate over domain features in a Biopython SeqRecord.

    :param record: Biopython SeqRecord object
    :return: list of domain SeqFeature objects
    .. note:: antiSMASH domain features are usually 'aSDomain'
    """
    return [f for f in record.features if f.type == "aSDomain"]


def _in_bounds(child: SeqFeature, parent: SeqFeature) -> bool:
    """
    Check if a child feature is within the bounds of a parent feature.

    :param child: child SeqFeature object
    :param parent: parent SeqFeature object
    :return: True if child is within parent bounds, else False
    """
    _, cs, ce = _start_end(child)
    _, ps, pe = _start_end(parent)
    return (ps <= cs) and (ce <= pe)


def _domain_rec_from_feat(feat: SeqFeature) -> DomainRec:
    """
    Create a DomainRec instance from a SeqFeature.

    :param feat: Biopython SeqFeature object
    :return: DomainRec instance
    """
    _, s, e = _start_end(feat)
    kind = _q1(feat, ("aSDomain", "domain", "label"))
    aa_seq = _q1(feat, ("translation",))
    name = _q1(feat, ("label", "product", "note"))
    return DomainRec(
        start=s,
        end=e,
        kind=kind,
        aa_seq=aa_seq,
        name=name,
        raw_qualifiers={k: v for k, v in feat.qualifiers.items()},
    )


def _gene_rec_from_feat(feat: SeqFeature) -> GeneRec:
    """
    Create a GeneRec instance from a SeqFeature.

    :param feat: Biopython SeqFeature object
    :return: GeneRec instance
    """
    strand, s, e = _start_end(feat)
    name = _gene_name(feat)
    prot = _q1(feat, ("translation",))

    # Common textual fields for search
    product = _q1(feat, ("product",))
    note = _q1(feat, ("note",))
    description = _q1(feat, ("function", "inference", "standard_name", "comment"))
    locus_tag = _q1(feat, ("locus_tag",))
    gene_symbol = _q1(feat, ("gene",))
    protein_id = _q1(feat, ("protein_id",))
    ec = _q1(feat, ("EC_number",))  # EC number(s) are often a list in 'EC_number'

    return GeneRec(
        name=name,
        strand=strand,
        start=s,
        end=e,
        protein_seq=prot,
        product=product,
        note=note,
        description=description,
        locus_tag=locus_tag,
        gene_symbol=gene_symbol,
        protein_id=protein_id,
        ec_number=ec,
        raw_qualifiers={k: v for k, v in feat.qualifiers.items()},
    )


def _collect_candidate_cluster(record: SeqRecord) -> list[CandidateClusterRec]:
    """ """
    clusters = _iter_candidate_clusters(record)
    cds_list = _iter_cds(record)
    dom_list = _iter_domains(record)

    cluster_recs: list[CandidateClusterRec] = []

    for cc in clusters:
        _, cs, ce = _start_end(cc)

        # Try several common qualifier names for the index
        acc_vals = (
            cc.qualifiers.get("candidate_cluster_number")
            or cc.qualifiers.get("cand_cluster_number")
            or cc.qualifiers.get("cluster_number")
            or cc.qualifiers.get("cluster_idx")
            or [0]
        )
        accession = int(acc_vals[0]) if acc_vals else None

        products = cc.qualifiers.get("product", []) or []

        # Genes inside cluster, sorted by coordinate
        gene_feats = [g for g in cds_list if _in_bounds(g, cc)]
        gene_feats.sort(key=lambda gf: (int(gf.location.start), int(gf.location.end)))

        genes: list[GeneRec] = []
        for gf in gene_feats:
            g = _gene_rec_from_feat(gf)

            # Domains inside this gene, sorted by genomic start
            gene_doms = [df for df in dom_list if _in_bounds(df, gf)]
            gene_doms.sort(key=lambda df: (int(df.location.start), int(df.location.end)))
            dom_recs = [_domain_rec_from_feat(dd) for dd in gene_doms]

            # Oritentation normalization
            if g.strand == -1:
                dom_recs = dom_recs[::-1]

            g.domains = dom_recs
            genes.append(g)

        cluster_recs.append(
            CandidateClusterRec(
                record_id=record.id,
                accession=accession,
                start=cs,
                end=ce,
                product_tags=products,
                genes=genes,
            )
        )

    return cluster_recs


def _collect_region(record: SeqRecord) -> list[RegionRec]:
    """
    Collect region records from a Biopython SeqRecord.

    :param record: Biopython SeqRecord object
    :return: list of RegionRec instances
    """
    regions = _iter_regions(record)
    cds_list = _iter_cds(record)
    dom_list = _iter_domains(record)

    region_recs: list[RegionRec] = []

    for reg in regions:
        _, rs, re = _start_end(reg)

        accessions = reg.qualifiers.get("region_number", [0])
        accession = int(accessions[0]) if accessions else None

        products = reg.qualifiers.get("product", []) or []

        # Genes inside region, sorted by coordinate
        gene_feats = [g for g in cds_list if _in_bounds(g, reg)]
        gene_feats.sort(key=lambda gf: (int(gf.location.start), int(gf.location.end)))

        # Make gene recs
        genes: list[GeneRec] = []
        for gf in gene_feats:
            g = _gene_rec_from_feat(gf)

            # Domains inside this gene, sorted by genomic start
            gene_doms = [df for df in dom_list if _in_bounds(df, gf)]
            gene_doms.sort(key=lambda df: (int(df.location.start), int(df.location.end)))
            dom_recs = [_domain_rec_from_feat(dd) for dd in gene_doms]

            # Orientation normalization
            # If gene is reverse, reverse the domain list so the readout is consistent (left to right)
            if g.strand == -1:
                dom_recs = dom_recs[::-1]

            g.domains = dom_recs
            genes.append(g)

        region_recs.append(
            RegionRec(
                record_id=record.id,
                accession=accession,
                start=rs,
                end=re,
                product_tags=products,
                genes=genes,
            )
        )

    return region_recs


def parse_region_gbk_string(
    src: str,
    top_level: Literal["region", "cand_cluster"] = "region",
) -> list[RegionRec]:
    """
    Parse antiSMASH region GenBank string into RegionRec instances.

    :param src: GenBank formatted string
    :param top_level: top-level feature to parse ('region' or 'cand_cluster')
    :return: list of RegionRec instances
    :raises AssertionError: if top_level is not 'region' or 'cand_cluster'
    """
    handle = StringIO(src)
    out: list[RegionRec] = []

    for record in SeqIO.parse(handle, "genbank"):
        if top_level == "region":
            out.extend(_collect_region(record))
        elif top_level == "cand_cluster":
            out.extend(_collect_candidate_cluster(record))
        else:
            raise ValueError(f"Unknown top_level '{top_level}'; must be 'region' or 'cand_cluster'")

    return out


def parse_region_gbk_file(filepath: str, top_level: Literal["region", "cand_cluster"] = "region") -> list[RegionRec]:
    """
    Parse antiSMASH region GenBank file into RegionRec instances.

    :param filepath: path to GenBank file
    :param top_level: top-level feature to parse ('region' or 'cand_cluster')
    :return: list of RegionRec instances
    """
    out: list[RegionRec] = []

    with open(filepath) as handle:
        for record in SeqIO.parse(handle, "genbank"):
            if top_level == "region":
                out.extend(_collect_region(record))
            elif top_level == "cand_cluster":
                out.extend(_collect_candidate_cluster(record))
            else:
                raise ValueError(f"Unknown top_level '{top_level}'; must be 'region' or 'cand_cluster'")

    return out
