"""Tests for antismash parsing functions."""

import io
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from biocracker.antismash import (
    DomainRec,
    GeneRec,
    RegionRec,
    _collect_region,
    _domain_rec_from_feat,
    _gene_name,
    _gene_rec_from_feat,
    _in_bounds,
    _iter_cds,
    _iter_domains,
    _iter_regions,
    _q1,
    _start_end,
    parse_region_gbk_file,
    parse_region_gbk_string,
)


def make_region(start: int, end: int, product: str = "NRPS") -> SeqFeature:
    """
    Create a region feature.

    :param start: start position
    :param end: end position
    :param product: product qualifier
    :return: SeqFeature representing the region
    """
    return SeqFeature(
        FeatureLocation(start, end, strand=1),
        type="region",
        qualifiers={"product": [product]},
    )


def make_cds(start: int, end: int, strand: int = 1, qualifiers: dict | None = None) -> SeqFeature:
    """
    Create a CDS feature.

    :param start: start position
    :param end: end position
    :param strand: strand (1 or -1)
    :param qualifiers: additional qualifiers to include
    :return: SeqFeature representing the CDS"""
    q = {
        "product": ["hypothetical protein"],
        "translation": ["M" * 10],
        "locus_tag": ["geneA"],
        "EC_number": ["1.1.1.1"],
        "protein_id": ["PID123"],
        "gene": ["foo"],
    }
    if qualifiers:
        q.update(qualifiers)
    return SeqFeature(
        FeatureLocation(start, end, strand=strand),
        type="CDS",
        qualifiers=q,
    )


def make_domain(
    start: int, end: int, kind: str = "AMP-binding", label: str = "A", translation: str = "AAA"
) -> SeqFeature:
    """
    Create a aSDomain feature.

    :param start: start position
    :param end: end position
    :param kind: kind of domain
    :param label: label of the domain
    :param translation: amino acid translation
    :return: SeqFeature representing the domain
    """
    return SeqFeature(
        FeatureLocation(start, end, strand=1),
        type="aSDomain",
        qualifiers={
            "aSDomain": [kind],
            "label": [label],
            "translation": [translation],
            "note": [f"{kind} domain"],
        },
    )


def make_record() -> SeqRecord:
    """
    Build a small synthetic record:
    - region: 100..900
    - gene1 (fwd): 120..400, domains at 140..200 (A), 220..260 (PCP)
    - gene2 (rev): 500..800, domains at 650..680 (C), 700..750 (TE)  (note order on genome)

    :return: SeqRecord with features
    """
    rec = SeqRecord(Seq("N" * 1200), id="REC1", name="REC1", description="test")
    # Required by Biopython's GenBank writer:
    rec.annotations["molecule_type"] = "DNA"
    # Optional but harmless
    rec.annotations["topology"] = "linear"
    rec.annotations["organism"] = "Unknown"
    rec.annotations["data_file_division"] = "UNK"

    region = make_region(100, 900, product="NRPS")

    cds1 = make_cds(120, 400, strand=1, qualifiers={"locus_tag": ["geneFwd"], "product": ["enzyme X"]})
    dom1a = make_domain(140, 200, kind="AMP-binding", label="A", translation="AA1")
    dom1b = make_domain(220, 260, kind="PCP", label="PCP", translation="BB2")

    cds2 = make_cds(500, 800, strand=-1, qualifiers={"locus_tag": ["geneRev"], "product": ["enzyme Y"]})
    # Domains on genome left-to-right: 650..680 then 700..750
    dom2a = make_domain(650, 680, kind="C", label="C", translation="CC3")
    dom2b = make_domain(700, 750, kind="TE", label="TE", translation="DD4")

    rec.features.extend([region, cds1, cds2, dom1a, dom1b, dom2a, dom2b])
    return rec


def test_q1_and_start_end_and_gene_name_fallback() -> None:
    """
    Test _q1, _start_end, and _gene_name fallback behavior.
    """
    feat = make_cds(10, 50, strand=1, qualifiers={"locus_tag": ["abc"], "gene": ["gX"]})
    # _q1 returns first present key
    assert _q1(feat, ("nope", "gene", "locus_tag")) == "gX"
    # start/end/strand normalized
    strand, s, e = _start_end(feat)
    assert (strand, s, e) == (1, 10, 50)

    # Fallback name when no usual qualifiers
    feat_no_name = make_cds(100, 120, strand=-1, qualifiers={"product": ["p"]})
    for k in ("locus_tag", "gene", "protein_id", "Name"):
        feat_no_name.qualifiers.pop(k, None)
    fallback = _gene_name(feat_no_name)
    assert fallback == "CDS_100_120_rev"


def test_iterators_and_bounds() -> None:
    """
    Test the iterators for regions, CDS, and domains, and that features are within bounds.
    """
    rec = make_record()
    regs = _iter_regions(rec)
    cds = _iter_cds(rec)
    doms = _iter_domains(rec)
    assert len(regs) == 1
    assert len(cds) == 2
    assert len(doms) == 4

    # All CDS/domains should be within the region
    for f in cds + doms:
        assert _in_bounds(f, regs[0])


def test_domain_and_gene_rec_builders() -> None:
    """
    Test the construction of GeneRec and DomainRec from features.
    """
    rec = make_record()
    # Use the first CDS and a domain
    cds_feat = [f for f in rec.features if f.type == "CDS"][0]
    dom_feat = [f for f in rec.features if f.type == "aSDomain"][0]

    grecord = _gene_rec_from_feat(cds_feat)
    assert isinstance(grecord, GeneRec)
    assert grecord.name == "geneFwd"
    assert grecord.strand == 1
    assert grecord.start == 120 and grecord.end == 400
    assert grecord.protein_seq == "M" * 10
    assert grecord.product == "enzyme X"
    assert grecord.ec_number == "1.1.1.1"
    assert grecord.protein_id == "PID123"
    assert "translation" in grecord.raw_qualifiers

    drecord = _domain_rec_from_feat(dom_feat)
    assert isinstance(drecord, DomainRec)
    assert drecord.kind == "AMP-binding"
    assert drecord.name == "A"
    assert drecord.aa_seq == "AA1"
    assert drecord.start == 140 and drecord.end == 200
    assert "aSDomain" in drecord.raw_qualifiers


def test_collect_region_gene_and_domain_ordering() -> None:
    """
    Test the ordering of genes and domains within a region.
    """
    rec = make_record()
    regions = _collect_region(rec)
    assert len(regions) == 1
    R = regions[0]
    assert isinstance(R, RegionRec)
    assert R.record_id == "REC1"
    assert R.product_tags == ["NRPS"]
    assert len(R.genes) == 2

    g1, g2 = R.genes
    # Genes are sorted by genomic start
    assert g1.name == "geneFwd"
    assert g2.name == "geneRev"

    # Domains inside g1 (forward) remain left-to-right
    kinds_g1 = [d.kind for d in g1.domains]
    assert kinds_g1 == ["AMP-binding", "PCP"]

    # Domains inside g2 (reverse) should be reversed relative to genomic sort
    # On genome: C (650..680), TE (700..750) => sorted asc ["C","TE"]
    # For reverse gene, implementation reverses => ["TE","C"]
    kinds_g2 = [d.kind for d in g2.domains]
    assert kinds_g2 == ["TE", "C"]


def test_parse_region_gbk_string_roundtrip() -> None:
    """
    Test parsing a GenBank string via the full roundtrip: build record -> write GenBank -> parse GenBank
    """
    # Build record, write to a real GenBank string with Biopython, then parse via the function
    rec = make_record()
    buf = io.StringIO()
    SeqIO.write([rec], buf, "genbank")
    gbk_text = buf.getvalue()

    regions = parse_region_gbk_string(gbk_text)
    assert len(regions) == 1
    R = regions[0]
    assert [g.name for g in R.genes] == ["geneFwd", "geneRev"]
    # Confirm domain normalization persisted through the full parse
    assert [d.kind for d in R.genes[1].domains] == ["TE", "C"]


def test_parse_region_gbk_file(tmp_path: Path) -> None:
    """
    Test parsing a GenBank file via the full roundtrip: build record -> write GenBank file -> parse GenBank file
    """
    rec = make_record()
    gbk = tmp_path / "mini.gbk"
    with gbk.open("w") as handle:
        SeqIO.write([rec], handle, "genbank")

    regions = parse_region_gbk_file(str(gbk))
    assert len(regions) == 1
    assert isinstance(regions[0], RegionRec)
    assert len(regions[0].genes) == 2


def test_gene_name_priority_order() -> None:
    """
    Test the priority of gene name sources.
    """
    # Ensure priority: locus_tag > gene > protein_id > Name
    # Start by removing all, then add them progressively
    base = make_cds(10, 20, qualifiers={})
    for key in ("locus_tag", "gene", "protein_id", "Name"):
        base.qualifiers.pop(key, None)

    # Add only Name
    base.qualifiers["Name"] = ["N1"]
    assert _gene_name(base) == "N1"

    # Add protein_id overrides Name
    base.qualifiers["protein_id"] = ["PID"]
    assert _gene_name(base) == "PID"

    # Add gene overrides protein_id
    base.qualifiers["gene"] = ["gSym"]
    assert _gene_name(base) == "gSym"

    # Add locus_tag overrides gene
    base.qualifiers["locus_tag"] = ["LT"]
    assert _gene_name(base) == "LT"


def test_in_bounds_false_when_outside() -> None:
    """
    Test that _in_bounds returns False when child feature is outside parent bounds.
    """
    parent = make_region(100, 200)
    child = make_domain(50, 120)  # starts before
    assert not _in_bounds(child, parent)
    child2 = make_domain(150, 250)  # ends after
    assert not _in_bounds(child2, parent)
    # Exactly on edges is acceptable
    child3 = make_domain(100, 200)
    assert _in_bounds(child3, parent)
