"""Module contains text mining utilities for BioCracker."""

import re
from collections.abc import Iterable, Mapping

from biocracker.antismash import CandidateClusterRec, RegionRec

TOKENSPEC_SIDEROPHORE: dict = {
    "any": {
        # Direct words
        "NI-siderophore",
        "siderophore",
        "iron chelator",
        "feo-",
        "fe3+ transporter",
        # Terms often present around siderophores
        "entb",
        "enterobactin",
        "vibriobactin",
        "aerobactin",
        "pyoverdine",
        "yersiniabactin",
        "bacillibactin",
        "ferrichrome",
        "desferrioxamine",
        "myxochelin",
        "salicylate",
        "catecholate",
        "dhb",
        "2,3-dihydroxybenzoate",
        "isochorismate",
        "tonb",
        "tonb-dependent receptor",
        "fhu",
        "fep",
        "fepA",
        "fepB",
    },
    "rx": [
        re.compile(r"\b(iuc|ent|vib|fep|fhu)\w*\b", re.I),  # common locus prefixes
        re.compile(r"\b(iron[-\s]?uptake|iron[-\s]?transport)\b", re.I),
    ],
    "bonus_if": {
        # Add weight when these appear anywhere in the record
        "AMP-binding",
        "Heterocyclization",
        "NRPS",
        "siderophore biosynthesis",
    },
    "weight": 1.0,
    "bonus_weight": 0.5,
    "min_score": 4.0,
}

TOKENSPEC_LIPOPEPTIDE: dict = {
    "any": {
        "lipopeptide",
        "lipoinitiation",
        "starter c",
        "cstarter",
        "acyltransferase",
        "CAL_domain",
        "FAAL",
        "acyl-CoA ligase",
        "acyl-CoA synthetase",
    },
    "rx": [
        re.compile(r"\b(Cstarter|Condensation[_\s-]?starter)\b", re.I),
        re.compile(r"\b(acyl[-\s]?(ligase|co[a\-]?synthetase))\b", re.I),
    ],
    "bonus_if": {
        "AMP-binding",
        "PP-binding",
        "ACP",
        "Condensation",
    },
    "weight": 1.0,
    "bonus_weight": 0.5,
    "min_score": 2.0,
}

TOKENSPEC_METHYLTRANSFERAESE: dict = {
    "any": {
        # Common names / domain kinds
        "methyltransferase",
        "N-methyltransferase",
        "O-methyltransferase",
        "C-methyltransferase",
        "SAM-dependent methyltransferase",
        "S-adenosylmethionine",
        "AdoMet",
        "MT",
        "Methyltransferase",
    },
    "rx": [
        # Frequent abbreviations in annotations
        re.compile(r"\b[ONC]-?MT\b", re.I),  # OMT / NMT / CMT
        re.compile(r"\bSAM[-\s]?dependent\b", re.I),
    ],
    "bonus_if": {
        # Mild boosts when tailoring context is present
        "NRPS",
        "T1PKS",
        "PKS",
        "ACP",
        "PP-binding",
        "oxidoreductase",
    },
    "weight": 1.0,
    "bonus_weight": 0.3,
    "min_score": 4.0,
}

TOKENSPEC_HALOGENASES: dict = {
    "any": {
        # Keep very specific strings only (broad 'halogenase' moved to regex to avoid 'dehalogenase')
        "flavin-dependent halogenase",
        "flavin dependent halogenase",
        "FAD-dependent halogenase",
        "FAD dependent halogenase",
        "chlorinase",
        "fluorinase",
        "brominase",
        "halogenation",
    },
    "rx": [
        # Generic halogenase, but NOT 'dehalogenase'
        re.compile(r"(?<!de)\bhalogenase\b", re.I),
        # Tolerant to hyphen/space and optional 'dependent'
        re.compile(r"\b(flavin|FAD|FMN)[-\s]?(dependent\s*)?halogenase\b", re.I),
        # Classic gene names / families
        re.compile(r"\b(prnA|rebH|radH|chlA)\b", re.I),
        # Specific functional mentions you sometimes see
        re.compile(r"\btryptophan\s*\d?-?halogenase\b", re.I),
    ],
    "bonus_if": {
        # Cofactor/context clues and BGC context
        "FAD",
        "FMN",
        "flavin",
        "monooxygenase",
        "oxidoreductase",
        "chlorination",
        "bromination",
        "fluorination",
        "PKS",
        "NRPS",
        "trans-AT",
        "tAT",
    },
    "weight": 1.0,
    "bonus_weight": 0.4,
    "min_score": 4.0,
}

TOKENSPEC_GLYCOSYLTRANSFERASES: dict = {
    "any": {
        "glycosyltransferase",
        "glycosyl transferase",
        "glucosyltransferase",
        "rhamnosyltransferase",
        "xylosyltransferase",
        "galactosyltransferase",
        "O-glycosyltransferase",
        "C-glycosyltransferase",
        "UGT",
        "Glycosyltransferase",
        "GT-B fold",
    },
    "rx": [
        re.compile(r"\bUGT\d*\b", re.I),
        re.compile(r"\bGT(?:-|_)?\d+\b", re.I),  # GT-1, GT2, etc.
        re.compile(r"\b(glycosylat(e|ion|ing))\b", re.I),
    ],
    "bonus_if": {
        # sugar-handling context
        "dTDP",
        "TDP",
        "NDP",
        "glycoside",
        "sugar transfer",
        "rhamnose",
        "glucose",
        "mycarose",
        "NRPS",
        "PKS",
    },
    "weight": 1.0,
    "bonus_weight": 0.3,
    "min_score": 1.5,
}


def get_default_tokenspecs() -> Mapping[str, Mapping]:
    """
    Returns the default token specifications for text mining.

    :return: mapping of token names to their specifications
    """
    return {
        "siderophore": TOKENSPEC_SIDEROPHORE,
        "lipopeptide": TOKENSPEC_LIPOPEPTIDE,
        "methyltransferase": TOKENSPEC_METHYLTRANSFERAESE,
        "halogenase": TOKENSPEC_HALOGENASES,
        "glycosyltransferase": TOKENSPEC_GLYCOSYLTRANSFERASES,
    }


def _harvest_text_for_token_mining(rec: RegionRec | CandidateClusterRec) -> list[tuple[str, str]]:
    """
    Harvests textual fields from a RegionRec or CandidateClusterRec for token mining.

    :param rec: RegionRec or CandidateClusterRec object
    :return: list of (field_label, field_value) tuples containing textual hints
    """
    fields: list[tuple[str, str]] = []

    # Record-level fields
    rid = getattr(rec, "record_id", "") or ""
    fields.append(("record_id", str(rid)))

    # Product tags (region/cand_cluster)
    if getattr(rec, "product_tags", None):
        fields.append(("product_tags", " ".join(rec.product_tags)))

    # Genes and domains
    for g in rec.genes:
        # Gene-level textual hints
        for label, val in (
            ("gene_name", g.name),
            ("gene_product", g.product or ""),
            ("gene_note", g.note or ""),
            ("gene_desc", g.description or ""),
            ("gene_symbol", g.gene_symbol or ""),
            ("locus_tag", g.locus_tag or ""),
            ("protein_id", g.protein_id or ""),
        ):
            if val:
                fields.append((label, str(val)))

        # Domain-level hints
        for d in g.domains:
            if d.kind:
                fields.append(("domain_kind", d.kind))
            if d.name:
                fields.append(("domain_name", d.name))
            # Qualifiers often hold useful textual hints
            for k, v in (d.raw_qualifiers or {}).items():
                if isinstance(v, (list, tuple)):
                    fields.append((f"q:{k}", " ".join(map(str, v))))
                elif v:
                    fields.append((f"q:{k}", str(v)))

    return fields


def mine_virtual_tokens(rec: RegionRec | CandidateClusterRec, tokenspecs: Mapping[str, Mapping]) -> list[dict]:
    """ """
    fields = _harvest_text_for_token_mining(rec)

    # Flatten corpus for bonus checks
    full_blob = " ".join(t for _, t in fields).lower()

    results: list[dict] = []

    for token, spec in tokenspecs.items():
        weight = float(spec.get("weight", 1.0))
        bonus_weight = float(spec.get("bonus_weight", 0.5))
        min_score = float(spec.get("min_score", 1.0))

        any_terms: Iterable[str] = spec.get("any") or []
        rx_terms: Iterable[re.Pattern] = spec.get("rx") or []
        bonus_terms: Iterable[str] = spec.get("bonus_if") or []

        score = 0.0
        matches: list[tuple[str, str]] = []
        evidence_labels: set[str] = set()

        # Substring matches (case-insensitive)
        lower_terms = [t.lower() for t in any_terms]
        for label, text in fields:
            tl = text.lower()
            hit = False

            for t in lower_terms:
                if t and t in tl:
                    score += weight
                    hit = True

            for rx in rx_terms:
                if rx.search(text):
                    score += weight
                    hit = True

            if hit:
                matches.append((label, text))
                evidence_labels.add(label)

        # Bonus checks: bonus if any of bonus_terms appear anywhere
        for b in bonus_terms:
            if b.lower() in full_blob:
                score += bonus_weight

        if score >= min_score:
            results.append(
                {
                    "token": token,
                    "score": round(score, 3),
                    "matches": matches[:10],  # cap evidence length
                    "evidence": sorted(evidence_labels),
                }
            )

    # Highest score first, deterministic
    results.sort(key=lambda r: (-r["score"], r["token"]))
    return results
