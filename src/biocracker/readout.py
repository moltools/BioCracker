"""Module contains functions for reading out RegionRec/CandidateClusterRec objects."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from biocracker.antismash import CandidateClusterRec, DomainRec, GeneRec, RegionRec
from biocracker.paras import predict_amp_domain_substrate

PKS_KINDS = {
    "PKS_KS",
    "PKS_AT",
    "PKS_KR",
    "PKS_DH",
    "PKS_ER",
}
PKS_TE_ALIASES = {"Thioesterase", "PKS_TE", "TE"}

# Common NRPS domain labels found in antiSMASH outputs
NRPS_A = "AMP-binding"
NRPS_C = "Condensation"
NRPS_T_ALIASES = {"PCP", "Thiolation", "T", "Peptidyl-carrier-protein"}
NRPS_E = "Epimerization"
NRPS_MT_ALIASES = {"N-Methyltransferase", "MT"}
NRPS_OX_ALIASES = {"Oxidase", "Ox", "Oxidoreductase"}
NRPS_R_ALIASES = {"Thioester-reductase", "R", "Reductase"}
NRPS_TE = "Thioesterase"


@dataclass
class PKSModuleReadout:
    """
    Readout dictionary for a PKS module.

    :param kind: kind of the module (always "PKS_module")
    :param module_type: type of the PKS module (PKS_A, PKS_B, PKS_C, PKS_D, UNCLASSIFIED)
    :param present_domains: list of present domain kinds in the module
    :param at_source: source of the acyltransferase domain
    :param module_index_in_gene: index of the module within the gene
    :param start: start position of the module
    :param end: end position of the module
    :param gene_name: name of the gene containing the module
    :param has_KR: whether the module has a Ketoreductase domain
    :param has_DH: whether the module has a Dehydratase domain
    :param has_ER: whether the module has an Enoylreductase domain
    :param has_AT: whether the module has an Acyltransferase domain
    """

    kind: Literal["PKS_module"]
    module_type: Literal["PKS_A", "PKS_B", "PKS_C", "PKS_D", "UNCLASSIFIED"]
    present_domains: list[str]
    at_source: Literal["CIS", "TRANS", "UNKNOWN"]
    module_index_in_gene: int
    start: int
    end: int
    gene_name: str

    # Anatomy
    has_active_KR: bool
    has_active_DH: bool
    has_active_ER: bool
    has_AT: bool
    role: Literal["starter", "elongation", "terminal", "starter+terminal", "unknown"]

    def __getitem__(self, key: str) -> object:
        """
        Allow dictionary-like access to attributes.

        :param key: attribute name
        :return: attribute value
        :raises KeyError: if attribute is not found
        """
        if not hasattr(self, key):
            raise KeyError(f"{key} not found in PKSModuleReadout")
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        """
        Allow dictionary-like access to attributes with a default value.

        :param key: attribute name
        :param default: default value if attribute is not found
        :return: attribute value or default
        """
        return getattr(self, key, default)


@dataclass
class NRPSModuleReadout:
    """
    Readout dictionary for an NRPS module.

    :param kind: kind of the module (always "NRPS_module")
    :param gene_name: name of the gene containing the module
    :param module_index_in_gene: index of the module within the gene
    :param start: start position of the module
    :param end: end position of the module
    :param present_domains: list of present domain kinds in the module
    :param has_C: whether the module has a Condensation domain
    :param has_T: whether the module has a Thiolation domain
    :param has_E: whether the module has an Epimerization domain
    :param has_MT: whether the module has a Methyltransferase domain
    :param has_Ox: whether the module has an Oxidase domain
    :param has_R: whether the module has a Reductase domain
    :param has_TE: whether the module has a Thioesterase domain
    :param role: role of the module (starter, elongation, terminal, starter+terminal, unknown)
    :param substrate: predicted substrate for the A-domain, if any
    :param score: prediction score for the substrate, if any
    :param raw_pred: raw prediction dictionary from the substrate predictor
    """

    kind: Literal["NRPS_module"]
    gene_name: str
    module_index_in_gene: int
    start: int
    end: int
    present_domains: list[str]

    # Core anatomy
    has_C: bool
    has_T: bool
    has_E: bool
    has_MT: bool
    has_Ox: bool
    has_R: bool
    has_TE: bool
    role: Literal["starter", "elongation", "terminal", "starter+terminal", "unknown"]

    # Substrate call from A-domain predictor
    substrate_name: str | None  # top predicted substrate name
    substrate_smiles: str | None  # top predicted substrate SMILES
    score: float | None  # top prediction score
    raw_preds: list[dict] | None  # raw prediction dictionaries from the predictor

    def __getitem__(self, key: str) -> object:
        """
        Allow dictionary-like access to attributes.

        :param key: attribute name
        :return: attribute value
        :raises KeyError: if attribute is not found
        """
        if not hasattr(self, key):
            raise KeyError(f"{key} not found in NRPSModuleReadout")
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        """
        Allow dictionary-like access to attributes with a default value.

        :param key: attribute name
        :param default: default value if attribute is not found
        :return: attribute value or default
        """
        return getattr(self, key, default)


def _domain_kinds(domains: Sequence[DomainRec]) -> set[str]:
    """
    Helper function to get the set of domain kinds from a list of DomainRec objects.

    :param domains: list of DomainRec objects
    :return: set of domain kinds
    """
    return {domain.kind for domain in domains if domain.kind}


def _is_kind(d: DomainRec, label: str | set[str]) -> bool:
    """
    Helper function to check if a DomainRec matches a given kind or set of kinds.

    :param d: DomainRec object
    :param label: kind label (str) or set of kind labels (set[str])
    :return: True if the domain matches the label, False otherwise
    """
    if not d.kind:
        return False

    if isinstance(label, set):
        return d.kind in label

    return d.kind == label


def _is_active_domain(d: DomainRec) -> bool:
    """ """
    if not d.kind:
        return True  # can't tell, assume active

    if d.kind not in {"PKS_KR", "PKS_DH", "PKS_ER"}:
        return True  # only check PKS accessory domains

    texts = []
    if d.name:
        texts.append(d.name)
    for _, vals in d.raw_qualifiers.items():
        if isinstance(vals, (list, tuple)):
            texts.extend(map(str, vals))
        else:
            texts.append(str(vals))

    blob = " ".join(texts).lower()

    # Common antiSMASH phrasing patterns
    inactive_flags = [
        "inactive",
        "nonfunctional",
        "non-functional",
        "inactivated",
        "broken",
        "truncated",
    ]
    return not any(flag in blob for flag in inactive_flags)


def _is_at_only_gene(g: GeneRec) -> bool:
    """
    Helper function to determine if a gene is an acyltransferase-only gene.

    :param g: GeneRec object
    :return: True if the gene is an AT-only gene, False otherwise
    """
    kinds = _domain_kinds(g.domains)
    return ("PKS_AT" in kinds) and all(k in {"PKS_AT"} for k in kinds)


def _find_upstream_at_only_gene(all_genes: Sequence[GeneRec], idx: int) -> GeneRec | None:
    """
    Return the nearest upstream gene that is AT-only (relative to all_genes order).

    :param all_genes: list of GeneRec objects
    :param idx: index of the current gene in all_genes
    :return: GeneRec object of the nearest upstream AT-only gene, or None if not found
    """
    for j in range(idx - 1, -1, -1):
        if _is_at_only_gene(all_genes[j]):
            return all_genes[j]

    return None


def _is_cstarter(d: DomainRec) -> bool:
    """
    Helper function to determine if a Condensation domain is a starter C-domain.

    :param d: DomainRec object
    :return: True if the domain is a starter C-domain, False otherwise
    """
    if not d.kind or d.kind != "Condensation":
        return False

    txts = []
    if d.name:
        txts.append(d.name)

    for _, vals in d.raw_qualifiers.items():
        # Join lists and scalars; qualifiers may be list[str]
        if isinstance(vals, (list, tuple)):
            txts.extend(map(str, vals))
        else:
            txts.append(str(vals))

    blob = " ".join(txts).lower()

    return ("starter" in blob) or ("cstarter" in blob) or ("condensation_starter" in blob)


def _gene_has_loading_domains(g: GeneRec) -> bool:
    """
    Helper function to determine if a gene has loading domains (CAL or ACP).

    :param g: GeneRec object
    :return: True if the gene has loading domains, False otherwise
    """
    kinds = {d.kind for d in g.domains if d.kind}
    names = {(d.name or "") for d in g.domains}

    # Domain-kind signals
    has_cal = ("CAL_domain" in kinds) or any("faal" in (n.lower()) for n in names)
    has_acp = ("PP-binding" in kinds) or ("ACP" in kinds) or any("acp" in (n.lower()) for n in names)

    return has_cal or has_acp


def _upstream_loading_cassette(all_genes: list[GeneRec], gi: int, max_bp: int = 20000) -> bool:
    """
    Check for loading cassette (CAL + ACP) in upstream genes within max_bp distance.

    :param all_genes: list of all GeneRec objects in the region/cluster
    :param gi: index of the current gene in all_genes
    :param max_bp: maximum base pair distance to search upstream
    :return: True if loading cassette is found upstream, False otherwise
    """
    cur_start = all_genes[gi].start

    seen_cal = False
    seen_acp = False
    for j in range(gi - 1, -1, -1):
        g = all_genes[j]
        if cur_start - g.end > max_bp:
            break
        kinds = {d.kind for d in g.domains if d.kind}
        names = {(d.name or "") for d in g.domains}
        if ("CAL_domain" in kinds) or any("faal" in n.lower() for n in names):
            seen_cal = True
        if ("PP-binding" in kinds) or ("ACP" in kinds) or any("acp" in n.lower() for n in names):
            seen_acp = True
        if seen_cal and seen_acp:
            return True

    return False


def _upstream_has_nrps_A(all_genes: Sequence[GeneRec], gi: int) -> bool:
    """
    Check if there is an upstream gene with an NRPS A-domain.

    :param all_genes: list of all GeneRec objects in the region/cluster
    :param gi: index of the gene g in all_genes
    """
    for j in range(gi - 1, -1, -1):
        if any(_is_kind(d, NRPS_A) for d in all_genes[j].domains):
            return True

    return False


def _upstream_has_pks_KS(all_genes: Sequence[GeneRec], gi: int, ks_start: int) -> bool:
    """
    Check if there is an upstream gene with a PKS KS-domain.

    :param all_genes: list of all GeneRec objects in the region/cluster
    :param gi: index of the gene g in all_genes
    :param ks_start: start position of the KS domain in gene g
    :return: True if there is an upstream KS-domain, False otherwise
    """
    # Genes upstream
    for j in range(gi - 1, -1, -1):
        if any(d.kind == "PKS_KS" for d in all_genes[j].domains):
            return True

    # Same gene, KS before this window's KS
    for d in all_genes[gi].domains:
        if d.kind == "PKS_KS" and d.start < ks_start:
            return True

    return False


def _standalone_pks_at_upstream(all_genes: Sequence[GeneRec], gi: int, ks_start: int, max_bp: int = 20000) -> bool:
    """
    Check for standalone PKS AT domain in upstream genes within max_bp distance.

    :param all_genes: list of all GeneRec objects in the region/cluster
    :param gi: index of the gene g in all_genes
    :param ks_start: start position of the KS domain in gene g
    :param max_bp: maximum base pair distance to search upstream
    :return: True if standalone PKS AT domain is found upstream, False otherwise
    """
    cur_start = ks_start

    # Same gene, before ks_start
    for d in all_genes[gi].domains:
        if d.kind == "PKS_AT" and d.end <= ks_start:
            return True

    # Upstream genes, within distance
    for j in range(gi - 1, -1, -1):
        g = all_genes[j]
        if cur_start - g.end > max_bp:
            break
        if any(d.kind == "PKS_AT" for d in g.domains):
            return True

    return False


def _is_last_global_KS(all_genes: Sequence[GeneRec], gi: int, ks_start: int) -> bool:
    """
    Check if the given KS domain is the last KS domain in the entire gene cluster/region.

    :param all_genes: list of all GeneRec objects in the region/cluster
    :param gi: index of the gene g in all_genes
    :param ks_start: start position of the KS domain in gene g
    :return: True if the KS domain is the last KS domain, False otherwise
    """
    # Same gene, any KS after this ks_start?
    for d in all_genes[gi].domains:
        if d.kind == "PKS_KS" and d.start > ks_start:
            return False

    # Downstream genes
    for j in range(gi + 1, len(all_genes)):
        if any(d.kind == "PKS_KS" for d in all_genes[j].domains):
            return False

    return True


def _downstream_has_te(all_genes: Sequence[GeneRec], gi: int, from_bp: int, max_bp: int = 20000) -> bool:
    """
    Check for downstream TE domain within max_bp distance.

    :param all_genes: list of all GeneRec objects in the region/cluster
    :param gi: index of the gene g in all_genes
    :param from_bp: position to start searching from
    :param max_bp: maximum base pair distance to search downstream
    :return: True if TE domain is found downstream, False otherwise
    """
    # Same gene after from_bp
    for d in all_genes[gi].domains:
        if d.kind in PKS_TE_ALIASES and d.end >= from_bp:
            return True

    # Next genes within window
    cur_end = from_bp
    for j in range(gi + 1, len(all_genes)):
        g = all_genes[j]
        if g.start - cur_end > max_bp:
            break
        if any(d.kind in PKS_TE_ALIASES for d in g.domains):
            return True

    return False


def _split_module_on_KS(domains: Sequence[DomainRec]) -> list[list[DomainRec]]:
    """
    Split a list of DomainRec objects into module windows anchored on PKS_KS domains.

    :param domains: list of DomainRec objects
    :return: list of lists of DomainRec objects, each sublist representing a module window
    """
    windows: list[list[DomainRec]] = []
    cur: list[DomainRec] = []

    for d in domains:
        if d.kind == "PKS_KS":
            # Start new module window anchored at this KS
            if cur:
                windows.append(cur)
            cur = [d]
        else:
            if cur:  # only append if we have started a module
                cur.append(d)

    if cur:
        windows.append(cur)

    return windows


def _classify_pks_window(window: Sequence[DomainRec]) -> tuple[str, set[str], bool, bool, bool, bool]:
    """
    Classify a PKS module based on the presence of domains in the given window.

    :param window: sequence of DomainRec objects representing a module window
    :return: tuple containing:
        - module type (str)
        - set of present domain kinds (set[str])
        - has_active_KR (bool)
        - has_active_DH (bool)
        - has_active_ER (bool)
        - has_AT (bool)
    """
    kinds_linear = [d.kind for d in window if d.kind in PKS_KINDS]
    present = set(kinds_linear)

    has_AT = "PKS_AT" in present
    has_active_KR = any("PKS_KR" in present and _is_active_domain(d) for d in window if d.kind == "PKS_KR")
    has_active_DH = any("PKS_DH" in present and _is_active_domain(d) for d in window if d.kind == "PKS_DH")
    has_active_ER = any("PKS_ER" in present and _is_active_domain(d) for d in window if d.kind == "PKS_ER")

    # Rules:
    # - KS + AT with neither KR nor DH nor ER => PKS_A
    # - KS + AT + KR (no DH and no ER) => PKS_B (KR after AT is naturally true in window order)
    # - KS + AT + KR + DH (no ER) => PKS_C
    # - KS + AT + KR + DH + ER => PKS_D
    if has_active_ER and has_active_DH and has_active_KR:
        mtype = "PKS_D"
    elif has_active_DH and has_active_KR and not has_active_ER:
        mtype = "PKS_C"
    elif has_active_KR and not has_active_DH and not has_active_ER:
        mtype = "PKS_B"
    elif not has_active_KR and not has_active_DH and not has_active_ER:
        mtype = "PKS_A"
    else:
        mtype = "UNCLASSIFIED"

    return mtype, present, has_active_KR, has_active_DH, has_active_ER, has_AT


def _window_bounds(window: Sequence[DomainRec]) -> tuple[int, int]:
    """
    Get the start and end positions of a module window.

    :param window: sequence of DomainRec objects representing a module window
    :return: tuple of (start, end) positions
    """
    return min(d.start for d in window), max(d.end for d in window)


def _pks_modules_for_gene(g: GeneRec, all_genes: Sequence[GeneRec], gi: int) -> list[PKSModuleReadout]:
    """
    Get PKS module readouts for a given gene.

    :param g: GeneRec object
    :param all_genes: list of all GeneRec objects in the region/cluster
    :param gi: index of the gene g in all_genes
    :return: list of PKSModuleReadout dictionaries
    """
    out: list[PKSModuleReadout] = []
    if all(d.kind != "PKS_KS" for d in g.domains):
        return out  # No KS domains, no modules

    windows = _split_module_on_KS(g.domains)
    for mi, win in enumerate(windows):
        mtype, present, has_active_KR, has_active_DH, has_active_ER, has_AT = _classify_pks_window(win)
        ks_start = win[0].start  # window is KS-anchored
        s, e = _window_bounds(win)

        if has_AT:
            at_src: Literal["CIS", "TRANS", "UNKNOWN"] = "CIS"
        else:
            at_src = "TRANS" if _find_upstream_at_only_gene(all_genes, gi) is not None else "UNKNOWN"

        # Assign provisional PKS role
        has_te_in_window = any(d.kind in PKS_TE_ALIASES for d in win)
        upstream_has_ks = _upstream_has_pks_KS(all_genes, gi, ks_start)
        starter = _standalone_pks_at_upstream(all_genes, gi, ks_start) and not upstream_has_ks

        terminal_by_TE = False
        if _is_last_global_KS(all_genes, gi, ks_start):
            terminal_by_TE = has_te_in_window or _downstream_has_te(all_genes, gi, from_bp=e)

        if starter and terminal_by_TE:
            role: Literal["starter", "elongation", "terminal", "starter+terminal", "unknown"] = "starter+terminal"
        elif starter:
            role = "starter"
        elif terminal_by_TE:
            role = "terminal"
        else:
            role = "elongation"

        s, e = _window_bounds(win)
        out.append(
            PKSModuleReadout(
                kind="PKS_module",
                module_type=mtype,
                present_domains=sorted(present),
                at_source=at_src,
                module_index_in_gene=mi,
                start=s,
                end=e,
                gene_name=g.name,
                has_active_KR=has_active_KR,
                has_active_DH=has_active_DH,
                has_active_ER=has_active_ER,
                has_AT=has_AT,
                role=role,
            )
        )

    return out


def _nrps_modules_for_gene(
    g: GeneRec,
    all_genes: Sequence[GeneRec],
    gi: int,
    *,
    cache_dir_override: Path | str | None = None,
    model: object | None = None,
    pred_threshold: float = 0.5,
) -> list[NRPSModuleReadout]:
    """
    Get NRPS module readouts for a given gene.

    :param g: GeneRec object
    :param all_genes: list of all GeneRec objects in the region/cluster
    :param gi: index of the gene g in all_genes
    :param cache_dir_override: optional cache directory override for substrate prediction
    :param model: optional model object for substrate prediction
    :param pred_threshold: prediction threshold for substrate prediction
    :return: list of NRPSModuleReadout dictionaries
    """
    doms: list[DomainRec] = list(g.domains)
    out: list[NRPSModuleReadout] = []

    # Indices of A-domains in left-to-right order
    a_idx = [i for i, d in enumerate(doms) if _is_kind(d, NRPS_A)]
    if not a_idx:
        return out  # no A-domains, no modules

    for mi, ai in enumerate(a_idx):
        # Extend window backward by one if there is an immediately previous C (same gene)
        start_i = ai
        if ai - 1 >= 0 and _is_kind(doms[ai - 1], NRPS_C):
            start_i = ai - 1

        # Extend forward until (but not including) the next A-domain
        end_i = a_idx[mi + 1] if mi + 1 < len(a_idx) else len(doms)

        window = doms[start_i:end_i]
        present = [d.kind for d in window if d.kind]

        has_C = any(_is_kind(d, NRPS_C) for d in window)
        has_Cstarter = any(_is_cstarter(d) for d in window)
        has_T = any(_is_kind(d, NRPS_T_ALIASES) for d in window)
        has_E = any(_is_kind(d, NRPS_E) for d in window)
        has_MT = any(_is_kind(d, NRPS_MT_ALIASES) for d in window)
        has_Ox = any(_is_kind(d, NRPS_OX_ALIASES) for d in window)
        has_R = any(_is_kind(d, NRPS_R_ALIASES) for d in window)
        has_TE = any(_is_kind(d, NRPS_TE) for d in window)

        # Fallback evidence of a separate loading cassette upstream
        loading_upstream = _upstream_loading_cassette(all_genes, gi)
        upstream_has_A = _upstream_has_nrps_A(all_genes, gi)

        # Role heuristic
        is_first_module_in_gene = mi == 0

        starter = (
            has_Cstarter
            or (is_first_module_in_gene and loading_upstream and not upstream_has_A)
            or ((not has_C) and not upstream_has_A)
        )
        terminal = has_TE or has_R

        if starter and terminal:
            role: Literal["starter", "elongation", "terminal", "starter+terminal", "unknown"] = "starter+terminal"
        elif starter:
            role = "starter"
        elif terminal:
            role = "terminal"
        elif has_C:
            role = "elongation"
        else:
            role = "unknown"

        # Substrate prediction from the A-domain inside this window (the anchor)
        A = doms[ai]
        preds = predict_amp_domain_substrate(
            domain=A,
            cache_dir_override=cache_dir_override,
            model=model,
            pred_threshold=pred_threshold,
        )
        # Pull top call and score
        top_pred = preds[0] if preds else None
        if top_pred:
            substrate_name = top_pred.get("substrate_name")
            substrate_smiles = top_pred.get("substrate_smiles")
            score = top_pred.get("score")
        else:
            substrate_name = None
            substrate_smiles = None
            score = None

        s = min(d.start for d in window)
        e = max(d.end for d in window)

        out.append(
            NRPSModuleReadout(
                kind="NRPS_module",
                gene_name=g.name,
                module_index_in_gene=mi,
                start=s,
                end=e,
                present_domains=present,
                has_C=has_C,
                has_T=has_T,
                has_E=has_E,
                has_MT=has_MT,
                has_Ox=has_Ox,
                has_R=has_R,
                has_TE=has_TE,
                role=role,
                substrate_name=substrate_name,
                substrate_smiles=substrate_smiles,
                score=score,
                raw_preds=preds,
            )
        )

    return out


def linear_readouts(
    rec: RegionRec | CandidateClusterRec,
    cache_dir_override: Path | str | None = None,
    *,
    level: Literal["rec", "gene"] = "rec",
    model: object | None = None,
    pred_threshold: float = 0.5,
) -> Generator[dict, None, None]:
    """
    Reads out a RegionRec or CandidateClusterRec object and returns a list substrates.

    :param rec: RegionRec or CandidateClusterRec object to read out
    :param level: level of readout, either "rec" for region/cluster level or "gene" for gene level
    :return: Generator of substrate specificities per level as dictionaries
    :raises AssertionError: if level is not "rec" or "gene" or rec is not a RegionRec or CandidateClusterRec
    """
    assert level in {"rec", "gene"}, 'Level must be either "rec" or "gene"'
    assert isinstance(rec, (RegionRec, CandidateClusterRec)), "rec must be a RegionRec or CandidateClusterRec object"

    # Ensure genes are left-to-right
    genes = list(rec.genes)
    genes.sort(key=lambda g: (g.start, g.end))

    readout_rec: list[dict] = []

    for gi, gene in enumerate(genes):
        items: list[PKSModuleReadout | NRPSModuleReadout] = []

        # Collect AMP (A-domain) events for this gene
        items.extend(
            _nrps_modules_for_gene(
                gene,
                all_genes=genes,
                gi=gi,
                cache_dir_override=cache_dir_override,
                model=model,
                pred_threshold=pred_threshold,
            )
        )

        # Collect PKS module events
        items.extend(_pks_modules_for_gene(gene, genes, gi))

        # Merge in strict genomic order
        items.sort(key=lambda d: (int(d.get("start", 0)), int(d.get("end", 0))))

        if level == "gene" and items:
            yield {"rec": gene, "readout": items}
        else:
            readout_rec.extend(items)

    if level == "rec" and readout_rec:
        yield {"rec": rec, "readout": readout_rec}
