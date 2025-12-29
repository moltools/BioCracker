"""
Module for constructing linear readouts from genomic regions.

Note: upstream/downstream scans are genomic (coordinate-based), not biosynthetic!
"""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, overload

from biocracker.model.region import Region
from biocracker.model.gene import Gene, Strand
from biocracker.model.domain import Domain


PKS_TYPES = {
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


class ModuleType(Enum):
    """
    Enumeration of module types.
    
    :cvar NRPS: Nonribosomal Peptide Synthetase module
    :cvar PKS: Polyketide Synthase module
    """

    NRPS = "NRPS"
    PKS = "PKS"



class ModuleRole(Enum):
    """
    Enumeration of module roles.
    
    :cvar STARTER: Starter module
    :cvar ELONGATION: Elongation module
    :cvar TERMINAL: Terminal module
    :cvar STARTER_TERMINAL: Starter and terminal module
    :cvar UNKNOWN: Unknown role
    """

    STARTER = "starter"
    ELONGATION = "elongation"
    TERMINAL = "terminal"
    STARTER_TERMINAL = "starter+terminal"
    UNKNOWN = "unknown"


@dataclass
class Module(ABC):
    """
    Base class for a module in a linear readout.

    :param module_index_in_gene: index of the module within its gene
    :param start: starting position of the module
    :param end: ending position of the module
    :param gene_id: ID of the gene containing the module
    :param gene_strand: strand of the gene containing the module
    :param present_domains: list of domain types present in the module
    :param role: functional role of the module
    """
    module_index_in_gene: int
    start: int
    end: int
    gene_id: str
    gene_strand: Strand
    present_domains: list[str]
    role: ModuleRole

    @property
    @abstractmethod
    def type(self) -> ModuleType:
        """
        Abstract property to get the type of the module.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def substrate(self) -> Any:
        """
        Abstract property to get the substrate information for the module.
        """
        raise NotImplementedError
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Module object to a dictionary representation.

        :return: Dictionary representation of the Module
        """
        raise NotImplementedError
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Module":
        """
        Create a Module object from a dictionary representation.

        :param data: Dictionary representation of the Module
        :return: Module object
        """
        raise NotImplementedError
    

@dataclass
class NRPSAnatomy:
    """
    Anatomy of a Nonribosomal Peptide Synthetase (NRPS) module.

    :param has_C: presence of condensation domain
    :param has_T: presence of thiolation domain
    :param has_E: presence of epimerization domain
    :param has_MT: presence of methyltransferase domain
    :param has_Ox: presence of oxidase domain
    :param has_R: presence of reductase domain
    :param has_TE: presence of thioesterase domain
    """

    has_C: bool
    has_T: bool
    has_E: bool
    has_MT: bool
    has_Ox: bool
    has_R: bool
    has_TE: bool

    def to_dict(self) -> dict[str, bool]:
        """
        Convert the NRPSAnatomy object to a dictionary representation.

        :return: Dictionary representation of the NRPSAnatomy
        """
        return {
            "has_C": self.has_C,
            "has_T": self.has_T,
            "has_E": self.has_E,
            "has_MT": self.has_MT,
            "has_Ox": self.has_Ox,
            "has_R": self.has_R,
            "has_TE": self.has_TE,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, bool]) -> "NRPSAnatomy":
        """
        Create a NRPSAnatomy object from a dictionary representation.

        :param data: Dictionary representation of the NRPSAnatomy
        :return: NRPSAnatomy object
        """
        return cls(
            has_C=data.get("has_C", False),
            has_T=data.get("has_T", False),
            has_E=data.get("has_E", False),
            has_MT=data.get("has_MT", False),
            has_Ox=data.get("has_Ox", False),
            has_R=data.get("has_R", False),
            has_TE=data.get("has_TE", False),
        )


@dataclass
class NRPSSubstrate:
    """
    Substrate information for a Nonribosomal Peptide Synthetase (NRPS) module.

    :param name: name of the predicted substrate
    :param smiles: SMILES representation of the substrate
    :param score: confidence score of the substrate prediction
    """

    name: str | None
    smiles: str | None
    score: float | None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the NRPSSubstrate object to a dictionary representation.

        :return: Dictionary representation of the NRPSSubstrate
        """
        return {
            "name": self.name,
            "smiles": self.smiles,
            "score": self.score,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NRPSSubstrate":
        """
        Create a NRPSSubstrate object from a dictionary representation.

        :param data: Dictionary representation of the NRPSSubstrate
        :return: NRPSSubstrate object
        """
        return cls(
            name=data.get("name", None),
            smiles=data.get("smiles", None),
            score=data.get("score", None),
        )


class ATLoadingMode(Enum):
    """
    Enumeration of acyltransferase (AT) loading modes.

    :cvar CIS: cis-acting AT domain
    :cvar TRANS: trans-acting AT domain
    :cvar UNKNOWN: unknown AT loading mode
    """

    CIS = "cis"
    TRANS = "trans"
    UNKNOWN = "unknown"


@dataclass 
class PKSAnatomy:
    """
    Anatomy of a Polyketide Synthase (PKS) module.

    :param has_active_KR: presence of active ketoreductase domain
    :param has_active_DH: presence of active dehydratase domain
    :param has_active_ER: presence of active enoylreductase domain
    :param has_AT: presence of acyltransferase domain
    """
    AT_loading_mode: ATLoadingMode

    has_active_KR: bool
    has_active_DH: bool
    has_active_ER: bool

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the PKSAnatomy object to a dictionary representation.

        :return: Dictionary representation of the PKSAnatomy
        """
        return {
            "AT_loading_mode": self.AT_loading_mode.value,
            "has_active_KR": self.has_active_KR,
            "has_active_DH": self.has_active_DH,
            "has_active_ER": self.has_active_ER,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PKSAnatomy":
        """
        Create a PKSAnatomy object from a dictionary representation.

        :param data: Dictionary representation of the PKSAnatomy
        :return: PKSAnatomy object
        """
        raw = (data.get("AT_loading_mode") or "unknown").lower()
        AT_loading_mode = ATLoadingMode(raw) if raw in {"cis","trans","unknown"} else ATLoadingMode.UNKNOWN

        return cls(
            AT_loading_mode=AT_loading_mode,
            has_active_KR=data.get("has_active_KR", False),
            has_active_DH=data.get("has_active_DH", False),
            has_active_ER=data.get("has_active_ER", False),
        )


class PKSExtenderUnit(Enum):
    """
    Enumeration of PKS extender unit types.

    :cvar PKS_A: PKS extender unit type A
    :cvar PKS_B: PKS extender unit type B
    :cvar PKS_C: PKS extender unit type C
    :cvar PKS_D: PKS extender unit type D
    :cvar UNCLASSIFIED: unclassified extender unit type
    """

    PKS_A = "PKS_A"
    PKS_B = "PKS_B"
    PKS_C = "PKS_C"
    PKS_D = "PKS_D"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass
class PKSSubstrate:
    """
    Substrate information for a Polyketide Synthase (PKS) module.

    :param extender_unit: type of extender unit used in the PKS module
    """

    extender_unit: PKSExtenderUnit
    substituent_type: int | None = None

    def to_dict(self) -> dict[str, str]:
        """
        Convert the PKSSubstrate object to a dictionary representation.

        :return: Dictionary representation of the PKSSubstrate
        """
        return {
            "extender_unit": self.extender_unit.value,
            "substituent_type": self.substituent_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "PKSSubstrate":
        """
        Create a PKSSubstrate object from a dictionary representation.

        :param data: Dictionary representation of the PKSSubstrate
        :return: PKSSubstrate object
        """
        return cls(
            extender_unit=PKSExtenderUnit(data.get("extender_unit", "UNCLASSIFIED")),
            substituent_type=data.get("substituent_type", None),
        )


@dataclass
class NRPSModule(Module):
    """
    Nonribosomal peptide synthetase (NRPS) module.

    :param role: functional role of the module
    :param anatomy: anatomical features of the NRPS module
    :param substrate: predicted substrate information for the NRPS module
    """

    role: ModuleRole
    anatomy: NRPSAnatomy    
    predicted_substrate: NRPSSubstrate | None = None

    @property
    def type(self) -> ModuleType:
        """
        Get the type of the module.

        :return: ModuleType.NRPS
        """
        return ModuleType.NRPS

    @property
    def substrate(self) -> NRPSSubstrate | None:
        """
        Get the predicted substrate information for the NRPS module.

        :return: NRPSSubstrate object containing substrate information, or None if not available
        """
        return self.predicted_substrate
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the NRPSModule object to a dictionary representation.

        :return: Dictionary representation of the NRPSModule
        """
        return {
            "type": self.type.value,
            "module_index_in_gene": self.module_index_in_gene,
            "start": self.start,
            "end": self.end,
            "gene_id": self.gene_id,
            "gene_strand": self.gene_strand.value,
            "present_domains": self.present_domains,
            "role": self.role.value,
            "anatomy": self.anatomy.to_dict(),
            "predicted_substrate": self.predicted_substrate.to_dict() if self.predicted_substrate else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NRPSModule":
        """
        Create a NRPSModule object from a dictionary representation.

        :param data: Dictionary representation of the NRPSModule
        :return: NRPSModule object
        """
        anatomy_data = data.get("anatomy", {})
        substrate_data = data.get("predicted_substrate", None)

        return cls(
            module_index_in_gene=data["module_index_in_gene"],
            start=data["start"],
            end=data["end"],
            gene_id=data["gene_id"],
            gene_strand=Strand(data["gene_strand"]),
            present_domains=data["present_domains"],
            role=ModuleRole(data["role"]),
            anatomy=NRPSAnatomy.from_dict(anatomy_data),
            predicted_substrate=NRPSSubstrate.from_dict(substrate_data) if substrate_data else None,
        )


@dataclass
class PKSModule(Module):
    """
    Polyketide synthase (PKS) module.

    :param type: module type (PKS)
    :param role: functional role of the module
    :param anatomy: anatomical features of the PKS module
    """

    role: ModuleRole
    anatomy: PKSAnatomy

    @property
    def type(self) -> ModuleType:
        """
        Get the type of the module.

        :return: ModuleType.PKS
        """
        return ModuleType.PKS

    @property
    def substrate(self) -> PKSSubstrate:
        """
        Get the predicted substrate information for the PKS module.

        :return: PKSSubstrate object containing substrate information
        """
        # Configure factory type
        def setup_substrate(extender_unit: PKSExtenderUnit) -> PKSSubstrate:
            return PKSSubstrate(extender_unit=extender_unit)

        # Rules:
        # - KS + AT with neither KR nor DH nor ER => PKS_A
        # - KS + AT + KR (no DH and no ER) => PKS_B (KR after AT is naturally true in window order)
        # - KS + AT + KR + DH (no ER) => PKS_C
        # - KS + AT + KR + DH + ER => PKS_D
        # - else UNCLASSIFIED
        # Note: assumes that presence of AT domain is already established
        match (
            self.anatomy.has_active_KR,
            self.anatomy.has_active_DH,
            self.anatomy.has_active_ER,
        ):
            case (True,  True,  True  ): return setup_substrate(PKSExtenderUnit.PKS_D)
            case (True,  True,  False ): return setup_substrate(PKSExtenderUnit.PKS_C)
            case (True,  _,     _     ): return setup_substrate(PKSExtenderUnit.PKS_B)  # presence of ER doesn't matter if DH is not present
            case (False, _,     _     ): return setup_substrate(PKSExtenderUnit.PKS_A)  # presence of DH/ER doesn't matter if no KR
            case _:                      return setup_substrate(PKSExtenderUnit.UNCLASSIFIED)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the PKSModule object to a dictionary representation.

        :return: Dictionary representation of the PKSModule
        """
        return {
            "type": self.type.value,
            "module_index_in_gene": self.module_index_in_gene,
            "start": self.start,
            "end": self.end,
            "gene_id": self.gene_id,
            "gene_strand": self.gene_strand.value,
            "present_domains": self.present_domains,
            "role": self.role.value,
            "anatomy": self.anatomy.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PKSModule":
        """
        Create a PKSModule object from a dictionary representation.

        :param data: Dictionary representation of the PKSModule
        :return: PKSModule object
        """
        anatomy_data = data.get("anatomy", {})

        return cls(
            module_index_in_gene=data["module_index_in_gene"],
            start=data["start"],
            end=data["end"],
            gene_id=data["gene_id"],
            gene_strand=Strand(data["gene_strand"]),
            present_domains=data["present_domains"],
            role=ModuleRole(data["role"]),
            anatomy=PKSAnatomy.from_dict(anatomy_data),
        )


@dataclass
class LinearReadout:
    """
    A linear readout consisting of a sequence of modules.
    
    :param id: unique identifier for the linear readout
    :param start: starting position of the linear readout
    :param end: ending position of the linear readout
    :param qualifiers: additional metadata or qualifiers associated with the linear readout
    :param modules: list of modules in the linear readout
    """
    
    id: str
    start: int
    end: int
    qualifiers: dict[str, Any] = field(default_factory=dict)

    modules: list[Module] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """
        String representation of the LinearReadout.
        
        :return: string representation of the LinearReadout
        """
        return f"LinearReadout(id={self.id}, start={self.start}, end={self.end}, modules={len(self.modules)})"
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the LinearReadout object to a dictionary representation.

        :return: Dictionary representation of the LinearReadout
        """
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "qualifiers": self.qualifiers,
            "modules": [module.to_dict() for module in self.modules],
            "modifiers": self.modifiers,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinearReadout":
        """
        Create a LinearReadout object from a dictionary representation.

        :param data: Dictionary representation of the LinearReadout
        :return: LinearReadout object
        """
        modules_data = data.get("modules", [])
        modules: list[Module] = []

        for mod_data in modules_data:
            mod_type = mod_data.get("type", None)
            if mod_type == ModuleType.NRPS.value:
                modules.append(NRPSModule.from_dict(mod_data))
            elif mod_type == ModuleType.PKS.value:
                modules.append(PKSModule.from_dict(mod_data))
            else:
                raise ValueError(f"Unknown module type: {mod_type}")

        return cls(
            id=data["id"],
            start=data["start"],
            end=data["end"],
            qualifiers=data.get("qualifiers", {}),
            modules=modules,
            modifiers=data.get("modifiers", []),
        )
    
    @overload
    def biosynthetic_order(self, by_orf: Literal[False] = False) -> list[Module]: ...
    @overload
    def biosynthetic_order(self, by_orf: Literal[True] = True) -> list[tuple[str, list[Module]]]: ...

    def biosynthetic_order(self, by_orf: bool = False):
        """
        Return modules in biosynthetic order.

        :param by_orf: if True, group modules by their originating gene (ORF)
        :return: list of Module objects in biosynthetic order, or list of tuples (gene_id, list of Module) if by_orf is True
        """
        if not self.modules:
            return []
        
        # Group modules by gene
        by_gene: dict[str, list[Module]] = {}
        for m in self.modules:
            by_gene.setdefault(m.gene_id, []).append(m)

        # Infer strand per gene (sanity check)
        gene_strand: dict[str, Strand] = {}
        for gid, mods in by_gene.items():
            s = mods[0].gene_strand
            if any(m.gene_strand is not s for m in mods):
                raise ValueError(f"mixed gene_strand in gene_id={gid}")
            gene_strand[gid] = s

        # Infer global biosyntehtic direction
        strand_counts = Counter(gene_strand.values())
        global_reverse = strand_counts[Strand.REVERSE] > strand_counts[Strand.FORWARD]

        # Order genes along biosynthetic direction using genomic position
        gene_ids = sorted(
            by_gene.keys(),
            key=lambda gid: min(m.start for m in by_gene[gid]),
            reverse=global_reverse,
        )

        if by_orf:
            grouped: list[tuple[str, list[Module]]] = []
            for gid in gene_ids:
                mods = by_gene[gid]
                if gene_strand[gid] is Strand.FORWARD:
                    mods_sorted = sorted(mods, key=lambda m: m.start)
                else:
                    mods_sorted = sorted(mods, key=lambda m: m.start, reverse=True)
                grouped.append((gid, mods_sorted))

            return grouped

        # Flatten modules in biosynthetic order
        out: list[Module] = []
        for gid in gene_ids:
            mods = by_gene[gid]
            if gene_strand[gid] is Strand.FORWARD:
                mods_sorted = sorted(mods, key=lambda m: m.start)
            else:
                mods_sorted = sorted(mods, key=lambda m: m.start, reverse=True)
            out.extend(mods_sorted)

        return out
    

def _domain_index_by_obj(doms: list[Domain]) -> dict[int, int]:
    """
    Helper function to create a mapping from Domain object IDs to their indices in a list.

    :param doms: list of Domain objects
    :return: dictionary mapping Domain object IDs to their indices
    """
    return {id(d): i for i, d in enumerate(doms)}


def _domain_types(domains: list[Domain]) -> set[str]:
    """
    Helper function to extract the set of domain types from a list of Domain objects.
    
    :param domains: List of Domain objects
    :return: Set of domain type strings
    """
    return {d.type for d in domains if d.type is not None}


def _is_domain_type(domain: Domain, label: str | set[str]) -> bool:
    """
    Check if a domain matches a given type label or set of labels.

    :param domain: Domain object to check
    :param label: domain type label or set of labels to match against
    :return: True if the domain type matches the label(s), False otherwise
    """
    if not domain.type:
        return False

    if isinstance(label, set):
        return domain.type in label
    
    return domain.type == label


def _is_Cstarter(domain: Domain) -> bool:
    """
    Determine if a condensation domain is a C-starter domain based on its qualifiers.

    :param domain: Domain object to evaluate
    :return: True if the domain is a C-starter, False otherwise
    """
    if not domain.type or domain.type != "Condensation":
        return False
    
    txts = []
    if domain.id:
        txts.append(domain.id)

    for _, vals in domain.raw_qualifiers.items():
        # Join lists and scalars; qualifiers may be list[str]
        if isinstance(vals, (list, tuple)):
            txts.extend(map(str, vals))
        else:
            txts.append(str(vals))
    
    blob = " ".join(txts).lower()

    return ("starter" in blob) or ("cstarter" in blob) or ("condensation_starter" in blob)


def _upstream_loading_cassette(all_genes: list[Gene], gene_idx_in_genomic_order: int, max_bp: int = 20_000) -> bool:
    """
    Check for upstream loading cassette (CAL + ACP) in upstream genes within max_bp distance.

    :param all_genes: list of all Gene objects in the region/cluster
    :param gene_idx_in_genomic_order: index of the gene gene in all_genes
    :param max_bp: maximum base pair distance to search upstream
    :return: True if a loading cassette is found upstream within max_bp, False otherwise
    """
    cur_start = all_genes[gene_idx_in_genomic_order].start

    seen_cal = False
    seen_acp = False
    for j in range(gene_idx_in_genomic_order - 1, -1, -1):
        g = all_genes[j]
        if cur_start - g.end > max_bp:
            break  # exceeded max distance
        types = _domain_types(g.domains)
        d_ids = {d.id for d in g.domains if d.id}
        if ("CAL_domain" in types) or any("faal" in d_id.lower() for d_id in d_ids):
            seen_cal = True
        if ("PP-binding" in types) or ("ACP" in types) or any("acp" in d_id.lower() for d_id in d_ids):
            seen_acp = True
        if seen_cal and seen_acp:
            return True
    
    return False


def _upstream_has_nrps_A(all_genes: list[Gene], gene_idx_in_genomic_order: int) -> bool:
    """
    Check if there is an upstream gene with an NRPS A-domain.

    :param all_genes: list of all Gene objects in the region/cluster
    :param gene_idx_in_genomic_order: index of the gene gene in all_genes
    :return: True if there is an upstream NRPS A-domain, False otherwise
    """
    for j in range(gene_idx_in_genomic_order - 1, -1, -1):
        if any(_is_domain_type(d, NRPS_A) for d in all_genes[j].domains):
            return True
        
    return False


def _split_module_on_KS(domains: list[Domain]) -> list[list[Domain]]:
    """
    Split a list of domains into windows based on PKS KS domains.

    :param domains: List of Domain objects
    :return: List of lists of Domain objects, each representing a module window
    """
    windows: list[list[Domain]] = []
    cur: list[Domain] = []

    for d in domains:
        if d.type == "PKS_KS":
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


def _is_active_accessory_domain(domain: Domain) -> bool:
    """
    Determine if an accessory domain (KR, DH, ER) is active based on its qualifiers.
    
    :param domain: Domain object to evaluate
    :return: True if the domain is active, False if inactive
    """
    if not domain.type:
        return True  # can't tell, assume active
    
    if domain.type not in {"PKS_KR", "PKS_DH", "PKS_ER"}:
        return True  # not a reducible domain, consider active by default
    
    texts = []
    if domain.id:
        texts.append(domain.id)
    for _, vals in domain.raw_qualifiers.items():
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
    is_active = not any(flag in blob for flag in inactive_flags)

    return is_active


def _classify_pks_window(window: list[Domain]) -> tuple[set[str], bool, bool, bool, bool]:
    """
    Classify a PKS module window based on the presence and activity of domains.

    :param window: list of Domain objects in the module window
    :return: tuple containing:
        - module type (str)
        - set of present domain types (set[str])
        - has active KR (bool)
        - has active DH (bool)
        - has active ER (bool)
        - has AT (bool)
    """
    types_linear = [d.type for d in window if d.type in PKS_TYPES]
    present = set(types_linear)

    has_AT = "PKS_AT" in present
    has_active_KR = any("PKS_KR" in present and _is_active_accessory_domain(d) for d in window if d.type == "PKS_KR")
    has_active_DH = any("PKS_DH" in present and _is_active_accessory_domain(d) for d in window if d.type == "PKS_DH")
    has_active_ER = any("PKS_ER" in present and _is_active_accessory_domain(d) for d in window if d.type == "PKS_ER")

    return present, has_active_KR, has_active_DH, has_active_ER, has_AT


def _window_bounds(window: list[Domain]) -> tuple[int, int]:
    """
    Get the start and end positions of a domain window.
    
    :param window: list of Domain objects in the module window
    :return: tuple of (start, end) positions
    """
    return min(d.start for d in window), max(d.end for d in window)


def _is_AT_only_gene(gene: Gene) -> bool:
    """
    Helper function to determine if a gene is an acyltransferase-domain-only gene.
    
    :param g: Gene object
    :return: True if the gene is an AT-only gene, False otherwise
    """
    types = _domain_types(gene.domains) 
    return ("PKS_AT" in types) and all(t in {"PKS_AT"} for t in types)


def _find_genomic_upstream_AT_only_gene(all_genes: list[Gene], gene_idx_in_genomic_order: int) -> Gene | None:
    """
    Return the nearest upstream gene that is AT-only (relative to all_genes order).

    :param all_genes: list of Gene objects
    :param gene_idx_in_genomic_order: index of the current gene in all_genes
    :return: Gene object of the nearest upstream AT-only gene, or None if not found
    """
    for j in range(gene_idx_in_genomic_order - 1, -1, -1):
        if _is_AT_only_gene(all_genes[j]):
            return all_genes[j]
        
    return None


def _upstream_has_pks_KS(
    all_genes: list[Gene],
    gene_idx_in_genomic_order: int,
    doms: list[Domain],
    ks_domain: Domain,
) -> bool:
    """
    Check if there is an upstream gene with a PKS KS-domain.
    
    :param all_genes: list of all Gene objects in the region/cluster
    :param gene_idx_in_genomic_order: index of the gene gene in all_genes
    :param doms: list of Domain objects in the current gene, sorted for biosynthetic order
    :param ks_domain: the KS Domain object to check upstream of
    :return: True if there is an upstream KS-domain, False otherwise
    """
    # Genes upstream
    for j in range(gene_idx_in_genomic_order -1, -1, -1):
        if any(d.type == "PKS_KS" for d in all_genes[j].domains):
            return True
        
    # Same gene: any KS earlier in biosynthetic order than this KS?
    idx = _domain_index_by_obj(doms)
    ks_i = idx[id(ks_domain)]
    return any(d.type == "PKS_KS" and idx[id(d)] < ks_i for d in doms)


def _standalone_pks_AT_upstream(
    all_genes: list[Gene],
    gene_idx_in_genomic_order: int,
    doms: list[Domain],
    ks_domain: Domain,
    max_bp: int = 20_000
) -> bool:
    """
    Check for standalone PKS AT domain in upstream genes within max_bp distance.

    :param all_genes: list of all Gene objects in the region/cluster
    :param gene_idx_in_genomic_order: index of the gene gene in all_genes
    :param doms: list of Domain objects in the current gene, sorted for biosynthetic order
    :param ks_domain: the KS Domain object to check upstream of
    :param max_bp: maximum base pair distance to search upstream
    :return: True if a standalone PKS AT domain is found upstream within max_bp, False otherwise
    """
    # Same gene: any AT earlier than this KS in biosynthetic order?
    idx = _domain_index_by_obj(doms)
    ks_i = idx[id(ks_domain)]
    if any(d.type == "PKS_AT" and idx[id(d)] < ks_i for d in doms):
        return True

    # Other genes upstream, within distance (still genomic)
    ks_start = ks_domain.start
    cur_start = ks_start
    for j in range(gene_idx_in_genomic_order - 1, -1, -1):
        g = all_genes[j]
        if cur_start - g.end > max_bp:
            break
        if any(d.type == "PKS_AT" for d in g.domains):
            return True
        
    return False


def _is_last_global_KS(
    all_genes: list[Gene],
    gene_idx_in_genomic_order: int,
    doms: list[Domain],
    ks_domain: Domain,
) -> bool:
    """
    Check if the given KS domain is the last KS domain in the entire gene cluster/region.
    
    :param all_genes: list of all Gene objects in the region/cluster
    :param gene_idx_in_genomic_order: index of the gene gene in all_genes
    :param doms: list of Domain objects in the current gene, sorted for biosynthetic order
    :param ks_domain: the KS Domain object to check
    :return: True if this is the last KS domain, False otherwise
    """
    # Same gene: any KS later in biosynthetic order?
    idx = _domain_index_by_obj(doms)
    ks_i = idx[id(ks_domain)]
    if any(d.type == "PKS_KS" and idx[id(d)] > ks_i for d in doms):
        return False
        
    # Downstream genes (genomic list order)
    for j in range(gene_idx_in_genomic_order + 1, len(all_genes)):
        if any(d.type == "PKS_KS" for d in all_genes[j].domains):
            return False 
    
    return True


def _genomic_downstream_has_TE(
    all_genes: list[Gene],
    gene_idx_in_genomic_order: int,
    doms: list[Domain],
    win: list[Domain],
    max_bp: int = 20_000
) -> bool:
    """
    Check for downstream thioesterase (TE) domain in downstream genes within max_bp distance.

    :param all_genes: list of all Gene objects in the region/cluster
    :param gene_idx_in_genomic_order: index of the gene gene in all_genes
    :param doms: list of Domain objects in the current gene, sorted for biosynthetic order
    :param win: current module window (list of Domain objects)
    :param max_bp: maximum base pair distance to search downstream
    :return: True if a thioesterase domain is found downstream within max_bp, False otherwise
    """
    idx = _domain_index_by_obj(doms)
    last_i = idx[id(win[-1])]

    # Same gene: any TE later than the window end in biosynthetic order?
    if any(d.type in PKS_TE_ALIASES and idx[id(d)] > last_i for d in doms):
        return True
        
    # Other genes downstream, within distance (still genomic)
    from_bp = max(d.end for d in win)  # genomic coordinate for distance window
    cur_end = from_bp
    for j in range(gene_idx_in_genomic_order + 1, len(all_genes)):
        gene = all_genes[j]
        if gene.start - cur_end > max_bp:
            break  # exceeded max distance

        if any(d.type in PKS_TE_ALIASES for d in gene.domains):
            return True
        
    return False


def domains_biosynthetic(gene: Gene) -> list[Domain]:
    """
    Return domains in biosynthetic order within a gene.
    
    :param gene: Gene object
    :return: list of Domain objects in biosynthetic order
    .. note:: we assume Domain.start/end are genomic coordinates
    """
    doms = sorted(gene.domains, key=lambda d: d.start)
    if gene.strand is Strand.REVERSE:
        doms = list(reversed(doms))

    return doms


def collect_nrps_modules(gene: Gene, gene_idx_in_genomic_order: int, all_genes: list[Gene]) -> list[NRPSModule]:
    """
    Collect NRPS modules from a given gene.
    
    :param gene: Gene object to analyze
    :param gene_idx_in_genomic_order: index of the gene in the region's gene list
    :param all_genes: List of all genes in the region
    :return: List of NRPSModule objects"""
    doms: list[Domain] = domains_biosynthetic(gene)
    out: list[NRPSModule] = []

    # Indices of A domains in left-to-right order
    a_idx = [i for i, d in enumerate(doms) if _is_domain_type(d, NRPS_A)]
    if not a_idx:
        return out  # no A domains, no modules
    
    for mi, ai in enumerate(a_idx):
        # Extend window backward by one if there is an immediately previous C (same gene)
        start_i = ai
        if ai - 1 >= 0 and _is_domain_type(doms[ai - 1], NRPS_C):
            start_i = ai - 1
        
        # Extend forward until (but not including) the next A-domain
        end_i = a_idx[mi + 1] if mi + 1 < len(a_idx) else len(doms)

        window = doms[start_i:end_i]
        present = _domain_types(window)

        has_C = any(_is_domain_type(d, NRPS_C) for d in window)
        has_Cstarter = any(_is_Cstarter(d) for d in window)
        has_T = any(_is_domain_type(d, NRPS_T_ALIASES) for d in window)
        has_E = any(_is_domain_type(d, NRPS_E) for d in window)
        has_MT = any(_is_domain_type(d, NRPS_MT_ALIASES) for d in window)
        has_Ox = any(_is_domain_type(d, NRPS_OX_ALIASES) for d in window)
        has_R = any(_is_domain_type(d, NRPS_R_ALIASES) for d in window)
        has_TE = any(_is_domain_type(d, NRPS_TE) for d in window)

        # Fallback evidence of a separate loading cassette upstream
        loading_upstream = _upstream_loading_cassette(all_genes, gene_idx_in_genomic_order)
        upstream_has_A = _upstream_has_nrps_A(all_genes, gene_idx_in_genomic_order)

        # Role heuristic
        is_first_module_in_gene = mi == 0

        starter = (
            has_Cstarter
            or (is_first_module_in_gene and loading_upstream and not upstream_has_A)
            or ((not has_C) and not upstream_has_A)
        )
        terminal = has_TE or has_R

        def _get_module_role(starter: bool, terminal: bool) -> ModuleRole:
            match (starter, terminal):
                case (True,  True ): return ModuleRole.STARTER_TERMINAL
                case (True,  False): return ModuleRole.STARTER
                case (False, True ): return ModuleRole.TERMINAL
                case (False, False): return ModuleRole.ELONGATION
        role: ModuleRole = _get_module_role(starter, terminal)

        s = min(d.start for d in window)
        e = max(d.end for d in window)

        # Retrieve A domain substrate specificity prediction
        A = doms[ai]
        anns = A.annotations
        substrate_pred: NRPSSubstrate | None = None
        if anns:
            preds = anns.results

            # Highest confidence first
            preds_sorted = sorted(preds, key=lambda r: r.score or 0.0, reverse=True)
    
            # Get highest confidence prediction, if any
            top_pred = preds_sorted[0] if preds_sorted else None

            if top_pred:
                substrate_pred = NRPSSubstrate(
                    name=top_pred.label,
                    smiles=top_pred.metadata.get("smiles", None),
                    score=top_pred.score,
                )

        out.append(NRPSModule(
            module_index_in_gene=mi,
            start=s,
            end=e,
            gene_id=gene.id,
            gene_strand=gene.strand,
            present_domains=list(present),
            role=role,
            anatomy=NRPSAnatomy(
                has_C=has_C,
                has_T=has_T,
                has_E=has_E,
                has_MT=has_MT,
                has_Ox=has_Ox,
                has_R=has_R,
                has_TE=has_TE,
            ),
            predicted_substrate=substrate_pred,
        ))

    return out


def collect_pks_modules(gene: Gene, gene_idx_in_genomic_order: int, all_genes: list[Gene]) -> list[PKSModule]:
    """
    Collect PKS modules from a given gene.
    
    :param gene: Gene object to analyze
    :param gene_idx_in_genomic_order: index of the gene in the region's gene list
    :param all_genes: list of all genes in the region
    :return: list of PKSModule objects
    """
    out: list[PKSModule] = []

    if all(d.type != "PKS_KS" for d in gene.domains):
        return out  # no KS domains, no modules
    
    doms = domains_biosynthetic(gene)
    windows = _split_module_on_KS(doms)
    for mi, win in enumerate(windows):
        (
            present,
            has_active_KR,
            has_active_DH,
            has_active_ER,
            has_AT,
        ) = _classify_pks_window(win)

        s, e = _window_bounds(win)

        if has_AT:
            AT_src: ATLoadingMode = ATLoadingMode.CIS
        else:
            AT_src: ATLoadingMode = (
                ATLoadingMode.TRANS 
                if _find_genomic_upstream_AT_only_gene(all_genes, gene_idx_in_genomic_order) is not None
                else ATLoadingMode.UNKNOWN
            )

        # Assign provisional PKS role
        has_TE_in_window = any(d.type in PKS_TE_ALIASES for d in win)
        KS_domain = win[0]  # first domain in window is KS since we split on KS
        upstream_has_KS = _upstream_has_pks_KS(all_genes, gene_idx_in_genomic_order, doms, KS_domain)
        starter = _standalone_pks_AT_upstream(all_genes, gene_idx_in_genomic_order, doms, KS_domain) and not upstream_has_KS

        terminal_by_TE = False
        if _is_last_global_KS(all_genes, gene_idx_in_genomic_order, doms, KS_domain):
            terminal_by_TE = has_TE_in_window or _genomic_downstream_has_TE(all_genes, gene_idx_in_genomic_order, doms, win)

        def _get_module_role(starter: bool, terminal_by_TE: bool) -> ModuleRole:
            match (starter, terminal_by_TE):
                case (True,  True ): return ModuleRole.STARTER_TERMINAL
                case (True,  False): return ModuleRole.STARTER
                case (False, True ): return ModuleRole.TERMINAL
                case (False, False): return ModuleRole.ELONGATION
        role: ModuleRole = _get_module_role(starter, terminal_by_TE)

        s, e = _window_bounds(win)
        out.append(PKSModule(
            module_index_in_gene=mi,
            start=s,
            end=e,
            gene_id=gene.id,
            gene_strand=gene.strand,
            present_domains=list(present),
            role=role,
            anatomy=PKSAnatomy(
                AT_loading_mode=AT_src,
                has_active_KR=has_active_KR,
                has_active_DH=has_active_DH,
                has_active_ER=has_active_ER,
            ),
        ))

    return out


def linear_readout(region: Region) -> LinearReadout:
    """
    Construct a linear readout from the given genomic region.

    :param region: Region object representing the genomic region
    :return: LinearReadout object containing the collected modules
    """
    assert isinstance(region, Region), "region must be an instance of Region"

    collected: list[Module] = []
    modifiers: list[str] = []

    for gi, gene in enumerate(region.iter_genes()):

        # Collect NRPS modules
        nrps_modules = collect_nrps_modules(gene, gi, region.genes)
        collected.extend(nrps_modules)

        # Collect PKS modules
        pks_modules = collect_pks_modules(gene, gi, region.genes)
        collected.extend(pks_modules)

        # Check if there are any gene-level modifiers
        if gene.annotations:
            for result in gene.annotations.results:
                label = result.label
                modifiers.append(label)

    return LinearReadout(
        id=region.id,
        start=region.start,
        end=region.end,
        qualifiers=region.qualifiers,
        modules=collected,
        modifiers=modifiers,
    )
