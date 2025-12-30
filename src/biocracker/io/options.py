"""Module for defining options related to loading genomic regions from various sources."""

from dataclasses import dataclass, field
from typing import Literal, Union, TypeAlias


AntismashReadoutLevel: TypeAlias = Literal["region", "cand_cluster"]


@dataclass(frozen=True)
class AntiSmashOptions:
    """
    Options for loading biosynthetic regions from antiSMASH GenBank files.
    
    :param source: source type
    :param readout_level: parsing level for biosynthetic regions
    :param wanted_qualifiers: list of feature qualifiers to extract from regions
    :param gene_identifiers: list of gene feature types to look for
    :param domain_identifiers: list of domain feature types to look for
    """

    source: Literal["antismash_gbk"] = "antismash_gbk"
    readout_level: AntismashReadoutLevel = "region"
    wanted_qualifiers: list[str] | None = field(default_factory=lambda: ["product"])
    gene_identifiers: list[str] | None = field(default_factory=lambda: ["CDS"])
    domain_identifiers: list[str] | None = field(default_factory=lambda: ["aSDomain"])

    def __post_init__(self) -> None:
        """
        Validate options after initialization.
        """
        valid_levels = {"region", "cand_cluster"}
        if self.readout_level not in valid_levels:
            raise ValueError(f"expected readout_level to be one of {valid_levels}, got {self.readout_level}")


RegionLoadOptions = Union[AntiSmashOptions]
