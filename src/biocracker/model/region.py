"""Region data model."""

from dataclasses import dataclass, field, asdict
from typing import Any

from biocracker.model.gene import Gene


@dataclass
class Region:
    """
    Represents a biological region containing multiple genes.
    
    :param id: the unique identifier of the region
    :param file_name: the name of the file from which the region was parsed
    :param start: the starting position of the region within the sequence
    :param end: the ending position of the region within the sequence
    :param qualifiers: additional metadata or qualifiers associated with the region
    :param genes: the list of genes contained within the region
    """

    id: str
    file_name: str
    start: int
    end: int
    qualifiers: dict[str, Any] = field(default_factory=dict)
    genes: list[Gene] = field(default_factory=list)

    def iter_genes(self) -> list[Gene]:
        """
        Returns the list of genes sorted by their starting position.
        
        :return: a list of genes sorted by start position
        """
        return sorted(self.genes, key=lambda g: g.start)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Region instance to a dictionary.
        
        :return: a dictionary representation of the Region
        """
        region_dict = asdict(self)
        region_dict["genes"] = [gene.to_dict() for gene in self.genes]
        return region_dict
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Region":
        """
        Creates a Region instance from a dictionary.
        
        :param data: a dictionary containing region data
        :return: a Region instance
        """
        genes = [Gene.from_dict(gene_data) for gene_data in data.get("genes", [])]
        return cls(
            id=data["id"],
            file_name=data["file_name"],
            start=data["start"],
            end=data["end"],
            qualifiers=data.get("qualifiers", {}),
            genes=genes,
        )
