"""Gene data model."""

from dataclasses import dataclass, field, asdict
from enum import Enum

from biocracker.model.domain import Domain
from biocracker.model.annotations import AnnotationSet


class Strand(Enum):
    """
    Represents the strand orientation of a gene.
    
    :cvar FORWARD: the forward strand
    :cvar REVERSE: the reverse strand
    """
    FORWARD = "+"
    REVERSE = "-"


@dataclass
class Gene:
    """
    Represents a gene within a biological sequence.
    
    :param id: the unique identifier of the gene
    :param start: the starting position of the gene within the sequence
    :param end: the ending position of the gene within the sequence
    :param strand: the strand orientation of the gene
    :param sequence: the nucleotide sequence of the gene
    :param domains: the list of domains associated with the gene
    :param annotations: the set of annotations associated with the gene
    """

    id: str
    start: int
    end: int
    strand: Strand
    sequence: str

    domains: list[Domain] = field(default_factory=list)
    annotations: AnnotationSet = field(default_factory=AnnotationSet)

    def iter_domains(self) -> list[Domain]:
        """
        Returns the list of domains sorted by their starting position.
        
        :return: a list of domains sorted by start position
        """
        return sorted(self.domains, key=lambda d: d.start)
    
    def to_dict(self) -> dict:
        """
        Converts the gene to a dictionary representation.
        
        :return: a dictionary representation of the gene
        """
        data = asdict(self)
        data["strand"] = self.strand.value
        data["domains"] = [d.to_dict() for d in self.domains]
        data["annotations"] = self.annotations.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Gene":
        """
        Creates a Gene instance from a dictionary representation.
        
        :param data: a dictionary representation of the gene
        :return: a Gene instance
        """
        domains = [Domain.from_dict(d) for d in data.get("domains", [])]
        annotations = AnnotationSet.from_dict(data.get("annotations", {}))
        return cls(
            id=data["id"],
            start=data["start"],
            end=data["end"],
            strand=Strand(data["strand"]),
            sequence=data["sequence"],
            domains=domains,
            annotations=annotations,
        )