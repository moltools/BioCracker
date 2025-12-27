"""Domain data model."""

from dataclasses import dataclass, field, asdict
from typing import Any

from biocracker.model.annotations import AnnotationSet


@dataclass
class Domain:
    """
    Represents a biological domain within a gene.
    
    :param id: the unique identifier of the domain
    :param type: the type or name of the domain
    :param start: the starting position of the domain within the gene
    :param end: the ending position of the domain within the gene
    :param sequence: the amino acid sequence of the domain
    :param raw_qualifiers: raw qualifiers or metadata associated with the domain
    :param annotations: the set of annotations associated with the domain
    """

    id: str
    type: str
    start: int
    end: int
    sequence: str
    raw_qualifiers: dict[str, Any] = field(default_factory=dict)

    annotations: AnnotationSet = field(default_factory=AnnotationSet)

    def to_dict(self) -> dict:
        """
        Converts the domain to a dictionary representation.
        
        :return: a dictionary representation of the domain
        """
        data = asdict(self)
        data["annotations"] = self.annotations.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Domain":
        """
        Creates a Domain instance from a dictionary representation.
        
        :param data: a dictionary representation of the domain
        :return: a Domain instance
        """
        annotations = AnnotationSet.from_dict(data.get("annotations", {}))
        return cls(
            id=data["id"],
            type=data["type"],
            start=data["start"],
            end=data["end"],
            sequence=data["sequence"],
            raw_qualifiers=data.get("raw_qualifiers", {}),
            annotations=annotations,
        )
