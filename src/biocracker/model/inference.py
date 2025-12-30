"""Inference result data model."""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any


class TargetType(Enum):
    """
    Enumeration of possible target types for model inference.
    
    :cvar DOMAIN: target type representing a domain-level prediction
    :cvar GENE: target type representing a gene-level prediction
    """
    
    DOMAIN = "domain"
    GENE = "gene"


@dataclass
class InferenceResult:
    """
    Represents the result of a model inference.
    
    :param model: the name of the model used for inference
    :param target: the target type of the inference (e.g., DOMAIN, GENE)
    :param label: the predicted label or class
    :param score: the confidence score of the prediction (optional)
    :param metadata: additional metadata related to the inference (optional)
    """
    model: str
    target: TargetType
    label: str
    score: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the InferenceResult instance to a dictionary.
        
        :return: a dictionary representation of the InferenceResult
        """
        result_dict = asdict(self)
        result_dict["target"] = self.target.value
        return result_dict
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceResult":
        """
        Create an InferenceResult instance from a dictionary.
        
        :param data: a dictionary containing the inference result data
        :return: an InferenceResult instance
        """
        data = data.copy()
        data["target"] = TargetType(data["target"])
        return cls(**data)
