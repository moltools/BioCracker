"""Annotation set data model."""

from dataclasses import dataclass, field, asdict

from biocracker.model.inference import InferenceResult


@dataclass
class AnnotationSet:
    """
    Represents a set of annotations derived from model inferences.
    
    :param results: a list of inference results
    """
    results: list[InferenceResult] = field(default_factory=list)

    def add(self, result: InferenceResult) -> None:
        """
        Adds an inference result to the annotation set.
        
        :param result: the inference result to add
        """
        self.results.append(result)

    def by_model(self, model_name: str) -> list[InferenceResult]:
        """
        Retrieves all inference results from a specific model.
        
        :param model_name: the name of the model
        :return: a list of inference results from the specified model
        """
        return [r for r in self.results if r.model == model_name]
    
    def to_dict(self) -> dict:
        """
        Converts the annotation set to a dictionary representation.
        
        :return: a dictionary representation of the annotation set
        """
        return {"results": [r.to_dict() for r in self.results]}
    
    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationSet":
        """
        Creates an AnnotationSet instance from a dictionary representation.
        
        :param data: a dictionary representation of the annotation set
        :return: an AnnotationSet instance
        """
        results = [InferenceResult(**r) for r in data.get("results", [])]
        return cls(results=results)
