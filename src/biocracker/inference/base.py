"""Base class for domain inference models."""

from abc import ABC, abstractmethod
from typing import Any

from biocracker.model.domain import Domain
from biocracker.model.gene import Gene
from biocracker.model.inference import InferenceResult, TargetType


class GeneInferenceModel(ABC):
    """
    Base class for gene inference models.
    
    :param name: the name of the inference model
    :param target: the target type of the inference model
    """

    name: str
    target: TargetType = TargetType.GENE

    @abstractmethod
    def predict(self, gene: Gene) -> list[InferenceResult]:
        """
        Make predictions for a given gene.

        :param gene: The gene to make predictions for.
        :return: a list of InferenceResult objects containing the predictions.
        """
        ...

    def result(
        self,
        *,
        label: str,
        score: float | None = None,
        metadata: dict[str, Any] | None = None
    ) -> InferenceResult:
        """
        Helper method to create an InferenceResult object.

        :param label: the predicted label or class
        :param score: the confidence score of the prediction (optional)
        :param metadata: additional metadata related to the inference (optional)
        :return: an InferenceResult object
        """
        return InferenceResult(
            model=self.name,
            target=self.target,
            label=label,
            score=score,
            metadata=metadata
        )


class DomainInferenceModel(ABC):
    """
    Base class for domain inference models.
    
    :param name: the name of the inference model
    :param target: the target type of the inference model
    """

    name: str
    target: TargetType = TargetType.DOMAIN

    @abstractmethod
    def predict(self, domain: Domain) -> list[InferenceResult]:
        """
        Make predictions for a given domain.

        :param domain: The domain to make predictions for.
        :return: A list of InferenceResult objects containing the predictions.
        """
        ...

    def result(
        self,
        *,
        label: str,
        score: float | None = None,
        metadata: dict[str, Any] | None = None
    ) -> InferenceResult:
        """
        Helper method to create an InferenceResult object.

        :param label: the predicted label or class
        :param score: the confidence score of the prediction (optional)
        :param metadata: additional metadata related to the inference (optional)
        :return: an InferenceResult object
        """
        return InferenceResult(
            model=self.name,
            target=self.target,
            label=label,
            score=score,
            metadata=metadata
        )
