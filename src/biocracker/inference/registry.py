"""Registry for inference models."""

from biocracker.inference.base import GeneInferenceModel, DomainInferenceModel

GENE_MODELS: list[GeneInferenceModel] = []
DOMAIN_MODELS: list[DomainInferenceModel] = []


def register_gene_model(model: GeneInferenceModel) -> None:
    """
    Register a gene inference model.
    
    :param model: the gene inference model to register
    :raises TypeError: if the model is not an instance of GeneInferenceModel
    """
    # Make sure model is an instance of GeneInferenceModel
    if not isinstance(model, GeneInferenceModel):
        raise TypeError("model must be an instance of GeneInferenceModel")

    # Check that the model is not already registered; if it is, do not register it again
    registered = [m for m in GENE_MODELS if m.name == model.name]
    if registered:
        return

    GENE_MODELS.append(model)


def register_domain_model(model: DomainInferenceModel) -> None:
    """
    Register a domain inference model.

    :param model: the domain inference model to register
    """
    # Make sure model is an instance of DomainInferenceModel
    if not isinstance(model, DomainInferenceModel):
        raise TypeError("model must be an instance of DomainInferenceModel")

    # Check that the model is not already registered; if it is, do not register it again
    registered = [m for m in DOMAIN_MODELS if m.name == model.name]
    if registered:
        return

    DOMAIN_MODELS.append(model)


def get_gene_models() -> list[GeneInferenceModel]:
    """
    Get the list of registered gene inference models.
    
    :return: list of registered gene inference models
    """
    return GENE_MODELS


def get_domain_models() -> list[DomainInferenceModel]:
    """
    Get the list of registered domain inference models.

    :return: list of registered domain inference models
    """
    return DOMAIN_MODELS
