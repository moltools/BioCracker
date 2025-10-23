"""Module contains methods for making substrate specificity predictions with parasect."""

import logging
import os
from pathlib import Path

import joblib
from parasect.api import run_paras

from biocracker.antismash import DomainRec
from biocracker.config import LOGGER_NAME, PARAS_CACHE_DIR_NAME, PARAS_MODEL_DOWNLOAD_URL
from biocracker.helpers import download_and_prepare, get_biocracker_cache_dir

_PARAS_MODEL_CACHE: dict[str, object] = {}


def _load_parasect_model(cache_dir: Path) -> object:
    """
    Load the Parasect model from disk (cached in memory for reuse).

    :param cache_dir: Path to the cache directory
    :return: loaded Parasect model
    """
    global _PARAS_MODEL_CACHE

    # If model already loaded, return it immediately
    if PARAS_MODEL_DOWNLOAD_URL in _PARAS_MODEL_CACHE:
        return _PARAS_MODEL_CACHE[PARAS_MODEL_DOWNLOAD_URL]

    # Otherwise, ensure the file is downloaded and load it
    model_path = download_and_prepare(PARAS_MODEL_DOWNLOAD_URL, cache_dir)
    model = joblib.load(model_path)
    _PARAS_MODEL_CACHE[PARAS_MODEL_DOWNLOAD_URL] = model
    return model


def predict_amp_domain_substrate(
    domain: DomainRec,
    cache_dir_override: Path | str | None = None,
    *,
    model: object | None = None,
    pred_threshold: float = 0.5,
) -> list[tuple[str, str, float]] | None:
    """
    Predict substrate specificity for a given AMP-binding domain using Parasect.

    :param domain: DomainRec object representing the AMP-binding domain
    :param cache_dir_override: Optional path to override the default cache directory
    :param model: Optional already loaded parasect model, skip download and loading if provided
    :param pred_threshold: prediction threshold for substrate specificity (default: 0.5)
    :return: list of tuples with all predicted substrates as (name, smiles, score) above threshold
    :raises TypeError: if domain is not an instance of DomainRec
    .. note:: returns None if the domain is not of type "AMP-binding"
    .. note:: returns empty list if no predictions are above the threshold, or an error occurs
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not isinstance(domain, DomainRec):
        raise TypeError("Domain must be an instance of DomainRec")

    if domain.kind != "AMP-binding":
        return None

    # Define cache directory
    cache_dir = (
        Path(cache_dir_override)
        if cache_dir_override is not None
        else get_biocracker_cache_dir() / PARAS_CACHE_DIR_NAME
    )

    # Load parasect model if not provided
    if model is None:
        os.makedirs(cache_dir, exist_ok=True)
        model: object = _load_parasect_model(str(cache_dir))

    tmp_dir = cache_dir / "temp_parasect"
    os.makedirs(tmp_dir, exist_ok=True)

    # Prep fasta
    header = f">{domain.name if domain.name else 'AMP_domain'}|{domain.start}_{domain.end}"
    seq = domain.aa_seq
    fasta = f"{header}\n{seq}\n"

    # Ensure sequence is not empty
    if not seq:
        return []

    # Make prediction with parasect
    try:
        results = run_paras(
            selected_input=fasta,
            selected_input_type="fasta",
            path_temp_dir=tmp_dir,
            model=model,
            use_structure_guided_alignment=False,
        )
        assert len(results) == 1, "Expected exactly one parasect result for singular AMP-binding domain"
        result = results[0]
        preds = list(zip(result.prediction_labels, result._prediction_smiles, result.predictions, strict=True))
        preds = [(name, smiles, round(score, 3)) for name, smiles, score in preds if score >= pred_threshold]
        preds.sort(key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.error(f"{e}\nError during parasect prediction for domain {domain.name}, returning no predictions")
        preds = []

    return preds
