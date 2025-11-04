"""Module contains methods for making substrate specificity predictions with paras."""

import logging
import os
from pathlib import Path

import joblib

try:
    from parasect.api import run_paras  # module is called parasect, but we are using the paras model

    _HAS_PARAS = True
except ImportError:
    run_paras = None
    _HAS_PARAS = False

from biocracker.antismash import DomainRec
from biocracker.config import LOGGER_NAME, PARAS_CACHE_DIR_NAME, PARAS_MODEL_DOWNLOAD_URL
from biocracker.helpers import download_and_prepare, get_biocracker_cache_dir

_PARAS_MODEL_CACHE: dict[str, object] = {}


def has_paras() -> bool:
    """
    Check if parasect is installed.

    :return: True if parasect is installed, False otherwise
    """
    return _HAS_PARAS


def _load_paras_model(cache_dir: Path) -> object:
    """
    Load the paras model from disk (cached in memory for reuse).

    :param cache_dir: Path to the cache directory
    :return: loaded paras model
    """
    if not has_paras():
        raise ImportError("paras is not installed, cannot load paras model")

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
) -> list[dict] | None:
    """
    Predict substrate specificity for a given AMP-binding domain using paras.

    :param domain: DomainRec object representing the AMP-binding domain
    :param cache_dir_override: Optional path to override the default cache directory
    :param model: Optional already loaded paras model, skip download and loading if provided
    :param pred_threshold: prediction threshold for substrate specificity (default: 0.5)
    :return: list of dictionaries with all predicted substrates above the threshold, each with keys:
        'substrate_name' (str): substrate name,
        'substrate_smiles' (str): substrate SMILES,
        'score' (float): prediction score
    :raises TypeError: if domain is not an instance of DomainRec
    .. note:: returns None if the domain is not of type "AMP-binding"
    .. note:: returns empty list if no predictions are above the threshold, or an error occurs
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not isinstance(domain, DomainRec):
        raise TypeError("Domain must be an instance of DomainRec")

    if domain.kind != "AMP-binding":
        return None

    # If parasect is missing, log and return None
    if not has_paras():
        logger.warning("parasect not installed — skipping substrate prediction.")
        return None

    # Define cache directory
    cache_dir = (
        Path(cache_dir_override)
        if cache_dir_override is not None
        else get_biocracker_cache_dir() / PARAS_CACHE_DIR_NAME
    )

    # Load paras model if not provided
    if model is None:
        os.makedirs(cache_dir, exist_ok=True)
        model: object = _load_paras_model(str(cache_dir))

    tmp_dir = cache_dir / "temp_paras"
    os.makedirs(tmp_dir, exist_ok=True)

    # Prep fasta
    header = f">{domain.name if domain.name else 'AMP_domain'}|{domain.start}_{domain.end}"
    seq = domain.aa_seq
    fasta = f"{header}\n{seq}\n"

    # Ensure sequence is not empty
    if not seq:
        return []

    # Make prediction with paras
    try:
        results = run_paras(
            selected_input=fasta,
            selected_input_type="fasta",
            path_temp_dir=tmp_dir,
            model=model,
            use_structure_guided_alignment=False,
        )
        assert len(results) == 1, "Expected exactly one paras result for singular AMP-binding domain"
        result = results[0]
        preds = list(zip(result.prediction_labels, result._prediction_smiles, result.predictions, strict=True))
        preds = [(name, smiles, round(score, 3)) for name, smiles, score in preds if score >= pred_threshold]
        # Highest score first
        preds.sort(key=lambda x: x[2], reverse=True)
    except Exception as e:
        logger.error(f"{e}\nError during paras prediction for domain {domain.name}, returning no predictions")
        preds = []

    # Format predictions
    preds = [{"substrate_name": name, "substrate_smiles": smiles, "score": score} for name, smiles, score in preds]

    return preds
