"""Module contains functions for reading out RegionRec/CandidateClusterRec objects."""

from collections.abc import Generator
from pathlib import Path
from typing import Literal

from biocracker.antismash import CandidateClusterRec, RegionRec
from biocracker.paras import predict_amp_domain_substrate


def linear_readouts(
    rec: RegionRec | CandidateClusterRec,
    cache_dir_override: Path | str | None = None,
    *,
    level: Literal["rec", "gene"] = "rec",
    model: object | None = None,
    pred_threshold: float = 0.5,
) -> Generator[dict, None, None]:
    """
    Reads out a RegionRec or CandidateClusterRec object and returns a list substrates.

    :param rec: RegionRec or CandidateClusterRec object to read out
    :param level: level of readout, either "rec" for region/cluster level or "gene" for gene level
    :return: Generator of substrate specificities per level as dictionaries
    :raises AssertionError: if level is not "rec" or "gene" or rec is not a RegionRec or CandidateClusterRec
    """
    assert level in {"rec", "gene"}, 'Level must be either "rec" or "gene"'
    assert isinstance(rec, (RegionRec, CandidateClusterRec)), "rec must be a RegionRec or CandidateClusterRec object"

    readout_rec = []
    for gene in rec.genes:
        readout_gene = []
        for domain in gene.domains:
            if domain.kind == "AMP-binding":
                domain_preds = predict_amp_domain_substrate(
                    domain=domain, cache_dir_override=cache_dir_override, model=model, pred_threshold=pred_threshold
                )
                readout_gene.append(domain_preds)

        if level == "gene" and readout_gene:
            yield {"rec": gene, "readout": readout_gene}
        else:
            readout_rec.extend(readout_gene)

    if level == "rec" and readout_rec:
        yield {"rec": rec, "readout": readout_rec}
