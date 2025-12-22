"""Module for fast PARAS inference of substrate specificity A domains."""

from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
from pyhmmer import easel, plan7, hmmer

import biocracker.data


HMM_DB_PATH = str(files(biocracker.data).joinpath("AMP-binding_converted.hmm"))
with plan7.HMMFile(HMM_DB_PATH) as hmm_file:
    HMM_DB = list(hmm_file)


VALID = set("ACDEFGHIKLMNPQRSTVWY-")
FEATURE_NAMES = [
    "WOLS870101",
    "WOLS870102",
    "WOLS870103",
    "FAUJ880109",
    "GRAR740102",
    "RADA880108",
    "ZIMJ680103",
    "TSAJ990101",
    "CHOP780201",
    "CHOP780202",
    "CHOP780203",
    "ZIMJ680104",
    "NEU1",
    "NEU2",
    "NEU3",
]
FEATURES = {
    "-": [0.00, 0.00, 0.00, 1, 8.3, 0.21, 13.59, 145.2, 1.00, 1.03, 0.99, 6.03, 0.06, 0.00, 0.10], 
    "A": [0.07, -1.73, 0.09, 0, 8.1, -0.06, 0.00, 90.0, 1.42, 0.83, 0.66, 6.00, 0.06, -0.25, 0.25], 
    "C": [0.71, -0.97, 4.13, 0, 5.5, 1.36, 1.48, 103.3, 0.70, 1.19, 1.19, 5.05, -0.56, -0.40, -0.14], 
    "D": [3.64, 1.13, 2.36, 1, 13.0, -0.80, 49.70, 117.3, 1.01, 0.54, 1.46, 2.77, 0.97, -0.08, 0.08], 
    "E": [3.08, 0.39, -0.07, 1, 12.3, -0.77, 49.90, 142.2, 1.51, 0.37, 0.74, 3.22, 0.85, -0.10, -0.05], 
    "F": [-4.92, 1.30, 0.45, 0, 5.2, 1.27, 0.35, 191.9, 1.13, 1.38, 0.60, 5.48, -0.99, 0.18, 0.15], 
    "G": [2.23, -5.36, 0.30, 0, 9.0, -0.41, 0.00, 64.9, 0.57, 0.75, 1.56, 5.97, 0.32, -0.32, 0.28], 
    "H": [2.41, 1.74, 1.11, 1, 10.4, 0.49, 51.60, 160.0, 1.00, 0.87, 0.95, 7.59, 0.15, -0.03, -0.10], 
    "I": [-4.44, -1.68, -1.03, 0, 5.2, 1.31, 0.13, 163.9, 1.08, 1.60, 0.47, 6.02, -1.00, -0.03, 0.10], 
    "K": [2.84, 1.41, -3.14, 2, 11.3, -1.18, 49.50, 167.3, 1.16, 0.74, 1.01, 9.74, 1.00, 0.32, 0.11], 
    "L": [-4.19, -1.03, -0.98, 0, 4.9, 1.21, 0.13, 164.0, 1.21, 1.30, 0.59, 5.98, -0.83, 0.05, 0.01], 
    "M": [-2.49, -0.27, -0.41, 0, 5.7, 1.27, 1.43, 167.0, 1.45, 1.05, 0.60, 5.74, -0.68, -0.01, 0.04], 
    "N": [3.22, 1.45, 0.84, 2, 11.6, -0.48, 3.38, 124.7, 0.67, 0.89, 1.56, 5.41, 0.70, -0.06, 0.17], 
    "P": [-1.22, 0.88, 2.23, 0, 8.0, 1.1, 1.58, 122.9, 0.57, 0.55, 1.52, 6.30, 0.45, 0.23, 0.41], 
    "Q": [2.18, 0.53, -1.14, 2, 10.5, -0.73, 3.53, 149.4, 1.11, 1.10, 0.98, 5.65, 0.71, -0.02, 0.12], 
    "R": [2.88, 2.52, -3.44, 4, 10.5, -0.84, 52.00, 194.0, 0.98, 0.93, 0.95, 10.76, 0.80, 0.19, -0.41], 
    "S": [1.96, -1.63, 0.57, 1, 9.2, -0.50, 1.67, 95.4, 0.77, 0.75, 1.43, 5.68, 0.48, -0.15, 0.23], 
    "T": [0.92, -2.09, -1.40, 1, 8.6, -0.27, 1.66, 121.5, 0.83, 1.19, 0.96, 5.66, 0.38, -0.10, 0.29], 
    "V": [-2.69, -2.53, -1.29, 0, 5.9, 1.09, 0.13, 139.0, 1.06, 1.70, 0.50, 5.96, -0.75, -0.19, 0.03], 
    "W": [-4.75, 3.65, 0.85, 1, 5.4, 0.88, 2.10, 228.2, 1.08, 1.37, 0.96, 5.89, -0.57, 0.31, 0.34], 
    "Y": [1.39, 2.32, 0.01, 1, 6.2, 0.33, 1.61, 197.0, 0.69, 1.47, 1.14, 5.66, -0.35, 0.40, -0.02], 
}
POSITIONS_ACTIVE_SITE = [
    13,
    16,
    17,
    41,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    55,
    93,
    94,
    125,
    126,
    127,
    128,
    129,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
]


@dataclass
class ADomain:
    """
    Dataclass representing an A domain.
    
    :param protein: name of the protein containing the A domain
    :param start: start position of the A domain
    :param end: end position of the A domain
    :param domain_nr: domain number of A domain in NRPS (optional)
    :param sequence: amino acid sequence of the A domain (optional)
    :param extended_signature: extended signature of the A domain (optional)
    """

    protein: str
    start: int
    end: int
    domain_nr: int | None = None
    sequence: str | None = None
    extended_signature: str | None = None


def _b2s(x: Any) -> str:
    """
    Convert input to string.
    
    :param x: input object
    :return: string representation
    """
    if isinstance(x, (bytes, bytearray)):
        return x.decode()

    if hasattr(x, "sequence"):
        s = x.sequence
        return s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
    
    return str(x)


def extract_domain_hits(
    seq_id: str,
    sequence: str,
    evalue_cutoff: float = 1e-5,
) -> list[dict[str, Any]]:
    """
    Extract domain hits from a given protein sequence using HMMER.

    :param seq_id: identifier for the protein sequence
    :param sequence: amino acid sequence of the protein
    :param evalue_cutoff: e-value cutoff for HMMER hits
    :return: list of dictionaries representing domain hits
    """
    alphabet = easel.Alphabet.amino()
    text_seq = easel.TextSequence(name=seq_id.encode(), sequence=sequence)
    seq = text_seq.digitize(alphabet)

    hits_iter = hmmer.hmmscan([seq], HMM_DB, cpus=1, E=evalue_cutoff)

    query_hits = next(hits_iter)  # expect only one sequence

    out = []
    for hit in query_hits:
        model_name = _b2s(hit.name)

        for dom in hit.domains:
            q_from = int(dom.env_from)
            q_to = int(dom.env_to)

            aln = dom.alignment
            hmm_aln = _b2s(aln.hmm_sequence)
            query_aln = _b2s(aln.target_sequence)

            out.append(
                dict(
                    seq_id=seq_id,
                    model=model_name,
                    q_from=q_from,
                    q_to=q_to,
                    evalue=float(dom.i_evalue),
                    score=float(dom.score),
                    hmm_aln=hmm_aln,
                    query_aln=query_aln,
                    domain_obj=dom,
                )
            )

    out.sort(key=lambda d: (d["q_from"], d["q_to"], d["model"]))

    return out


def pair_domains(
    domain_hits: list[dict[str, Any]],
    max_gap: int = 200,
) -> list[tuple[ADomain, str, str]]:
    """
    Pair AMP-binding and AMP-binding_C domain hits.
    
    :param domain_hits: list of domain hit dictionaries
    :param max_gap: maximum allowed gap between paired domains
    :return: list of tuples containing ADomain objects and their alignments
    """
    hits = sorted(domain_hits, key=lambda d: d["q_from"])

    a_domains: list[ADomain] = []
    for h1 in hits:
        if h1["model"] != "AMP-binding":
            continue
        
        n_from, n_to = h1["q_from"], h1["q_to"]

        matched = None
        for h2 in hits:
            if h2["model"] != "AMP-binding_C":
                continue

            c_from = h2["q_from"]

            if c_from > n_to and (c_from - n_to) <= max_gap:
                matched = h2
                break

        start0 = n_from - 1
        end0 = matched["q_to"] if matched is not None else n_to
        a_domains.append((ADomain(
            protein=h1["seq_id"],
            start=start0,
            end=end0),
            h1["hmm_aln"],
            h1["query_aln"]
        ))

    a_domains.sort(key=lambda t: t[0].start)
    for i, (d, _, _) in enumerate(a_domains, start=1):
        d.domain_nr = i
    
    return a_domains


def extract_signature_from_alignment(hmm_aln: str, query_aln: str) -> str | None:
    """
    Extract the extended signature from the given HMM and query alignments.
    
    :param hmm_aln: HMM alignment string
    :param query_aln: query alignment string
    :return: extended signature string or None if invalid
    """
    wanted = set(POSITIONS_ACTIVE_SITE)
    picked: dict[int, str] = {}

    hmm_pos = 0  # 1-based counter, increment when HMM char is not a gap

    for h, q in zip(hmm_aln, query_aln):
        if h != "-":
            hmm_pos += 1
            if hmm_pos in wanted and hmm_pos not in picked:
                picked[hmm_pos] = q

    # Quick fix
    missing = wanted - set(picked.keys())
    for m in missing:
        picked[m] = "-"

    out = []
    for p in POSITIONS_ACTIVE_SITE:
        if p not in picked:
            return None
        out.append(picked[p])
    
    sig = "".join(out).upper()
    if not sig or not all(c in VALID for c in sig):
        return None
    
    return sig


def fill_domain_sequences(
    domains: list[ADomain],
    protein_seq: str,
    min_len: int = 100,
) -> list[ADomain]:
    """
    Fill in the sequences for the given domains from the protein sequence.

    :param domains: list of ADomain objects
    :param protein_seq: amino acid sequence of the protein
    :param min_len: minimum length of domain sequence to keep
    :return: list of ADomain objects with sequences filled in
    """
    out = []

    for d in domains:
        seq = protein_seq[d.start:d.end]
        if len(seq) >= min_len:
            d.sequence = seq
            out.append(d)

    return out


def find_a_domains(
    seq_id: str,
    protein_seq: str,
    evalue_cutoff: float = 1e-5,
) -> list[ADomain]:
    """
    Find A domains in a given protein sequence using HMMER.
    
    :param seq_id: identifier for the protein sequence
    :param protein_seq: amino acid sequence of the protein
    :param evalue_cutoff: e-value cutoff for HMMER hits
    :return: list of ADomain objects representing found A domains
    """
    hits = extract_domain_hits(seq_id, protein_seq, evalue_cutoff)

    hits = [h for h in hits if h["model"] in {"AMP-binding", "AMP-binding_C"}]

    paired = pair_domains(hits, max_gap=200)

    domains_only: list[ADomain] = []
    for d, hmm_aln, query_aln in paired:
        d.extended_signature = extract_signature_from_alignment(hmm_aln, query_aln)
        domains_only.append(d)

    domains_only = fill_domain_sequences(domains_only, protein_seq, min_len=100)

    domains_only = [d for d in domains_only if d.extended_signature is not None]

    domains_only.sort(key=lambda d: (d.protein, d.start))

    return domains_only


def featurize_signature(sig: str) -> np.ndarray:
    """
    Featurize the given extended signature into a numerical feature array.

    :param sig: extended signature string
    :return: numpy array of features
    """
    assert len(sig) == len(POSITIONS_ACTIVE_SITE), "signature length mismatch"

    features: np.ndarray = np.zeros((len(POSITIONS_ACTIVE_SITE), len(FEATURE_NAMES)), dtype=np.float32)
    for i, aa in enumerate(sig):
        aa_feats = FEATURES.get(aa)
        if aa_feats is None:
            raise ValueError(f"invalid amino acid '{aa}' in signature")
        features[i, :] = np.array(aa_feats, dtype=np.float32)
    
    return features.flatten()  # shape (n_positions * n_features,)
