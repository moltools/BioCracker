"""Script to create a custom HMM model from selected PFAM families or clans."""

import argparse 
import logging
import subprocess
from pathlib import Path

from biocracker.utils.logging import setup_logging, add_file_handler
from biocracker.utils.download import download_and_prepare


PFAM_A_HMM_URL = r"https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
CLAN_MAP_URL = r"https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.0/Pfam-A.clans.tsv.gz"

LOG_LVL = "INFO"
log = logging.getLogger(__name__)


def run_cmd(cmd: list[str]) -> None:
    """
    Runs a command using subprocess and raises an error if it fails.

    :param cmd: command to run as a list of strings
    """
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command '{' '.join(cmd)}' failed ({res.returncode}):\n{res.stderr}")


def ensure_hmmer_available() -> None:
    """
    Ensures that HMMER suite is available in the system PATH.

    :raises RuntimeError: if HMMER is not available
    """
    try:
        subprocess.run(["hmmpress", "-h"], capture_output=True, check=True)
        subprocess.run(["hmmfetch", "-h"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("HMMER suite is not available; please install it to use this script")


def normalize_pfam_hmm_path(p: Path) -> Path:
    """
    Normalizes the PFAM HMM path to ensure it points to the Pfam-A.hmm file.

    :param p: path to PFAM HMM file or directory containing it
    :return: path to Pfam-A.hmm file
    :raises FileNotFoundError: if the Pfam-A.hmm file does not exist
    """
    if p.is_dir():
        p = p / "Pfam-A.hmm"
    if not p.exists():
        raise FileNotFoundError(f"PFAM HMM file not found at {p}")
    return p


def ensure_hmmpressed(hmm_file: Path) -> None:
    """
    Ensures that the HMM file is pressed using hmmpress.

    :param hmm_file: path to HMM file
    """
    needed = [".h3f", ".h3i", ".h3m", ".h3p"]
    if all(Path(str(hmm_file) + s).exists() for s in needed):
        return

    run_cmd(["hmmpress", "-f", str(hmm_file)])


def load_clan_map(clan_map_path: Path) -> dict[str, set[str]]:
    """
    Loads the clan map from the given file.

    :param clan_map_path: path to clan map file
    :return: mapping of clan IDs to sets of PFAM model names
    """
    clan_map: dict[str, set[str]] = {}
    with open(clan_map_path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            _, clan_id, _, model_name = parts[:4]
            if clan_id:
                clan_map.setdefault(clan_id, set()).add(model_name)

    return clan_map


def resolve_model_names(inputs: list[str], clan_map: dict[str, set[str]]) -> set[str]:
    """
    Resolves PFAM model names from given inputs, which can be PFAM model names or clan IDs.

    :param inputs: list of PFAM model names or clan IDs
    :param clan_map: mapping of clan IDs to sets of PFAM model names
    :return: set of resolved PFAM model names
    """
    targets = [t.upper() for t in inputs]
    model_names: set[str] = set()

    for t in targets:
        if t.startswith("PF"):
            err_msg = "please provide PFAM model names; not PFAM IDs (e.g., use 'SH3_1' instead of 'PF00018')"
            log.error(err_msg)
            raise ValueError(err_msg)
        if t.startswith("CL"):
            model_names |= clan_map.get(t, set())
            if t not in clan_map:
                log.warning(f"clan ID {t} not found in clan map")
        else:
            # allow direct model names too (e.g., SH3_1)
            model_names.add(t)

    model_names.discard("")  # remove empty strings if any
    return model_names


def cli() -> argparse.Namespace:
    """
    Configures command line interface for HMM model creation script.
    
    :return: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="path to output HMM file")
    parser.add_argument("--model-names", nargs="+", type=str, required=True, help="PFAM model and/or clan IDs to include in the HMM model")
    parser.add_argument("--cache-dir", type=str, default=None, help="directory to cache downloaded PFAM database files")
    parser.add_argument("--name", type=str, default="custom_pfam_model", help="name for the created HMM model")
    return parser.parse_args()


def main() -> None:
    ensure_hmmer_available()

    args = cli()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(LOG_LVL)
    add_file_handler(str(out_dir / "create_hmm_model.log"), level=LOG_LVL)

    log.info("command line arguments:")
    for arg, val in vars(args).items():
        log.info(f"\t{arg}: {val}")
    
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pfam_hmm_path = Path(download_and_prepare(PFAM_A_HMM_URL, cache_dir))
    clan_map_path = Path(download_and_prepare(CLAN_MAP_URL, cache_dir))

    pfam_hmm_file = normalize_pfam_hmm_path(pfam_hmm_path)
    ensure_hmmpressed(pfam_hmm_file)

    clan_map = load_clan_map(clan_map_path)
    model_names = resolve_model_names(args.model_names, clan_map)

    if not model_names:
        raise ValueError("no PFAM model names resolved from the provided inputs")
    
    log.info(f"collected {len(model_names)} PFAM model names for HMM model creation")

    keys_path = out_dir / f"{args.name}.keys.txt"
    keys_path.write_text("\n".join(sorted(model_names)) + "\n")

    output_hmm_path = out_dir / f"{args.name}.hmm"
    run_cmd(["hmmfetch", "-f", "-o", str(output_hmm_path), str(pfam_hmm_file), str(keys_path)])

    log.info(f"created custom HMM model at {output_hmm_path}")


if __name__ == "__main__":
    main()
