"""Module for loading biosynthetic regions from GenBank files."""

from pathlib import Path

from biocracker.io.options import RegionLoadOptions, AntiSmashOptions
from biocracker.io.gbk_antismash import parse_antismash_gbk
from biocracker.model.region import Region


def load_regions(path: Path | str, options: RegionLoadOptions) -> list[Region]:
    """
    Load biosynthetic regions from a GenBank file.
    
    :param path: path to the GenBank file
    :param source: source of the biosynthetic regions
    :return: list of biosynthetic regions
    :raises NotImplementedError: if the source is not implementedr
    """

    if isinstance(path, str):
        path = Path(path)

    match options:
        case AntiSmashOptions():
            return parse_antismash_gbk(path, options)
        case _:
            raise NotImplementedError(f"loading regions from source {options.source} is not implemented")
