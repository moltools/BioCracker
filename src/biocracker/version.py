"""Version information for BioCracker."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("biocracker")
except PackageNotFoundError:
    # When running in a source checkout and haven’t installed it yet,
    # importlib.metadata might not find “biocracker” in site‐packages.
    # Fallback to a hard-coded default or raise an error:
    __version__ = "0.0.0"
