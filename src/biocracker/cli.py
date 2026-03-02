#!/usr/bin/env python3

import argparse


def cli() -> argparse.Namespace:
    """
    Command-line interface for BioCracker.

    :return: parsed command-line arguments
    """
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main() -> None:
    """
    Entry point for the BioCracker command-line interface.
    """
    args = cli()
    print(args)


if __name__ == "__main__":
    main()
