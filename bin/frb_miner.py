#!/usr/bin/env python
import numpy as np
import yaml
import argparse

def main(args):

    return 5

def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(
        prog = "frb_miner.py",
        description = "Python FRB searcher. It performs RFI excision, candidate search and classification. The options should be parsed via a config YAML file.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,

    )
    parser.add_argument(
        '-f',
        '--file',
        type = string,
        help = "SIGPROC .fil file (required)",
        required = True,
    )
    parser.add_argument(
        '-c',
        '--config',
        type = string,
        help = "YAML config file for the search",
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = _get_parser()

    main(args)
