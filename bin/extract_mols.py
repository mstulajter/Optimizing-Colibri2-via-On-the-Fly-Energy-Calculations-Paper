#!/usr/bin/env python3
"""
Extract unique fragment SMILES from a GraphML network.

Authors:
    Miko Stulajter

Version 2.0.0

Usage:
    extract_mols.py -i INPUT.graphml.bz2 [-o OUTPUT.csv]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from graphml_utils import GRAPHML_NS, default_molecules_csv, open_graphml

PROTON_SMILES = "[H+]"
WATER_SMILES = "O"
HYDROXONIUM_SMILES = "[OH3+]"

# If [H+] appears in the network, ensure aqueous reference species are listed.
AQUEOUS_EXTRAS = (WATER_SMILES, HYDROXONIUM_SMILES)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique molecule SMILES from a GraphML network.",
    )
    parser.add_argument(
        "-i",
        dest="input_file",
        type=Path,
        required=True,
        help="Input GraphML file (.graphml or .graphml.bz2)",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        help="Output CSV (default: *_molecules.csv next to input)",
    )
    return parser.parse_args()


def extract_molecules(input_file: Path) -> set[str]:
    molecules: set[str] = set()
    with open_graphml(input_file, "rb") as handle:
        data = handle.read()

    context = ET.iterparse(io.BytesIO(data), events=("end",))
    for _event, element in context:
        if element.tag != f"{GRAPHML_NS}node":
            continue
        data_elem = element.find(f"{GRAPHML_NS}data")
        if data_elem is not None and data_elem.text:
            for fragment in data_elem.text.split("."):
                fragment = fragment.strip()
                if fragment:
                    molecules.add(fragment)
        element.clear()
    return molecules


def write_molecules_csv(molecules: set[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles"])
        for molecule in sorted(molecules):
            writer.writerow([molecule])


def main() -> int:
    args = parse_arguments()

    if not args.input_file.is_file():
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        return 1

    molecules = extract_molecules(args.input_file)

    if PROTON_SMILES in molecules:
        molecules.update(AQUEOUS_EXTRAS)

    output_file = args.output_file or default_molecules_csv(args.input_file)
    write_molecules_csv(molecules, output_file)
    print(f"Wrote {len(molecules)} molecules to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
