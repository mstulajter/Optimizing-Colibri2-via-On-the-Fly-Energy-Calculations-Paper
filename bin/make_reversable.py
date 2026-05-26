#!/usr/bin/env python3
"""
Ensure every directed edge in a GraphML network has a reverse counterpart.

Authors:
    Miko Stulajter

Version 2.0.0

Usage:
    make_reversable.py [-h] [-o OUTPUT] INPUT.graphml.bz2
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path

import bz2
import networkit as nk

from graphml_utils import (
    DATA_D1_RE,
    EDGE_OPEN_RE,
    default_reversable_output,
    escape_xml_for_graphml,
    open_graphml,
    unescape_xml,
)

GRAPH_CLOSE_RE = re.compile(r"\s*</graph>\s*$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add missing reverse edges to a GraphML network.",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input GraphML file (.graphml or .graphml.bz2)",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        help="Output path (default: *_reversable.graphml.bz2 next to input)",
    )
    return parser.parse_args()


def flip_transformation(transformation: str) -> str:
    """Swap reactant and product sides around ``>>`` or ``&gt;&gt;``."""
    for separator in (">>", "&gt;&gt;"):
        if separator in transformation:
            left, right = transformation.split(separator, 1)
            return f"{right}{separator}{left}"
    return transformation


def count_missing_reverse_edges(input_file: Path) -> int:
    with open_graphml(input_file, "rb") as handle:
        data = handle.read()

    data_bytes = BytesIO(data)
    reader = nk.graphio.GraphMLReader()
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            graph = reader.read(data_bytes)
        finally:
            sys.stdout = old_stdout

    total = graph.numberOfEdges()
    undirected = nk.graphtools.toUndirected(graph)
    undirected.removeMultiEdges()
    return len(list(undirected.iterEdges())) * 2 - total


def write_reversible_network(input_file: Path, output_file: Path) -> None:
    unmatched: dict[
        tuple[str, str],
        tuple[list[bytes], str | None, bool],
    ] = {}

    with bz2.open(input_file, "rb") as in_f, bz2.open(output_file, "wb") as out_f:
        collecting = False
        in_data = False
        current_source = current_target = None
        current_transformation: str | None = None
        current_uses_escaped_gt = False
        edge_lines: list[bytes] = []

        for line_bytes in in_f:
            line = line_bytes.decode("utf-8", errors="replace")

            edge_match = EDGE_OPEN_RE.search(line)
            if edge_match:
                current_source, current_target = edge_match.group(1), edge_match.group(2)
                collecting = True
                in_data = False
                current_transformation = None
                current_uses_escaped_gt = False
                edge_lines = [line_bytes]
                continue

            if collecting:
                edge_lines.append(line_bytes)

                if '<data key="d1">' in line:
                    uses_escaped_gt = "&gt;&gt;" in line
                    data_match = DATA_D1_RE.search(line)
                    if data_match:
                        current_transformation = unescape_xml(data_match.group(1))
                        current_uses_escaped_gt = uses_escaped_gt
                    elif "</data>" not in line:
                        start = line.find('<data key="d1">') + len('<data key="d1">')
                        current_transformation = line[start:]
                        current_uses_escaped_gt = uses_escaped_gt
                        in_data = True

                if in_data:
                    if "</data>" in line:
                        end = line.find("</data>")
                        current_transformation = unescape_xml(
                            (current_transformation or "") + line[:end]
                        )
                        in_data = False
                    elif current_transformation is not None:
                        current_transformation += line.strip()

                if "</edge>" in line:
                    collecting = False
                    inverse_key = (current_target, current_source)
                    if inverse_key in unmatched:
                        inverse_lines, _, _ = unmatched.pop(inverse_key)
                        out_f.writelines(edge_lines)
                        out_f.writelines(inverse_lines)
                    else:
                        unmatched[(current_source, current_target)] = (
                            edge_lines,
                            current_transformation,
                            current_uses_escaped_gt,
                        )
                    edge_lines = []
                    current_source = current_target = None
                    current_transformation = None
                    in_data = False
                continue

            if GRAPH_CLOSE_RE.match(line):
                for (source, target), (stored_lines, transformation, uses_gt) in (
                    unmatched.items()
                ):
                    out_f.writelines(stored_lines)
                    flipped = (
                        flip_transformation(transformation) if transformation else ""
                    )
                    formatted = escape_xml_for_graphml(flipped, uses_gt)
                    out_f.write(
                        f'    <edge source="{target}" target="{source}">\n'.encode("utf-8")
                    )
                    out_f.write(
                        f'      <data key="d1">{formatted}</data>\n'.encode("utf-8")
                    )
                    out_f.write(b"    </edge>\n")

                out_f.write(line_bytes)
                continue

            out_f.write(line_bytes)

        out_f.flush()


def main() -> int:
    args = parse_arguments()

    if not args.input_file.is_file():
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        return 1

    output_file = args.output_file or default_reversable_output(args.input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    missing = count_missing_reverse_edges(args.input_file)
    if missing == 0:
        shutil.copyfile(args.input_file, output_file)
    else:
        write_reversible_network(args.input_file, output_file)

    label = args.input_file.name.replace(".graphml.bz2", "").replace(".graphml", "")
    print(f"{label}\t:\t{missing} missing reverse edge(s)")
    print(f"Written to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
