#!/usr/bin/env python3
"""
Add edge weights to a GraphML network from per-molecule energies in a CSV.

Nodes without energies for every fragment SMILES are removed; edges touching
removed nodes are dropped. Valid edges receive a ``weight`` attribute (d2).

Authors:
    Miko Stulajter

Version 2.0.0

Usage:
    add_edge_weights.py [-h] [-En ENERGY_COLUMN] [-o OUTPUT] \\
        INPUT.graphml.bz2 ENERGIES.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bz2
import pandas as pd
from rdkit import RDLogger

from _repo_paths import ensure_repo_bin_on_path

ensure_repo_bin_on_path()

from graphml_utils import (
    DATA_D0_RE,
    EDGE_OPEN_RE,
    NODE_OPEN_RE,
    canonicalize_multimol_smiles,
    compute_edge_weight,
    count_nodes_and_edges,
    default_weighted_output,
    iter_graphml_lines,
    unescape_xml,
)

RDLogger.DisableLog("rdApp.*")

WEIGHT_KEY_LINE = (
    '  <key id="d2" for="edge" attr.name="weight" attr.type="float" />\n'
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add edge weights from molecule energies; drop nodes/edges without data.",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input GraphML file (.graphml or .graphml.bz2)",
    )
    parser.add_argument(
        "input_energy_file",
        type=Path,
        help="CSV with smiles and energy columns",
    )
    parser.add_argument(
        "-En",
        dest="energy_column",
        default="energy",
        help="Energy column name (default: energy)",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        help="Output GraphML path (default: *_weighted.graphml.bz2 next to input)",
    )
    return parser.parse_args()


def load_energy_table(path: Path, energy_column: str) -> dict[str, float]:
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"CSV must contain a 'smiles' column: {path}")
    if energy_column not in df.columns:
        raise ValueError(f"CSV must contain '{energy_column}' column: {path}")

    energy_dict: dict[str, float] = {}
    for _, row in df.iterrows():
        smiles = str(row["smiles"]).strip()
        value = row[energy_column]
        if pd.isna(value):
            continue
        energy = float(value)
        canonical = canonicalize_multimol_smiles(smiles)
        if canonical:
            energy_dict[canonical] = energy
            if canonical != smiles:
                energy_dict[smiles] = energy
    return energy_dict


def _lookup_fragment_energy(fragment: str, energy_dict: dict[str, float]) -> float | None:
    energy = energy_dict.get(fragment)
    if energy is not None:
        return energy
    canonical = canonicalize_multimol_smiles(fragment)
    if canonical:
        return energy_dict.get(canonical)
    return None


def _node_energy(
    node_smiles: str, energy_dict: dict[str, float], missing: set[str]
) -> float | None:
    total = 0.0
    for fragment in node_smiles.split("."):
        energy = _lookup_fragment_energy(fragment, energy_dict)
        if energy is None:
            missing.add(fragment)
            return None
        total += energy
    return total


def parse_nodes_and_edges(
    input_file: Path, energy_dict: dict[str, float]
) -> tuple[dict[str, float], dict[str, float], set[str], set[tuple[str, str]]]:
    node_energies: dict[str, float] = {}
    valid_node_ids: set[str] = set()
    missing_mol: set[str] = set()

    in_node = False
    collecting_smiles = False
    current_node_id: str | None = None
    current_node_smiles: str | None = None

    for line in iter_graphml_lines(input_file):
        node_match = NODE_OPEN_RE.search(line)
        if node_match:
            current_node_id = node_match.group(1)
            current_node_smiles = None
            in_node = True
            collecting_smiles = False
            continue

        if in_node and '<data key="d0">' in line:
            data_match = DATA_D0_RE.search(line)
            if data_match:
                current_node_smiles = unescape_xml(data_match.group(1))
            elif "</data>" not in line:
                start = line.find('<data key="d0">') + len('<data key="d0">')
                current_node_smiles = line[start:]
                collecting_smiles = True
            continue

        if collecting_smiles and in_node:
            if "</data>" in line:
                end = line.find("</data>")
                current_node_smiles = unescape_xml((current_node_smiles or "") + line[:end])
                collecting_smiles = False
            else:
                current_node_smiles = (current_node_smiles or "") + line.strip()
            continue

        if "</node>" in line and in_node:
            if current_node_id and current_node_smiles:
                energy = _node_energy(current_node_smiles, energy_dict, missing_mol)
                if energy is not None:
                    node_energies[current_node_id] = energy
                    valid_node_ids.add(current_node_id)
            in_node = False
            current_node_id = None
            current_node_smiles = None

    edge_weights: dict[str, float] = {}
    valid_edge_ids: set[tuple[str, str]] = set()

    for line in iter_graphml_lines(input_file):
        edge_match = EDGE_OPEN_RE.search(line)
        if not edge_match:
            continue
        source, target = edge_match.group(1), edge_match.group(2)
        source_energy = node_energies.get(source)
        target_energy = node_energies.get(target)
        if source_energy is None or target_energy is None:
            continue
        weight = compute_edge_weight(source_energy, target_energy)
        edge_weights[f"{source}-{target}"] = weight
        valid_edge_ids.add((source, target))

    total_nodes, _ = count_nodes_and_edges(input_file)
    print(f"Removed {total_nodes - len(valid_node_ids)} nodes out of {total_nodes}")
    print(f"Missing energies for {len(missing_mol)} unique fragments")

    return node_energies, edge_weights, valid_node_ids, valid_edge_ids


def write_graphml_with_weights(
    input_file: Path,
    output_file: Path,
    edge_weights: dict[str, float],
    valid_node_ids: set[str],
    valid_edge_ids: set[tuple[str, str]],
) -> None:
    d2_key_written = False

    with bz2.open(input_file, "rb") as in_f, bz2.open(output_file, "wb") as out_f:
        in_node = in_edge = False
        collecting_node = collecting_edge = False
        current_node_id: str | None = None
        current_edge_source = current_edge_target = None
        node_lines: list[bytes] = []
        edge_lines: list[bytes] = []

        for line_bytes in in_f:
            line = line_bytes.decode("utf-8", errors="replace")

            if line.strip().startswith(("<?xml", "<graphml")):
                out_f.write(line_bytes)
                continue

            if not d2_key_written and ("<graph " in line or line.strip() == "<graph>"):
                out_f.write(WEIGHT_KEY_LINE.encode("utf-8"))
                d2_key_written = True

            node_match = NODE_OPEN_RE.search(line)
            if node_match:
                current_node_id = node_match.group(1)
                in_node = True
                collecting_node = current_node_id in valid_node_ids
                node_lines = [line_bytes] if collecting_node else []
                continue

            if in_node:
                if collecting_node:
                    node_lines.append(line_bytes)
                    if "</node>" in line:
                        out_f.writelines(node_lines)
                        collecting_node = in_node = False
                        current_node_id = None
                        node_lines = []
                elif "</node>" in line:
                    in_node = False
                    current_node_id = None
                continue

            edge_match = EDGE_OPEN_RE.search(line)
            if edge_match:
                current_edge_source, current_edge_target = (
                    edge_match.group(1),
                    edge_match.group(2),
                )
                edge_key = (current_edge_source, current_edge_target)
                in_edge = True
                collecting_edge = edge_key in valid_edge_ids
                edge_lines = [line_bytes] if collecting_edge else []
                continue

            if in_edge:
                if collecting_edge:
                    if "</edge>" in line:
                        out_f.writelines(edge_lines)
                        edge_id = f"{current_edge_source}-{current_edge_target}"
                        weight = edge_weights.get(edge_id)
                        if weight is not None:
                            out_f.write(
                                f'      <data key="d2">{weight:.9f}</data>\n'.encode("utf-8")
                            )
                        out_f.write(line_bytes)
                        collecting_edge = in_edge = False
                        current_edge_source = current_edge_target = None
                        edge_lines = []
                    else:
                        edge_lines.append(line_bytes)
                elif "</edge>" in line:
                    in_edge = False
                    current_edge_source = current_edge_target = None
                continue

            out_f.write(line_bytes)

        out_f.flush()


def main() -> int:
    args = parse_arguments()

    if not args.input_file.is_file():
        print(f"Input network not found: {args.input_file}", file=sys.stderr)
        return 1
    if not args.input_energy_file.is_file():
        print(f"Energy file not found: {args.input_energy_file}", file=sys.stderr)
        return 1

    output_file = args.output_file or default_weighted_output(args.input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    energy_dict = load_energy_table(args.input_energy_file, args.energy_column)
    _, edge_weights, valid_node_ids, valid_edge_ids = parse_nodes_and_edges(
        args.input_file, energy_dict
    )

    _, total_edges = count_nodes_and_edges(args.input_file)
    print(
        f"Removed {total_edges - len(edge_weights)} edges, "
        f"keeping {len(edge_weights)}"
    )

    write_graphml_with_weights(
        args.input_file,
        output_file,
        edge_weights,
        valid_node_ids,
        valid_edge_ids,
    )
    print(f"Written to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
