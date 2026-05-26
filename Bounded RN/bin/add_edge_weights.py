#!/usr/bin/env python3
"""
Add edge weights to GraphML networks using node energies in the file, and add
missing reverse edges (reversible network).

Processes one or more inputs, or scans the project root (one stoichiometry level
deep) when no files are given. Skips existing ``*_wk*`` outputs. Writes
``<stoich>_graph_weight_counts.csv`` at the project root.

Authors:
    Miko Stulajter

Version 3.0.0

Usage:
  add_edge_weights.py
  add_edge_weights.py [-h] [-o OUTPUT] INPUT.graphml.bz2 [INPUT2 ...]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import bz2

from _repo_paths import ensure_repo_bin_on_path

ensure_repo_bin_on_path()

from graphml_utils import (
    EDGE_OPEN_RE,
    NODE_OPEN_RE,
    compute_edge_weight,
    default_wk_output,
    is_weighted_graphml,
    iter_graphml_lines,
    open_graphml,
    unescape_xml,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SKIP_DIR_NAMES = frozenset({"bin", "paths"})
_WEIGHT_COUNTS_CSV = "graph_weight_counts.csv"
_COUNT_FIELDNAMES = ("graph", "nodes", "edges", "nodes_all", "edges_all")
_GRAPH_CLOSE_RE = re.compile(r"\s*</graph>\s*$")


def _collect_graphml(folder: Path, found: set[Path]) -> None:
    for pattern in ("*.graphml.bz2", "*.graphml"):
        for path in folder.glob(pattern):
            if not is_weighted_graphml(path):
                found.add(path)


def discover_networks(directory: Path) -> list[Path]:
    """GraphML inputs under directory (one level deep), excluding weighted outputs."""
    found: set[Path] = set()
    _collect_graphml(directory, found)
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir() or subdir.name in _SKIP_DIR_NAMES:
            continue
        _collect_graphml(subdir, found)
    return sorted(found)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add edge weights and make networks reversible.",
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        type=Path,
        help="GraphML file(s); default: scan project root one level deep",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=PROJECT_ROOT,
        help=f"Scan directory when no inputs given (default: {PROJECT_ROOT})",
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        help="Output path (only with a single input file)",
    )
    return parser.parse_args()


def _scan_keys(filepath: Path) -> tuple[dict[str, str], dict[str, str], set[str]]:
    node_keys: dict[str, str] = {}
    edge_keys: dict[str, str] = {}
    all_ids: set[str] = set()
    key_re = re.compile(
        r'<key\s+id="([^"]+)"\s+for="(node|edge)"\s+attr\.name="([^"]+)"'
    )
    for line in iter_graphml_lines(filepath):
        match = key_re.search(line)
        if match:
            kid, kfor, kname = match.group(1), match.group(2), match.group(3)
            all_ids.add(kid)
            (node_keys if kfor == "node" else edge_keys)[kname] = kid
        if "<graph " in line or line.strip() == "<graph>":
            break
    return node_keys, edge_keys, all_ids


def graph_output_label(output_path: Path) -> str:
    name = output_path.name
    for suffix in (".graphml.bz2", ".graphml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return output_path.stem


def count_missing_reverse_edges(pairs: set[tuple[str, str]]) -> int:
    return sum(1 for src, tgt in pairs if (tgt, src) not in pairs)


def parse_nodes_and_edges(input_file: Path) -> tuple[
    dict[str, float],
    dict[str, float],
    set[str],
    set[tuple[str, str]],
    set[tuple[str, str]],
    dict[tuple[str, str], list[bytes]],
    dict[str, int],
]:
    node_keys, _, _ = _scan_keys(input_file)
    energy_key = node_keys.get("energy")
    if not energy_key:
        raise ValueError(
            f"Input must have node attr 'energy' (found: {list(node_keys)})"
        )

    node_energies: dict[str, float] = {}
    valid_node_ids: set[str] = set()
    total_nodes = 0
    missing_energy = 0
    energy_re = re.compile(rf'<data key="{re.escape(energy_key)}">(.*?)</data>')

    in_node = False
    cur_id: str | None = None
    cur_e: float | None = None

    for line in iter_graphml_lines(input_file):
        node_match = NODE_OPEN_RE.search(line)
        if node_match:
            cur_id, cur_e = node_match.group(1), None
            in_node = True
            continue
        if not in_node:
            continue
        energy_match = energy_re.search(line)
        if energy_match:
            try:
                cur_e = float(energy_match.group(1))
            except ValueError:
                pass
        if "</node>" in line:
            in_node = False
            total_nodes += 1
            if cur_id is not None and cur_e is not None:
                node_energies[cur_id] = cur_e
                valid_node_ids.add(cur_id)
            else:
                missing_energy += 1

    edge_weights: dict[str, float] = {}
    valid_edge_ids: set[tuple[str, str]] = set()
    existing_pairs: set[tuple[str, str]] = set()
    edge_data_map: dict[tuple[str, str], list[bytes]] = {}
    total_edges = 0

    with open_graphml(input_file, "rb") as handle:
        in_edge = False
        cur_src = cur_tgt = None
        cur_data: list[bytes] = []

        for raw in handle:
            line = raw.decode("utf-8", errors="replace")
            edge_match = EDGE_OPEN_RE.search(line)
            if edge_match:
                cur_src, cur_tgt = edge_match.group(1), edge_match.group(2)
                in_edge, cur_data = True, []
                continue
            if not in_edge:
                continue
            if "</edge>" in line:
                in_edge = False
                total_edges += 1
                existing_pairs.add((cur_src, cur_tgt))
                se = node_energies.get(cur_src)
                te = node_energies.get(cur_tgt)
                if se is not None and te is not None:
                    w = compute_edge_weight(se, te)
                    edge_weights[f"{cur_src}-{cur_tgt}"] = w
                    valid_edge_ids.add((cur_src, cur_tgt))
                    edge_data_map[(cur_src, cur_tgt)] = cur_data[:]
            else:
                cur_data.append(raw)

    reverse_all = count_missing_reverse_edges(existing_pairs)
    reverse_kept = count_missing_reverse_edges(valid_edge_ids)
    counts = {
        "nodes_all": total_nodes,
        "edges_all": total_edges + reverse_all,
        "nodes": len(valid_node_ids),
        "edges": len(valid_edge_ids) + reverse_kept,
    }

    print(
        f"Removed {total_nodes - counts['nodes']} nodes out of {total_nodes} "
        f"(missing energy: {missing_energy})"
    )
    print(
        f"Removed {total_edges - len(valid_edge_ids)} edges out of {total_edges}, "
        f"keeping {len(valid_edge_ids)} (+{reverse_kept} reverse)"
    )
    print(
        f"Counts (reversible): all {counts['nodes_all']} nodes, "
        f"{counts['edges_all']} edges; "
        f"with energy {counts['nodes']} nodes, {counts['edges']} edges"
    )

    return (
        node_energies,
        edge_weights,
        valid_node_ids,
        valid_edge_ids,
        existing_pairs,
        edge_data_map,
        counts,
    )


def _flip_transformation(data_lines: list[bytes]) -> list[bytes]:
    flipped: list[bytes] = []
    for raw in data_lines:
        line = raw.decode("utf-8", errors="replace")
        match = re.search(r"(<data key=\"[^\"]+\">)(.*?)(</data>)", line)
        if match:
            prefix, content, suffix = match.group(1), match.group(2), match.group(3)
            for sep in ("&gt;&gt;", ">>"):
                if sep in content:
                    left, right = content.split(sep, 1)
                    flipped.append(
                        f"{prefix}{right}{sep}{left}{suffix}\n".encode("utf-8")
                    )
                    break
            else:
                flipped.append(raw)
        else:
            flipped.append(raw)
    return flipped


def write_output(
    input_file: Path,
    output_file: Path,
    node_energies: dict[str, float],
    edge_weights: dict[str, float],
    valid_node_ids: set[str],
    valid_edge_ids: set[tuple[str, str]],
    existing_pairs: set[tuple[str, str]],
    edge_data_map: dict[tuple[str, str], list[bytes]],
) -> None:
    reverse_to_add: dict[tuple[str, str], float] = {}
    for src, tgt in valid_edge_ids:
        if (tgt, src) not in existing_pairs:
            reverse_to_add[(tgt, src)] = compute_edge_weight(
                node_energies[tgt], node_energies[src]
            )
    print(f"Adding {len(reverse_to_add)} reverse edges")

    _, _, all_ids = _scan_keys(input_file)
    weight_key = next(f"d{i}" for i in range(100) if f"d{i}" not in all_ids)

    with open_graphml(input_file, "rb") as inf, bz2.open(output_file, "wb") as outf:
        weight_key_written = False
        last_was_key = False
        in_node = collecting_node = False
        in_edge = collecting_edge = False
        node_buf: list[bytes] = []
        edge_buf: list[bytes] = []
        cur_nid = cur_esrc = cur_etgt = None

        for raw in inf:
            line = raw.decode("utf-8", errors="replace")

            if "<key " in line:
                last_was_key = True
                outf.write(raw)
                continue
            if last_was_key:
                last_was_key = False
                if not weight_key_written:
                    outf.write(
                        f'  <key id="{weight_key}" for="edge" '
                        f'attr.name="weight" attr.type="float" />\n'.encode("utf-8")
                    )
                    weight_key_written = True

            node_match = NODE_OPEN_RE.search(line)
            if node_match:
                cur_nid = node_match.group(1)
                in_node = True
                collecting_node = cur_nid in valid_node_ids
                node_buf = [raw] if collecting_node else []
                continue

            if in_node:
                if collecting_node:
                    node_buf.append(raw)
                    if "</node>" in line:
                        outf.writelines(node_buf)
                        collecting_node = in_node = False
                elif "</node>" in line:
                    in_node = False
                continue

            edge_match = EDGE_OPEN_RE.search(line)
            if edge_match:
                cur_esrc, cur_etgt = edge_match.group(1), edge_match.group(2)
                in_edge = True
                collecting_edge = (cur_esrc, cur_etgt) in valid_edge_ids
                edge_buf = [raw] if collecting_edge else []
                continue

            if in_edge:
                if collecting_edge:
                    if "</edge>" in line:
                        outf.writelines(edge_buf)
                        eid = f"{cur_esrc}-{cur_etgt}"
                        weight = edge_weights.get(eid)
                        if weight is not None:
                            outf.write(
                                f'      <data key="{weight_key}">{weight:.9f}</data>\n'.encode(
                                    "utf-8"
                                )
                            )
                        outf.write(raw)
                        collecting_edge = in_edge = False
                    else:
                        edge_buf.append(raw)
                elif "</edge>" in line:
                    in_edge = False
                continue

            if _GRAPH_CLOSE_RE.match(line):
                for (rsrc, rtgt), rw in reverse_to_add.items():
                    orig = (rtgt, rsrc)
                    data = edge_data_map.get(orig, [])
                    outf.write(
                        f'    <edge source="{rsrc}" target="{rtgt}">\n'.encode("utf-8")
                    )
                    for chunk in _flip_transformation(data):
                        outf.write(chunk)
                    outf.write(
                        f'      <data key="{weight_key}">{rw:.9f}</data>\n'.encode("utf-8")
                    )
                    outf.write(b"    </edge>\n")
                outf.write(raw)
                continue

            outf.write(raw)
        outf.flush()


def weight_counts_csv_path(project_root: Path, stoich: str) -> Path:
    return project_root / f"{stoich}_{_WEIGHT_COUNTS_CSV}"


def load_existing_weight_counts(folder: Path) -> dict[str, dict[str, object]]:
    stoich = folder.name
    project_root = folder.parent
    for path in (
        weight_counts_csv_path(project_root, stoich),
        folder / _WEIGHT_COUNTS_CSV,
    ):
        if not path.is_file():
            continue
        by_graph: dict[str, dict[str, object]] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                graph = (row.get("graph") or "").strip()
                if graph:
                    by_graph[graph] = dict(row)
        return by_graph
    return {}


def write_weight_counts_csv(folder: Path, rows: list[dict[str, object]]) -> None:
    out_path = weight_counts_csv_path(folder.parent, folder.name)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COUNT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


def process_network(input_file: Path, output_file: Path) -> dict[str, object]:
    (
        node_energies,
        edge_weights,
        valid_node_ids,
        valid_edge_ids,
        existing_pairs,
        edge_data_map,
        counts,
    ) = parse_nodes_and_edges(input_file)

    write_output(
        input_file,
        output_file,
        node_energies,
        edge_weights,
        valid_node_ids,
        valid_edge_ids,
        existing_pairs,
        edge_data_map,
    )
    print(f"Written to {output_file}")
    return {"graph": graph_output_label(output_file), **counts}


def main() -> int:
    args = parse_arguments()

    inputs = list(args.input_files) if args.input_files else discover_networks(args.directory)
    if not inputs:
        print(f"No GraphML networks found in {args.directory}", file=sys.stderr)
        return 1

    if args.output_file is not None and len(inputs) != 1:
        print("-o requires exactly one input file", file=sys.stderr)
        return 2

    by_folder: dict[Path, list[Path]] = defaultdict(list)
    for input_path in inputs:
        if input_path.is_file():
            by_folder[input_path.parent].append(input_path)

    for folder in sorted(by_folder):
        existing = load_existing_weight_counts(folder)
        n_skip = n_new = 0
        for input_path in sorted(by_folder[folder]):
            output_path = (
                args.output_file if args.output_file is not None else default_wk_output(input_path)
            )
            label = graph_output_label(output_path)
            if label in existing and output_path.is_file():
                print(f"Skip (already done): {label}")
                n_skip += 1
                continue
            print(f"\n=== {input_path.name} ===")
            existing[label] = process_network(input_path, output_path)
            n_new += 1

        if existing:
            print(
                f"{folder.name}: {len(existing)} graph(s) in CSV "
                f"({n_skip} skipped, {n_new} new this run)"
            )
            rows = sorted(existing.values(), key=lambda r: str(r["graph"]))
            write_weight_counts_csv(folder, rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
