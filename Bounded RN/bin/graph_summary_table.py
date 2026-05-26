#!/usr/bin/env python3
"""
Build a summary table for each stoichiometry folder under the Bounded RN project.

Within each folder, one row per weighted network (*_wk.graphml.bz2).
If none exist, uses raw *.graphml(.bz2) excluding *_wk* outputs.

Writes <project_root>/<stoich>_graph_summary_table.csv at the project root.

Loads existing tables, scans only graphs not already listed, merges, and writes back.
Optional combined CSV at project root via -o.

Usage:
  graph_summary_table.py [-r PROJECT_ROOT]
  graph_summary_table.py -r /path/to/Bounded_RN -o graph_summary_table.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

from _repo_paths import ensure_repo_bin_on_path

ensure_repo_bin_on_path()

from graphml_utils import is_weighted_graphml, open_graphml, strip_xml_namespace

RDLogger.DisableLog("rdApp.*")

_NEUTRAL_CACHE_SIZE = 500_000
_SKIP_DIR_NAMES = frozenset({"bin", "paths"})
_WEIGHT_COUNTS_CSV = "graph_weight_counts.csv"
_SUMMARY_CSV = "graph_summary_table.csv"
_MASTER_EQUIV_CSV = "bounded_master_molecules_equivalent.csv"


@lru_cache(maxsize=_NEUTRAL_CACHE_SIZE)
def is_neutral(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.GetFormalCharge(mol) == 0
    return False


def discover_smiles_key_id(path: Path) -> str | None:
    with open_graphml(path) as f:
        for event, elem in ET.iterparse(f, events=("start", "end")):
            tag = strip_xml_namespace(elem.tag)
            if tag == "graph" and event == "start":
                elem.clear()
                break
            if tag == "key" and event == "end":
                if elem.get("for") == "node" and elem.get("attr.name") == "smiles":
                    kid = elem.get("id")
                    elem.clear()
                    return kid
                elem.clear()
            elif event == "end":
                elem.clear()
    return None


def node_combined_smiles_is_neutral(combined: str) -> bool:
    if "." not in combined:
        s = combined.strip()
        return bool(s) and is_neutral(s)
    parts = [p.strip() for p in combined.split(".") if p.strip()]
    if not parts:
        return False
    return all(is_neutral(p) for p in parts)


def load_fragment_canonical(equivalent_csv: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    with equivalent_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "smiles" not in r.fieldnames:
            raise ValueError(
                f"{equivalent_csv}: expected column 'smiles', got {r.fieldnames!r}"
            )
        has_equiv = "equivalent" in r.fieldnames
        for row in r:
            left = (row.get("smiles") or "").strip()
            raw = (row.get("equivalent") or "").strip() if has_equiv else ""
            parts = [p.strip() for p in raw.split("|") if p.strip()] if raw else []
            if parts:
                canon = parts[0]
                members: list[str] = []
                if left:
                    members.append(left)
                members.extend(parts)
            else:
                if not left:
                    continue
                canon = left
                members = [left]
            for frag in members:
                if frag not in m:
                    m[frag] = canon
    return m


def normalize_node_smiles(combined: str, frag_map: dict[str, str]) -> str:
    parts = [p.strip() for p in combined.split(".") if p.strip()]
    if not parts:
        return ""
    normalized = [frag_map.get(p, p) for p in parts]
    normalized.sort()
    return ".".join(normalized)


def count_equivalence_classes(
    equivalent_csv: Path, graph_molecules: set[str]
) -> int:
    classes: set[str | tuple[str, ...]] = set()
    mapped: set[str] = set()
    with equivalent_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "equivalent" not in r.fieldnames:
            raise ValueError(
                f"{equivalent_csv}: expected column 'equivalent', got {r.fieldnames!r}"
            )
        for row in r:
            smi = (row.get("smiles") or "").strip()
            if smi not in graph_molecules:
                continue
            mapped.add(smi)
            raw = (row.get("equivalent") or "").strip()
            if "|" in raw:
                classes.add(tuple(sorted(p.strip() for p in raw.split("|") if p.strip())))
            else:
                classes.add(raw)
    for smi in graph_molecules - mapped:
        classes.add(smi)
    return len(classes)


def scan_graph(path: Path, smiles_key_id: str) -> dict[str, object]:
    is_neutral.cache_clear()
    n_nodes = 0
    n_edges = 0
    n_neutral = 0
    unique_mols: set[str] = set()
    neutral_smiles: list[str] = []

    with open_graphml(path) as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            tag = strip_xml_namespace(elem.tag)
            if tag == "node":
                n_nodes += 1
                combined = None
                for child in elem:
                    if strip_xml_namespace(child.tag) != "data":
                        continue
                    if child.get("key") == smiles_key_id:
                        combined = (child.text or "").strip()
                        break
                elem.clear()
                if combined:
                    for frag in combined.split("."):
                        frag = frag.strip()
                        if frag:
                            unique_mols.add(frag)
                    if node_combined_smiles_is_neutral(combined):
                        n_neutral += 1
                        neutral_smiles.append(combined)
            elif tag == "edge":
                n_edges += 1
                elem.clear()

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "unique_molecules": len(unique_mols),
        "neutral_nodes": n_neutral,
        "_graph_molecules": unique_mols,
        "_neutral_smiles": neutral_smiles,
    }


def count_merged_neutral(neutral_smiles: list[str], frag_map: dict[str, str]) -> int:
    seen: set[str] = set()
    for smi in neutral_smiles:
        key = normalize_node_smiles(smi, frag_map)
        if key:
            seen.add(key)
    return len(seen)


def find_graph_files(folder: Path) -> list[Path]:
    """Weighted (_wk) networks first; otherwise raw exports in this folder only."""
    weighted = sorted(
        p
        for pattern in ("*.graphml.bz2", "*.graphml")
        for p in folder.glob(pattern)
        if is_weighted_graphml(p)
    )
    if weighted:
        return weighted

    return sorted(
        p
        for pattern in ("*.graphml.bz2", "*.graphml")
        for p in folder.glob(pattern)
        if not is_weighted_graphml(p)
    )


def discover_stoich_dirs(project_root: Path, exclude: set[Path]) -> list[Path]:
    out: list[Path] = []
    for d in sorted(project_root.iterdir()):
        if not d.is_dir() or d.name in _SKIP_DIR_NAMES or d.resolve() in exclude:
            continue
        if find_graph_files(d):
            out.append(d)
    return out


def graph_label(path: Path) -> str:
    name = path.name
    for suffix in (".graphml.bz2", ".graphml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def summary_csv_path(project_root: Path, stoich: str) -> Path:
    return project_root / f"{stoich}_{_SUMMARY_CSV}"


def load_existing_summary(
    folder: Path, project_root: Path
) -> dict[str, dict[str, object]]:
    """Load prior summary rows from project root (legacy: stoich folder copy)."""
    name = folder.name
    for path in (
        summary_csv_path(project_root, name),
        folder / _SUMMARY_CSV,
    ):
        if not path.is_file():
            continue
        by_graph: dict[str, dict[str, object]] = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                graph = (row.get("graph") or "").strip()
                if graph:
                    by_graph[graph] = dict(row)
        return by_graph
    return {}


def weight_counts_csv_path(project_root: Path, stoich: str) -> Path:
    return project_root / f"{stoich}_{_WEIGHT_COUNTS_CSV}"


def load_weight_counts(folder: Path, project_root: Path) -> dict[str, dict[str, int]]:
    name = folder.name
    path: Path | None = None
    for candidate in (
        weight_counts_csv_path(project_root, name),
        folder / _WEIGHT_COUNTS_CSV,
    ):
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        return {}
    out: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            graph = (row.get("graph") or "").strip()
            if not graph:
                continue
            try:
                out[graph] = {
                    "nodes": int(row["nodes"]),
                    "edges": int(row["edges"]),
                    "nodes_all": int(row["nodes_all"]),
                    "edges_all": int(row["edges_all"]),
                }
            except (KeyError, ValueError):
                continue
    return out


def resolve_equivalence_table(
    project_root: Path, folder: Path
) -> tuple[Path | None, dict[str, str] | None]:
    """Load master equivalence map; fall back to per-stoich CSV."""
    master = project_root / _MASTER_EQUIV_CSV
    if master.is_file():
        try:
            return master, load_fragment_canonical(master)
        except ValueError as e:
            print(f"Warning: {master}: {e}", file=sys.stderr)
            return master, None

    fallback = folder / f"{folder.name}_molecules_equivalent.csv"
    if fallback.is_file():
        try:
            return fallback, load_fragment_canonical(fallback)
        except ValueError as e:
            print(f"Warning {folder.name}: {e}", file=sys.stderr)
            return fallback, None
    return None, None


def apply_equivalence_stats(
    equiv_path: Path,
    frag_map: dict[str, str],
    stats: dict[str, object],
    *,
    log_prefix: str = "",
) -> tuple[int | str, int | str]:
    n_equiv: int | str = ""
    n_merged: int | str = ""
    try:
        n_equiv = count_equivalence_classes(
            equiv_path, stats["_graph_molecules"]  # type: ignore[arg-type]
        )
    except ValueError as e:
        print(f"Warning {log_prefix}equivalence_classes: {e}", file=sys.stderr)
    try:
        n_merged = count_merged_neutral(
            stats["_neutral_smiles"], frag_map  # type: ignore[arg-type]
        )
    except ValueError as e:
        print(f"Warning {log_prefix}neutral_nodes_merged: {e}", file=sys.stderr)
    return n_equiv, n_merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-folder summary tables for Bounded RN GraphML networks.",
    )
    p.add_argument(
        "-r",
        "--root",
        type=Path,
        default=None,
        help="Project root (default: parent of bin/)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional combined CSV at project root (columns: folder, graph, ...)",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scan all graphs (ignore existing per-folder CSV rows)",
    )
    p.add_argument(
        "--update-equiv",
        action="store_true",
        help="For graphs already in CSV, recompute equivalence_classes and "
        "neutral_nodes_merged only (use after adding master equivalence table)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = (args.root or script_dir.parent).resolve()
    exclude = {script_dir.resolve()}
    combined_path = args.output.resolve() if args.output else None

    stoich_dirs = discover_stoich_dirs(project_root, exclude)
    if not stoich_dirs:
        print(
            f"No folders with GraphML networks under {project_root}",
            file=sys.stderr,
        )
        return 1

    fieldnames = [
        "graph",
        "nodes",
        "edges",
        "nodes_all",
        "edges_all",
        "unique_molecules",
        "equivalence_classes",
        "neutral_nodes",
        "neutral_nodes_merged",
    ]
    combined_fieldnames = ["folder", *fieldnames]
    combined_rows: list[dict[str, object]] = []
    folders_written = 0

    master_equiv = project_root / _MASTER_EQUIV_CSV
    shared_frag_map: dict[str, str] | None = None
    shared_equiv_path: Path | None = None
    if master_equiv.is_file():
        try:
            shared_frag_map = load_fragment_canonical(master_equiv)
            shared_equiv_path = master_equiv
            print(f"Using equivalence table: {master_equiv.name}")
        except ValueError as e:
            print(f"Warning: {master_equiv}: {e}", file=sys.stderr)
    else:
        print(
            f"Warning: missing {master_equiv.name} at project root "
            "(will try per-stoich *_molecules_equivalent.csv)",
            file=sys.stderr,
        )

    for folder in stoich_dirs:
        name = folder.name
        graphs = find_graph_files(folder)
        if shared_equiv_path is not None and shared_frag_map is not None:
            equiv_path = shared_equiv_path
            frag_map = shared_frag_map
            equiv_available = True
        else:
            equiv_path, frag_map = resolve_equivalence_table(project_root, folder)
            equiv_available = equiv_path is not None and frag_map is not None
            if not equiv_available and graphs:
                print(
                    f"Warning {name}: no equivalence table "
                    "(equivalence_classes and neutral_nodes_merged left empty)",
                    file=sys.stderr,
                )

        if args.refresh:
            rows_by_graph = {}
        else:
            rows_by_graph = load_existing_summary(folder, project_root)
        n_skip = 0
        n_new = 0
        n_equiv_updated = 0
        weight_counts = load_weight_counts(folder, project_root)
        for graph_path in graphs:
            label = graph_label(graph_path)
            if label in rows_by_graph and not args.refresh:
                if (
                    args.update_equiv
                    and equiv_available
                    and frag_map is not None
                    and equiv_path is not None
                ):
                    smiles_key = discover_smiles_key_id(graph_path)
                    if smiles_key:
                        stats = scan_graph(graph_path, smiles_key)
                        n_equiv, n_merged = apply_equivalence_stats(
                            equiv_path,
                            frag_map,
                            stats,
                            log_prefix=f"{name}/{label}: ",
                        )
                        row = rows_by_graph[label]
                        row["equivalence_classes"] = n_equiv
                        row["neutral_nodes_merged"] = n_merged
                        n_equiv_updated += 1
                        print(
                            f"Update equiv {name}/{label}: "
                            f"equiv_classes={n_equiv} merged={n_merged}"
                        )
                print(f"Skip (already in table): {name}/{label}")
                n_skip += 1
                continue

            smiles_key = discover_smiles_key_id(graph_path)
            if not smiles_key:
                print(
                    f"Skip {label}: no node smiles key in {graph_path.name}",
                    file=sys.stderr,
                )
                continue

            stats = scan_graph(graph_path, smiles_key)
            counts = weight_counts.get(label)
            if counts:
                n_nodes = counts["nodes"]
                n_edges = counts["edges"]
                nodes_all: int | str = counts["nodes_all"]
                edges_all: int | str = counts["edges_all"]
            else:
                n_nodes = stats["nodes"]
                n_edges = stats["edges"]
                nodes_all = ""
                edges_all = ""

            n_equiv: int | str = ""
            n_merged: int | str = ""
            if equiv_available and frag_map is not None and equiv_path is not None:
                n_equiv, n_merged = apply_equivalence_stats(
                    equiv_path,
                    frag_map,
                    stats,
                    log_prefix=f"{name}/{label}: ",
                )

            row = {
                "graph": label,
                "nodes": n_nodes,
                "edges": n_edges,
                "nodes_all": nodes_all,
                "edges_all": edges_all,
                "unique_molecules": stats["unique_molecules"],
                "equivalence_classes": n_equiv,
                "neutral_nodes": stats["neutral_nodes"],
                "neutral_nodes_merged": n_merged,
            }
            rows_by_graph[label] = row
            n_new += 1
            raw_note = (
                f" nodes_all={nodes_all} edges_all={edges_all}"
                if nodes_all != ""
                else ""
            )
            print(
                f"{name}/{label}: nodes={n_nodes} edges={n_edges}"
                f"{raw_note} "
                f"unique_mols={stats['unique_molecules']} equiv_classes={n_equiv} "
                f"neutral={stats['neutral_nodes']} merged={n_merged}"
            )

        if not rows_by_graph:
            continue

        rows = sorted(rows_by_graph.values(), key=lambda r: str(r["graph"]))
        for row in rows:
            combined_rows.append({"folder": name, **row})
        if rows:
            msg = (
                f"{name}: {len(rows)} row(s) in table "
                f"({n_skip} skipped, {n_new} new this run"
            )
            if n_equiv_updated:
                msg += f", {n_equiv_updated} equiv columns updated"
            msg += ")"
            print(msg)

        out_path = summary_csv_path(project_root, name)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {out_path}")
        folders_written += 1

    if combined_path:
        with combined_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=combined_fieldnames)
            w.writeheader()
            w.writerows(combined_rows)
        print(f"Wrote {len(combined_rows)} rows -> {combined_path}")

    if folders_written == 0 and not combined_path:
        print("No rows written.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
