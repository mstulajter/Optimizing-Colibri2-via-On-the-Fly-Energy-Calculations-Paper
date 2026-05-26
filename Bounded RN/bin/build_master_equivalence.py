#!/usr/bin/env python3
"""
Scrape GraphML networks under Bounded RN stoichiometry folders, build one master
SMILES list, then run find_equivalent_mols.py once.

Outputs under project root (default stem ``bounded_master_molecules``):
  bounded_master_molecules.csv
  bounded_master_molecules_equivalent.csv
  bounded_master_molecules_unique.csv
  equiv_master_scrape_manifest.csv

Incremental by default when prior outputs exist. Use --force-equiv to reclassify all.

Authors:
    Miko Stulajter

Usage:
  python build_master_equivalence.py
  python build_master_equivalence.py --collect-only
  python build_master_equivalence.py --equiv-only -j 64
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from _repo_paths import REPO_BIN, ensure_repo_bin_on_path

ensure_repo_bin_on_path()

from graphml_utils import is_weighted_graphml, open_graphml, strip_xml_namespace

_SKIP_DIR_NAMES = frozenset({"bin", "paths"})
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MASTER_STEM = "bounded_master_molecules"
MANIFEST_NAME = "equiv_master_scrape_manifest.csv"
PROTON_SMILES = "[H+]"
AQUEOUS_EXTRAS = ("O", "[OH3+]")


def iter_graphml_paths(folder: Path) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in ("*.graphml.bz2", "*.graphml"):
        for path in sorted(folder.glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def discover_smiles_key_id(path: Path) -> str | None:
    with open_graphml(path) as handle:
        for _event, elem in ET.iterparse(handle, events=("end",)):
            if strip_xml_namespace(elem.tag) == "key" and elem.get("for") == "node":
                if elem.get("attr.name") == "smiles":
                    kid = elem.get("id")
                    elem.clear()
                    return kid
            elif strip_xml_namespace(elem.tag) == "graph":
                elem.clear()
                break
            elem.clear()
    return None


def extract_molecules(graph_path: Path, smiles_key_id: str) -> set[str]:
    molecules: set[str] = set()
    with open_graphml(graph_path) as handle:
        for _event, elem in ET.iterparse(handle, events=("end",)):
            tag = strip_xml_namespace(elem.tag)
            if tag == "node":
                combined = None
                for child in elem:
                    if (
                        strip_xml_namespace(child.tag) == "data"
                        and child.get("key") == smiles_key_id
                    ):
                        combined = (child.text or "").strip()
                        break
                elem.clear()
                if combined:
                    for frag in combined.split("."):
                        frag = frag.strip()
                        if frag:
                            molecules.add(frag)
            elif tag == "edge":
                elem.clear()
    if PROTON_SMILES in molecules:
        molecules.update(AQUEOUS_EXTRAS)
    return molecules


def write_molecules_csv(molecules: set[str], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for smi in sorted(molecules):
            w.writerow([smi])


def load_molecules_csv(path: Path) -> set[str]:
    master: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "smiles" in reader.fieldnames:
            col = "smiles"
        else:
            col = reader.fieldnames[0] if reader.fieldnames else "smiles"
        for row in reader:
            smi = (row.get(col) or "").strip()
            if smi:
                master.add(smi)
    return master


def load_equiv_map_keys(path: Path) -> set[str]:
    return set(load_equiv_map(path))


def load_equiv_map(path: Path) -> dict[str, str | list[str]]:
    out: dict[str, str | list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "equivalent" not in reader.fieldnames:
            raise ValueError(f"{path}: expected columns smiles,equivalent")
        col = "smiles" if "smiles" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            smi = (row.get(col) or "").strip()
            if not smi:
                continue
            equivalent = (row.get("equivalent") or "").strip()
            if not equivalent:
                out[smi] = ""
            elif "|" in equivalent:
                out[smi] = [part.strip() for part in equivalent.split("|") if part.strip()]
            else:
                out[smi] = equivalent
    return out


def count_csv_data_rows(path: Path, column: str = "smiles") -> int:
    n = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = column if reader.fieldnames and column in reader.fieldnames else (
            reader.fieldnames[0] if reader.fieldnames else column
        )
        for row in reader:
            if (row.get(col) or "").strip():
                n += 1
    return n


def representatives_from_equiv_value(value: str | list[str]) -> set[str]:
    if isinstance(value, list):
        return set(value)
    if value:
        return {value}
    return set()


def validate_master_outputs(
    mol_csv: Path,
    equiv_csv: Path,
    unique_csv: Path,
) -> int:
    """
    Cross-check master, equivalence map, and unique representatives.
    Returns 0 if consistent, 1 if problems were found.
    """
    errors: list[str] = []
    warnings: list[str] = []

    print("\nValidating master molecule outputs...", flush=True)

    if not mol_csv.is_file():
        errors.append(f"Missing master list: {mol_csv}")
        _print_validation_report(errors, warnings)
        return 1

    master = load_molecules_csv(mol_csv)
    master_rows = count_csv_data_rows(mol_csv)
    if master_rows != len(master):
        errors.append(
            f"{mol_csv.name}: {master_rows} data row(s) but {len(master)} unique SMILES "
            f"(duplicate or blank rows)"
        )
    print(f"  {mol_csv.name}: {len(master)} unique SMILES", flush=True)

    if not equiv_csv.is_file():
        errors.append(f"Missing equivalence map: {equiv_csv}")
        _print_validation_report(errors, warnings)
        return 1
    if not unique_csv.is_file():
        errors.append(f"Missing unique representatives: {unique_csv}")
        _print_validation_report(errors, warnings)
        return 1

    try:
        equiv = load_equiv_map(equiv_csv)
    except ValueError as exc:
        errors.append(str(exc))
        _print_validation_report(errors, warnings)
        return 1

    equiv_keys = set(equiv)
    equiv_rows = count_csv_data_rows(equiv_csv)
    if equiv_rows != len(equiv_keys):
        errors.append(
            f"{equiv_csv.name}: {equiv_rows} data row(s) but {len(equiv_keys)} unique keys "
            f"(duplicate SMILES keys)"
        )

    missing_in_equiv = sorted(master - equiv_keys)
    extra_in_equiv = sorted(equiv_keys - master)
    if missing_in_equiv:
        errors.append(
            f"{equiv_csv.name}: {len(missing_in_equiv)} SMILES in master missing from map"
        )
        for smi in missing_in_equiv[:5]:
            errors.append(f"  missing: {smi}")
        if len(missing_in_equiv) > 5:
            errors.append(f"  ... and {len(missing_in_equiv) - 5} more")
    if extra_in_equiv:
        errors.append(
            f"{equiv_csv.name}: {len(extra_in_equiv)} map row(s) not in master list"
        )

    empty_equiv = [smi for smi, eq in equiv.items() if not representatives_from_equiv_value(eq)]
    if empty_equiv:
        errors.append(
            f"{equiv_csv.name}: {len(empty_equiv)} row(s) with empty equivalent field"
        )

    expected_unique: set[str] = set()
    for value in equiv.values():
        expected_unique |= representatives_from_equiv_value(value)

    unique = load_molecules_csv(unique_csv)
    unique_rows = count_csv_data_rows(unique_csv)
    if unique_rows != len(unique):
        errors.append(
            f"{unique_csv.name}: {unique_rows} data row(s) but {len(unique)} unique SMILES "
            f"(duplicate or blank rows)"
        )

    only_in_unique = sorted(unique - expected_unique)
    only_in_expected = sorted(expected_unique - unique)
    if only_in_unique or only_in_expected:
        errors.append(
            f"{unique_csv.name}: mismatch with equivalence representatives "
            f"(unique file {len(unique)}, expected {len(expected_unique)})"
        )
        if only_in_expected:
            warnings.append(
                f"{len(only_in_expected)} representative(s) missing from unique file"
            )
        if only_in_unique:
            warnings.append(
                f"{len(only_in_unique)} unique file entr(y/ies) not used as representatives"
            )

    not_in_master = sorted(expected_unique - master)
    if not_in_master:
        warnings.append(
            f"{len(not_in_master)} representative(s) are not in the master list "
            f"(allowed when the class rep is outside the input set)"
        )

    print(f"  {equiv_csv.name}: {len(equiv_keys)} mapped SMILES", flush=True)
    print(f"  {unique_csv.name}: {len(unique)} unique representatives", flush=True)

    _print_validation_report(errors, warnings)
    return 1 if errors else 0


def _print_validation_report(errors: list[str], warnings: list[str]) -> None:
    if warnings:
        print("Validation warnings:", flush=True)
        for msg in warnings:
            print(f"  WARNING: {msg}", flush=True)
    if errors:
        print("Validation FAILED:", flush=True)
        for msg in errors:
            print(f"  ERROR: {msg}", flush=True)
    elif not warnings:
        print("Validation OK: master, equivalence map, and unique file are consistent.", flush=True)
    else:
        print("Validation OK with warnings (see above).", flush=True)


def count_new_molecules_for_equiv(mol_csv: Path, equiv_csv: Path) -> int:
    master = load_molecules_csv(mol_csv)
    if not equiv_csv.is_file():
        return len(master)
    return len(master - load_equiv_map_keys(equiv_csv))


def load_scrape_manifest(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["folder"], row["graph"])
            rows[key] = {
                "folder": row["folder"],
                "graph": row["graph"],
                "n_molecules_in_graph": int(row["n_molecules_in_graph"]),
                "n_new_to_master": int(row["n_new_to_master"]),
            }
    return rows


def discover_folders(root: Path, only: set[str] | None) -> list[Path]:
    out: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in _SKIP_DIR_NAMES:
            continue
        if only is not None and d.name not in only:
            continue
        if iter_graphml_paths(d):
            out.append(d)
    return out


def iter_graph_jobs(folders: list[Path]) -> list[tuple[Path, Path]]:
    return [
        (folder, graph_path)
        for folder in folders
        for graph_path in iter_graphml_paths(folder)
    ]


def collect_master_molecules(
    folders: list[Path],
    *,
    master: set[str] | None = None,
    prior_manifest: dict[tuple[str, str], dict[str, object]] | None = None,
    incremental: bool = False,
) -> tuple[set[str], list[dict[str, object]], int]:
    """Union SMILES from GraphML. Returns (master, manifest_rows, n_new_smiles)."""
    master = set(master) if master is not None else set()
    prior_manifest = prior_manifest or {}
    manifest_rows: list[dict[str, object]] = []
    n_master_start = len(master)
    scraped_keys = set(prior_manifest)

    for folder, graph_path in iter_graph_jobs(folders):
        key = (folder.name, graph_path.name)
        if incremental and key in scraped_keys:
            manifest_rows.append(prior_manifest[key])
            continue

        smiles_key = discover_smiles_key_id(graph_path)
        if not smiles_key:
            raise RuntimeError(
                f"{graph_path}: no node smiles attribute in GraphML keys"
            )
        in_graph = extract_molecules(graph_path, smiles_key)
        before = len(master)
        master |= in_graph
        n_new = len(master) - before
        row = {
            "folder": folder.name,
            "graph": graph_path.name,
            "n_molecules_in_graph": len(in_graph),
            "n_new_to_master": n_new,
        }
        manifest_rows.append(row)
        print(
            f"{folder.name}/{graph_path.name}: "
            f"in_graph={len(in_graph)} new_to_master={n_new} "
            f"master_total={len(master)}",
            flush=True,
        )
    return master, manifest_rows, len(master) - n_master_start


def print_incremental_plan(
    jobs: list[tuple[Path, Path]],
    prior_manifest: dict[tuple[str, str], dict[str, object]],
    master: set[str],
    incremental: bool,
) -> tuple[list[tuple[Path, Path]], list[tuple[str, str]]]:
    """Summarize pending vs already-scraped graphs; return (pending_jobs, stale_keys)."""
    on_disk = {(folder.name, graph_path.name) for folder, graph_path in jobs}
    scraped = set(prior_manifest)
    stale = sorted(scraped - on_disk)
    if incremental:
        pending = [(f, g) for f, g in jobs if (f.name, g.name) not in scraped]
        skipped = len(jobs) - len(pending)
    else:
        pending = list(jobs)
        skipped = 0

    print(f"Existing master SMILES: {len(master)}", flush=True)
    if prior_manifest:
        print(
            f"Scrape manifest: {len(prior_manifest)} graph(s) recorded",
            flush=True,
        )
    print(f"GraphML on disk: {len(jobs)} graph(s) in {len({f for f, _ in jobs})} folder(s)", flush=True)
    if incremental:
        print(
            f"Incremental: skip {skipped} already in manifest, "
            f"scrape {len(pending)} new or updated graph(s)",
            flush=True,
        )
        if stale:
            print(
                f"Manifest entries not on disk (will drop): {len(stale)}",
                flush=True,
            )
            for folder, graph in stale[:10]:
                print(f"  - {folder}/{graph}", flush=True)
            if len(stale) > 10:
                print(f"  ... and {len(stale) - 10} more", flush=True)
    else:
        print(f"Full scrape: all {len(pending)} graph(s)", flush=True)
    return pending, stale


def write_scrape_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    manifest = root / MANIFEST_NAME
    fields = ["folder", "graph", "n_molecules_in_graph", "n_new_to_master"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return manifest


def run_find_equivalent(
    find_equiv_script: Path,
    rules_toml: Path,
    mol_csv: Path,
    equiv_csv: Path,
    unique_csv: Path,
    jobs: int,
    no_progress: bool,
    *,
    full: bool = False,
) -> None:
    cmd = [
        sys.executable,
        str(find_equiv_script),
        "-i",
        str(mol_csv),
        "-r",
        str(rules_toml),
        "-o",
        str(equiv_csv),
        "-u",
        str(unique_csv),
        "-j",
        str(jobs),
    ]
    if full:
        cmd.append("--full")
    if no_progress:
        cmd.append("--no-progress")
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Union SMILES from all GraphML networks under Bounded RN, then run "
            "find_equivalent_mols.py once on the master list."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (default: parent of bin/)",
    )
    p.add_argument(
        "--folder",
        action="append",
        default=None,
        metavar="NAME",
        help="Include only this stoichiometry folder (repeatable)",
    )
    p.add_argument(
        "--master-stem",
        default=DEFAULT_MASTER_STEM,
        help=f"Output file stem under --root (default: {DEFAULT_MASTER_STEM})",
    )
    p.add_argument(
        "--find-equiv",
        type=Path,
        default=REPO_BIN / "find_equivalent_mols.py",
        help="Path to find_equivalent_mols.py",
    )
    p.add_argument(
        "--rules",
        type=Path,
        default=Path(__file__).resolve().parent / "cho_equivalence_processed.toml",
        help="CHO equivalence TOML",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for find_equivalent_mols (0 = all CPU cores)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress in find_equivalent_mols",
    )
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Only build the master molecules CSV (skip equivalence)",
    )
    p.add_argument(
        "--equiv-only",
        action="store_true",
        help="Only run equivalence on an existing master molecules CSV",
    )
    p.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print scrape plan without writing files or running equivalence",
    )
    p.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Reuse master molecules CSV and equiv_master_scrape_manifest.csv; "
            "only scrape graphs not yet in the manifest"
        ),
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Rescrape every GraphML (ignore manifest); default if no prior outputs",
    )
    p.add_argument(
        "--force-equiv",
        action="store_true",
        help=(
            "Reclassify the full master list (ignore existing equivalence CSV). "
            "Default: only new SMILES vs the prior map; skip if there are none"
        ),
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Only cross-check master molecules CSV files and exit "
            "(no scrape or equivalence)"
        ),
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-run consistency check of master / equivalent / unique CSVs",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.collect_only and args.equiv_only:
        print("Use at most one of --collect-only and --equiv-only", file=sys.stderr)
        return 1
    if args.validate_only and (args.collect_only or args.equiv_only or args.dry_run):
        print("--validate-only cannot be combined with collect/equiv/dry-run flags", file=sys.stderr)
        return 1

    root = args.root.resolve()
    find_equiv = args.find_equiv.resolve()
    rules = args.rules.resolve()
    stem = args.master_stem

    mol_csv = root / f"{stem}.csv"
    equiv_csv = root / f"{stem}_equivalent.csv"
    unique_csv = root / f"{stem}_unique.csv"
    manifest_csv = root / MANIFEST_NAME

    if args.validate_only:
        return validate_master_outputs(mol_csv, equiv_csv, unique_csv)

    run_equiv = not args.collect_only
    run_collect = not args.equiv_only

    if args.full and args.incremental:
        print("Use at most one of --incremental and --full", file=sys.stderr)
        return 1
    incremental = args.incremental
    if run_collect and not args.full and not args.incremental:
        if mol_csv.is_file() or manifest_csv.is_file():
            incremental = True

    if run_equiv:
        if not find_equiv.is_file():
            print(f"find_equivalent_mols.py not found: {find_equiv}", file=sys.stderr)
            return 1
        if not rules.is_file():
            print(f"Rules TOML not found: {rules}", file=sys.stderr)
            return 1

    only = set(args.folder) if args.folder else None
    folders = discover_folders(root, only) if run_collect else []
    if run_collect and not folders:
        print(f"No stoichiometry folders with GraphML under {root}", file=sys.stderr)
        return 1
    if args.equiv_only and not mol_csv.is_file():
        print(f"Master molecules CSV not found: {mol_csv}", file=sys.stderr)
        return 1

    prior_master: set[str] = set()
    prior_manifest: dict[tuple[str, str], dict[str, object]] = {}
    if run_collect and incremental:
        if mol_csv.is_file():
            prior_master = load_molecules_csv(mol_csv)
        if manifest_csv.is_file():
            prior_manifest = load_scrape_manifest(manifest_csv)

    print(f"Project root: {root}")
    if run_collect:
        print(f"folders with GraphML: {len(folders)}")
        mode = "incremental" if incremental else "full"
        print(f"collect mode: {mode}")
    print(f"master molecules: {mol_csv}")
    if run_collect:
        print(f"scrape manifest: {manifest_csv}")
    if run_equiv:
        print(f"find_equivalent_mols: {find_equiv}")
        print(f"rules: {rules}")
        print(f"jobs: {args.jobs}")
    print()

    n_new_smiles = 0
    if run_collect:
        jobs = iter_graph_jobs(folders)
        pending, _stale = print_incremental_plan(
            jobs, prior_manifest, prior_master, incremental
        )
        if args.dry_run:
            print(
                f"\nDRY RUN: would write {mol_csv} "
                f"(starting from {len(prior_master)} SMILES, "
                f"scrape {len(pending)} graph(s))",
                flush=True,
            )
        else:
            print("\nCollecting master molecule list from GraphML...", flush=True)
            master, manifest_rows, n_new_smiles = collect_master_molecules(
                folders,
                master=prior_master if incremental else None,
                prior_manifest=prior_manifest if incremental else None,
                incremental=incremental,
            )
            write_molecules_csv(master, mol_csv)
            manifest = write_scrape_manifest(root, manifest_rows)
            print(
                f"\nWrote {mol_csv} ({len(master)} unique SMILES, "
                f"+{n_new_smiles} new this run)",
                flush=True,
            )
            print(f"Scrape manifest: {manifest}", flush=True)
            if incremental and n_new_smiles == 0:
                print("No new SMILES added to master list.", flush=True)

    if run_equiv:
        if args.dry_run:
            print(f"DRY RUN: would run find_equivalent_mols on {mol_csv}", flush=True)
            return 0
        if not mol_csv.is_file():
            print(f"Master molecules CSV not found: {mol_csv}", file=sys.stderr)
            return 1
        n_new_for_equiv = (
            n_new_smiles
            if run_collect and n_new_smiles > 0
            else count_new_molecules_for_equiv(mol_csv, equiv_csv)
        )
        if not args.force_equiv and equiv_csv.is_file() and n_new_for_equiv == 0:
            print(
                "\nSkipping equivalence (no new SMILES vs prior map; "
                "use --force-equiv to reclassify all)",
                flush=True,
            )
            if not args.no_validate:
                rc = validate_master_outputs(mol_csv, equiv_csv, unique_csv)
                if rc != 0:
                    return rc
            print("\nDone.")
            return 0

        if args.force_equiv or not equiv_csv.is_file():
            print(
                f"\nRunning equivalence on full master list "
                f"({len(load_molecules_csv(mol_csv))} SMILES)...",
                flush=True,
            )
        else:
            print(
                f"\nRunning incremental equivalence "
                f"({n_new_for_equiv} new SMILES; prior map {equiv_csv.name})...",
                flush=True,
            )
        try:
            run_find_equivalent(
                find_equiv,
                rules,
                mol_csv,
                equiv_csv,
                unique_csv,
                args.jobs,
                args.no_progress,
                full=args.force_equiv,
            )
        except subprocess.CalledProcessError as e:
            print(f"find_equivalent_mols failed: {e}", file=sys.stderr)
            return 1
        print(f"Wrote {equiv_csv}", flush=True)
        print(f"Wrote {unique_csv}", flush=True)

    if not args.no_validate and mol_csv.is_file():
        if equiv_csv.is_file() and unique_csv.is_file():
            rc = validate_master_outputs(mol_csv, equiv_csv, unique_csv)
            if rc != 0:
                return rc
        elif not args.collect_only:
            print(
                f"\nValidation skipped: missing {equiv_csv.name} or {unique_csv.name}",
                flush=True,
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

