#!/usr/bin/env python3
"""
Merge computed energies with equivalence-class mappings.

Outputs one row per real SMILES with the lowest energy among equivalent forms.
Derives [H+] energy from [OH3+] and O when those species are present.

Authors:
    Miko Stulajter

Version 2.0.0

Usage:
    merge_energies.py ENERGIES.csv EQUIVALENCE.csv OUTPUT.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from rdkit import RDLogger

from graphml_utils import canonicalize_multimol_smiles

RDLogger.DisableLog("rdApp.warning")

PROTON_SMILES = "[H+]"
WATER_SMILES = "O"
HYDROXONIUM_SMILES = "[OH3+]"


def load_equivalence_mapping(
    equiv_file: Path,
) -> tuple[dict[str, str | list[str]], set[str], dict[str, set[str]]]:
    real_to_equiv: dict[str, str | list[str]] = {}
    all_real_smiles: set[str] = set()
    smiles_to_all_equivs: dict[str, set[str]] = {}

    with equiv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            real_smiles = row["smiles"].strip()
            equiv_smiles_str = row["equivalent"].strip()

            if "|" in equiv_smiles_str:
                equiv_smiles: str | list[str] = [
                    s.strip() for s in equiv_smiles_str.split("|") if s.strip()
                ]
            else:
                equiv_smiles = equiv_smiles_str

            real_to_equiv[real_smiles] = equiv_smiles
            all_real_smiles.add(real_smiles)

            equiv_set = {real_smiles}
            if isinstance(equiv_smiles, list):
                equiv_set.update(equiv_smiles)
            elif equiv_smiles:
                equiv_set.add(equiv_smiles)

            for smiles in list(equiv_set):
                canonical = canonicalize_multimol_smiles(smiles)
                if canonical:
                    equiv_set.add(canonical)

            for smiles in equiv_set:
                smiles_to_all_equivs.setdefault(smiles, set()).update(equiv_set)

    return real_to_equiv, all_real_smiles, smiles_to_all_equivs


def load_energy_data(
    energy_file: Path,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    energy_data: dict[str, str] = {}
    canonical_to_originals: dict[str, list[str]] = {}

    with energy_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = row["smiles"].strip()
            energy = row.get("energy", "").strip()
            energy_data[smiles] = energy

            canonical = canonicalize_multimol_smiles(smiles)
            if canonical:
                canonical_to_originals.setdefault(canonical, []).append(smiles)

    return energy_data, canonical_to_originals


def normalize_equiv_forms(equiv_smiles: str | list[str]) -> tuple[str, ...]:
    if isinstance(equiv_smiles, list):
        return tuple(sorted(equiv_smiles))
    if equiv_smiles:
        return (equiv_smiles,)
    return tuple()


def get_best_energy_from_forms(
    forms: list[str],
    energy_data: dict[str, str],
    smiles_to_all_equivs: dict[str, set[str]] | None = None,
    canonical_to_originals: dict[str, list[str]] | None = None,
) -> str | None:
    best_energy: str | None = None
    best_energy_val = float("inf")

    all_forms_to_check = set(forms)
    if smiles_to_all_equivs:
        for form in forms:
            if form in smiles_to_all_equivs:
                all_forms_to_check.update(smiles_to_all_equivs[form])

    for form in list(all_forms_to_check):
        canonical = canonicalize_multimol_smiles(form)
        if canonical:
            all_forms_to_check.add(canonical)
            if canonical_to_originals and canonical in canonical_to_originals:
                all_forms_to_check.update(canonical_to_originals[canonical])

    for form in all_forms_to_check:
        if form not in energy_data or not energy_data[form]:
            continue
        try:
            energy_val = float(energy_data[form])
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid energy for {form}: {energy_data[form]}") from err
        if energy_val < best_energy_val:
            best_energy_val = energy_val
            best_energy = energy_data[form]

    return best_energy


def merge_energies(energy_file: Path, equiv_file: Path, output_file: Path) -> None:
    real_to_equiv, all_real_smiles, smiles_to_all_equivs = load_equivalence_mapping(
        equiv_file
    )
    energy_data, canonical_to_originals = load_energy_data(energy_file)

    equiv_groups: dict[tuple[str, ...], list[str]] = {}
    for real_smiles in all_real_smiles:
        normalized = normalize_equiv_forms(real_to_equiv.get(real_smiles, ""))
        equiv_groups.setdefault(normalized, []).append(real_smiles)

    group_energies: dict[tuple[str, ...], str] = {}
    for normalized_equiv, real_smiles_list in equiv_groups.items():
        all_forms = list(normalized_equiv)
        for real_smiles in real_smiles_list:
            if real_smiles in energy_data and energy_data[real_smiles]:
                all_forms.append(real_smiles)
        best = get_best_energy_from_forms(
            all_forms,
            energy_data,
            smiles_to_all_equivs,
            canonical_to_originals,
        )
        group_energies[normalized_equiv] = best or ""

    results = [
        {
            "smiles": real_smiles,
            "energy": group_energies.get(
                normalize_equiv_forms(real_to_equiv.get(real_smiles, "")), ""
            ),
        }
        for real_smiles in sorted(all_real_smiles)
    ]

    h_plus_idx = next(
        (i for i, row in enumerate(results) if row["smiles"] == PROTON_SMILES),
        None,
    )
    if h_plus_idx is not None:
        o_energy = oh3_energy = None
        for row in results:
            if not row["energy"]:
                continue
            try:
                value = float(row["energy"])
            except (ValueError, TypeError):
                continue
            if row["smiles"] == WATER_SMILES:
                o_energy = value
            elif row["smiles"] == HYDROXONIUM_SMILES:
                oh3_energy = value
        if o_energy is not None and oh3_energy is not None:
            results[h_plus_idx]["energy"] = str(oh3_energy - o_energy)

    results_with_energy = [r for r in results if r["energy"].strip()]
    missing_count = len(results) - len(results_with_energy)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles", "energy"])
        writer.writeheader()
        writer.writerows(results_with_energy)

    print(f"Processed {len(results)} molecules")
    print(f"Missing energy: {missing_count} molecules")
    print(f"Wrote {len(results_with_energy)} molecules to {output_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge energies with equivalence mappings (real SMILES in output).",
    )
    parser.add_argument("energy_file", type=Path, help="CSV: smiles, energy")
    parser.add_argument("equivalence_file", type=Path, help="CSV: smiles, equivalent")
    parser.add_argument("output_file", type=Path, help="Output CSV: smiles, energy")

    args = parser.parse_args()

    for path, label in (
        (args.energy_file, "Energy file"),
        (args.equivalence_file, "Equivalence file"),
    ):
        if not path.is_file():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 1

    merge_energies(args.energy_file, args.equivalence_file, args.output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
