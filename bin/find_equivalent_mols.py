#!/usr/bin/env python3
"""
Find equivalence classes for molecules using bond opening/closing rules.

Authors:
    Miko Stulajter

Version 2.0.0

Usage:
    find_equivalent_mols.py -i INPUT.csv -o OUTPUT.csv -r RULES.toml \\
        [-c smiles] [-d DEPTH] [-j JOBS] [-u UNIQUE.csv] [--no-progress]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import toml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from openbabel import pybel

# Charge patterns for detecting charged atoms in SMILES
_charge_patterns = {
    '[*+1]': Counter([1]),
    '[*+2]': Counter([2]),
    '[*+3]': Counter([3]),
    '[*-1]': Counter([-1]),
    '[*-2]': Counter([-2]),
    '[*-3]': Counter([-3])
}

# Charge exceptions (charge-separated pairs that should be treated differently)
_charge_exceptions = {'[C-1]#[O+1]': Counter([1, -1])}


class Charge:
    """
    Matching facility for charged and charge-separated compounds using SMILES strings.
    """

    def __init__(self):
        """
        Initialize charge patterns and exceptions with compiled SMARTS patterns.
        """
        self._patterns = _charge_patterns
        # Compile pybel representations of SMARTS patterns for efficient matching
        self._pysmarts = {s: pybel.Smarts(s) for s in self._patterns}
        self._exceptions = _charge_exceptions
        # Compile pybel representations of exception SMARTS patterns
        self._pyexceptions = {s: pybel.Smarts(s) for s in self._exceptions}

    def count(self, smiles):
        """
        Count numbers of charged atoms, removing charge exceptions.
        
        Args:
            smiles: SMILES string (may contain multiple molecules separated by '.')
            
        Returns:
            Counter object with net charges
        """
        charges = Counter()
        for s in smiles.split('.'):
            pymol = pybel.readstring('smi', s)
            # Count charged atoms using patterns
            for p, c in self._patterns.items():
                matches = len(self._pysmarts[p].findall(pymol))
                for m in range(matches):
                    charges += c
            # Remove charge exceptions (charge-separated pairs)
            for e, c in self._exceptions.items():
                matches = len(self._pyexceptions[e].findall(pymol))
                for m in range(matches):
                    charges -= c
        return charges

    def abs_charges(self, smiles):
        """
        Compute sum of absolute charges in SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Sum of absolute values of all charges
        """
        charges = self.count(smiles)
        return sum(abs(c) for c in charges.elements())


class Bonds:
    """Bond count (including order) from SMILES via Open Babel."""

    def num_bonds(self, smiles):
        """
        Return total number of bonds (including bond order/multiplicity).
        
        Args:
            smiles: SMILES string (may contain multiple molecules separated by '.')
            
        Returns:
            Total number of bonds (sum of bond orders)
        """
        nb = 0
        for s in smiles.split('.'):
            pymol = pybel.readstring('smi', s)
            pymol.addh()
            # Sum bond orders for all bonds
            for b in range(pymol.OBMol.NumBonds()):
                nb += pymol.OBMol.GetBondById(b).GetBondOrder()
        return nb


def abs_charges(smiles):
    """
    Return sum of absolute charges in SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Sum of absolute values of all charges
    """
    ch = Charge()
    return ch.abs_charges(str(smiles))


def num_bonds(smiles):
    """
    Return number of bonds (including multiplicity) from SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Total number of bonds (sum of bond orders)
    """
    bonds = Bonds()
    return bonds.num_bonds(str(smiles))


def stoichiometry_key(smiles):
    """
    Elemental composition signature (Hill formula) for a SMILES string.

    Equivalence via atom-conserving rules cannot change this; mismatches are skipped.
    Dot-disconnected SMILES are handled as one stoichiometry (whole system).

    Returns:
        Canonical formula string, or None if the structure cannot be parsed.
    """
    mol = _mol_from_smiles_for_reaction(smiles)
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def _mol_from_smiles_for_reaction(smiles):
    """
    Build a sanitized RDKit mol for reaction matching (RDKit 2025.x-safe).
    Unsanitized mols can trigger invariant failures inside RunReactants.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _reaction_from_smarts(reaction_smarts):
    """Parse reaction SMARTS and initialize templates for substructure matching."""
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    if rxn is None:
        return None
    try:
        rxn.Initialize()
    except Exception:
        return None
    try:
        AllChem.SanitizeRxn(rxn)
    except Exception:
        pass
    return rxn


def apply_reaction(smiles, reaction_smarts, forward=True):
    """
    Apply a reaction SMARTS pattern to a SMILES string.
    
    Args:
        smiles: Input SMILES string
        reaction_smarts: Reaction SMARTS pattern (format: reactant>>product)
        forward: If True, apply forward reaction; if False, apply reverse
        
    Returns:
        List of product SMILES strings (canonical, unique)
    """
    try:
        if not forward:
            parts = reaction_smarts.split('>>')
            if len(parts) == 2:
                reaction_smarts = '>>'.join([parts[1], parts[0]])
            else:
                return []

        rxn = _reaction_from_smarts(reaction_smarts)
        if rxn is None:
            return []

        mol = _mol_from_smiles_for_reaction(smiles)
        if mol is None:
            return []

        RDLogger.DisableLog('rdApp.error')
        RDLogger.DisableLog('rdApp.warning')
        try:
            products = rxn.RunReactants((mol,))
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return []
        finally:
            RDLogger.EnableLog('rdApp.error')
            RDLogger.EnableLog('rdApp.warning')

        result = []
        for product_tuple in products:
            for product in product_tuple:
                if product is None:
                    continue
                try:
                    Chem.SanitizeMol(product)
                    product_smiles = Chem.MolToSmiles(product, canonical=True)
                    if product_smiles:
                        result.append(product_smiles)
                except Exception:
                    try:
                        product_smiles = Chem.MolToSmiles(product, canonical=True)
                        if product_smiles:
                            result.append(product_smiles)
                    except Exception:
                        pass

        return list(set(result))
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return []


def find_equivalent_molecules(start_smiles, rules, zero_energy_types, max_depth=10):
    """
    Find all molecules equivalent to start_smiles via zero-energy transformations.
    
    Args:
        start_smiles: Starting SMILES string
        rules: Dictionary mapping reaction SMARTS to rule types
        zero_energy_types: List of rule types considered zero-energy (e.g., 'bond opening', 'bond closing')
        max_depth: Maximum search depth (default: 10)
        
    Returns:
        Set of equivalent SMILES strings
    """
    seen = set()
    queue = deque([(start_smiles, 0)])
    seen.add(start_smiles)
    equivalent = set([start_smiles])
    target_stoich = stoichiometry_key(start_smiles)

    # Filter to only zero-energy rules
    zero_energy_rules = {
        smarts: rule_type for smarts, rule_type in rules.items()
        if rule_type in zero_energy_types
    }

    # Breadth-first search through equivalent molecules
    while queue:
        current_smiles, depth = queue.popleft()

        if depth >= max_depth:
            continue

        # Try all zero-energy transformations
        for reaction_smarts, rule_type in zero_energy_rules.items():
            # Try both forward and reverse reactions
            for forward in [True, False]:
                products = apply_reaction(current_smiles, reaction_smarts, forward=forward)
                for product in products:
                    if product in seen:
                        continue
                    if target_stoich is not None:
                        if stoichiometry_key(product) != target_stoich:
                            seen.add(product)
                            continue
                    seen.add(product)
                    queue.append((product, depth + 1))
                    equivalent.add(product)

    return equivalent


def load_rules_from_yaml(path):
    """
    Load rules dict and zero-energy type list from a YAML file (ACSESS-style).

    Expected: top-level ``rules:`` mapping SMARTS -> type, optional ``zero_energy_types:`` list.
    """
    try:
        import yaml
    except ImportError as err:
        raise SystemExit(
            'YAML rules require PyYAML (pip install pyyaml). '
            'TOML rules do not need it.'
        ) from err
    with open(path, 'r', encoding='utf-8', errors='replace') as fd:
        rules_data = yaml.safe_load(fd)
    if not isinstance(rules_data, dict):
        raise SystemExit(f'YAML rules must be a mapping at root: {path}')
    rules = rules_data.get('rules', {})
    if not isinstance(rules, dict):
        rules = {}
    rules = {k.strip(): v for k, v in rules.items() if isinstance(k, str) and isinstance(v, str)}
    zero_cfg = rules_data.get('zero_energy_types')
    if isinstance(zero_cfg, list):
        zero_energy_types = [str(x) for x in zero_cfg]
    else:
        zero_energy_types = ['bond opening', 'bond closing']
    return rules, zero_energy_types


def load_rules_from_toml(rules_data):
    """
    Flatten equivalence-rule TOML into reaction SMARTS -> family name (table header).

    Expected layout: one table per family ([unimolecular], [cyclization], ...), each
    mapping rule_* keys to reaction SMARTS strings. All such rules are zero-energy by
    convention. Only top-level values that are tables (dicts) are read; other root keys
    """
    rules = {}
    for section, entries in rules_data.items():
        if not isinstance(entries, dict):
            continue
        for _key, smarts in entries.items():
            if isinstance(smarts, str) and smarts.strip():
                rules[smarts.strip()] = section
    return rules


def load_rules_and_zero_energy_types(rules_path: str | Path):
    """
    Load equivalence rules from ``.toml`` / ``.tml`` (CHO-style sections) or ``.yml`` / ``.yaml``.
    """
    rules_path = Path(rules_path).resolve()
    if not rules_path.is_file():
        raise SystemExit(f"Rules file not found: {rules_path}")

    lower = str(rules_path).lower()
    if lower.endswith(('.yml', '.yaml')):
        return load_rules_from_yaml(rules_path)

    with rules_path.open(encoding="utf-8", errors="replace") as fd:
        rules_text = fd.read()
    try:
        rules_data = toml.loads(rules_text)
    except toml.TomlDecodeError as err:
        raise SystemExit(
            f'Could not parse rules as TOML ({rules_path}): {err}. '
            f'Expected CHO-style sections or use a .yml with a top-level rules: mapping.'
        ) from err
    rules = load_rules_from_toml(rules_data)
    zero_energy_cfg = rules_data.get('zero_energy_types')
    if isinstance(zero_energy_cfg, list):
        zero_energy_types = list(zero_energy_cfg)
    else:
        zero_energy_types = sorted(set(rules.values()))
    return rules, zero_energy_types


def _pick_representative(all_equivalent_set, molecules_set, fallback_smiles):
    """Choose class representative from an equivalence set (bonds, charge, input, name)."""
    if not all_equivalent_set:
        return fallback_smiles

    representatives = []
    for s in all_equivalent_set:
        try:
            charge = abs_charges(s)
            bonds = num_bonds(s)
            in_input = 1 if s in molecules_set else 0
            representatives.append((s, bonds, charge, in_input))
        except Exception:
            in_input = 1 if s in molecules_set else 0
            representatives.append((s, 0, float('inf'), in_input))

    representatives.sort(key=lambda x: (-x[1], x[2], -x[3], x[0]))

    best_bonds = representatives[0][1]
    best_charge = representatives[0][2]
    best_in_input = representatives[0][3]

    tied_representatives = [
        r[0] for r in representatives
        if r[1] == best_bonds and r[2] == best_charge and r[3] == best_in_input
    ]

    if len(tied_representatives) > 1:
        return tied_representatives
    return tied_representatives[0]


def _explore_from_seed(seed, rules, zero_energy_types, max_depth):
    """Worker: BFS equivalence set from one input molecule (picklable top-level)."""
    all_equivalent_set = find_equivalent_molecules(seed, rules, zero_energy_types, max_depth)
    all_equivalent_set.add(seed)
    return all_equivalent_set


def _assign_equivalence_class(seed, all_equivalent_set, molecules_set, processed, molecule_to_class):
    """
    Record one equivalence class; return input molecules newly classified (0 if duplicate work).
    """
    equivalent_set = {s for s in all_equivalent_set if s in molecules_set}
    if equivalent_set & processed:
        return 0
    processed.update(equivalent_set)
    representative = _pick_representative(all_equivalent_set, molecules_set, seed)
    for s in equivalent_set:
        molecule_to_class[s] = representative
    return len(equivalent_set)


def _resolve_jobs(jobs):
    """0 or negative means use all available CPU cores."""
    if jobs is None or jobs <= 0:
        return os.cpu_count() or 1
    return jobs


class _FallbackProgress:
    """Minimal progress display when tqdm is not installed."""

    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.n = 0

    def update(self, n=1):
        self.n += n
        pct = 100.0 * self.n / self.total if self.total else 100.0
        print(f'\r{self.desc}: {self.n}/{self.total} ({pct:.0f}%)', end='', file=sys.stderr, flush=True)

    def close(self):
        if self.total:
            print(file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _progress_bar(total, desc, disable=False):
    if disable or total == 0:
        return _NullProgress()
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit='mol', file=sys.stderr)
    except ImportError:
        return _FallbackProgress(total, desc)


class _NullProgress:
    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


_PROGRESS_INTERVAL = 10000


class _ProgressTracker:
    """tqdm bar (stderr) plus a stdout line every N molecules classified."""

    def __init__(self, total, desc, disable=False, interval=_PROGRESS_INTERVAL):
        self.total = total
        self.desc = desc
        self.interval = interval
        self.n = 0
        self._next_report = interval
        self._disable = disable
        self._bar = _progress_bar(total, desc, disable=disable)

    def update(self, n=1):
        if n <= 0:
            return
        self.n += n
        self._bar.update(n)
        if self._disable or self.interval <= 0:
            return
        while self._next_report <= self.total and self.n >= self._next_report:
            pct = 100.0 * self.n / self.total
            print(
                f'{self.desc}: {self.n}/{self.total} ({pct:.1f}%)',
                flush=True,
            )
            self._next_report += self.interval

    def close(self):
        self._bar.close()

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, *args):
        # Final line when the run does not end on an exact interval milestone.
        last_reported = self._next_report - self.interval
        if not self._disable and self.n and self.n != last_reported:
            pct = 100.0 * self.n / self.total if self.total else 100.0
            print(
                f'{self.desc}: {self.n}/{self.total} ({pct:.1f}%)',
                flush=True,
            )
        return self._bar.__exit__(*args)


def find_equivalence_classes(molecules, rules, zero_energy_types, max_depth=10, jobs=1, progress=True):
    """
    Find equivalence classes for a list of molecules.
    
    Groups molecules into equivalence classes based on zero-energy transformations.
    Selects a representative for each class based on:
    - Number of bonds (prefer more bonds)
    - Absolute charge (prefer lower charge)
    - Presence in input set (prefer molecules from input)
    - Alphabetical order (for tie-breaking)
    
    Args:
        molecules: List of SMILES strings
        rules: Dictionary mapping reaction SMARTS to rule types
        zero_energy_types: List of rule types considered zero-energy
        max_depth: Maximum search depth for finding equivalents
        jobs: Worker processes; 1 = serial, 0 = all CPU cores
        progress: Show molecule-classification progress bar (default True)
        
    Returns:
        Dictionary mapping each input molecule to its equivalence class representative(s)
    """
    molecules_set = set(molecules)
    unique_molecules = list(dict.fromkeys(molecules))
    n_workers = _resolve_jobs(jobs)

    molecule_to_class = {}
    processed = set()
    unprocessed = set(unique_molecules)

    with _ProgressTracker(len(unique_molecules), 'Equivalence search', disable=not progress) as pbar:
        if n_workers == 1:
            for seed in unique_molecules:
                if seed in processed:
                    continue
                all_equivalent_set = _explore_from_seed(seed, rules, zero_energy_types, max_depth)
                n_new = _assign_equivalence_class(
                    seed, all_equivalent_set, molecules_set, processed, molecule_to_class
                )
                pbar.update(n_new)
        else:
            n_workers = min(n_workers, len(unique_molecules))
            inflight = {}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                while unprocessed or inflight:
                    while len(inflight) < n_workers and unprocessed:
                        seed = unprocessed.pop()
                        if seed in processed:
                            continue
                        fut = executor.submit(
                            _explore_from_seed, seed, rules, zero_energy_types, max_depth
                        )
                        inflight[fut] = seed

                    if not inflight:
                        break

                    done, _pending = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        seed = inflight.pop(fut)
                        try:
                            all_equivalent_set = fut.result()
                        except Exception:
                            continue
                        equiv_in_input = {s for s in all_equivalent_set if s in molecules_set}
                        unprocessed -= equiv_in_input
                        n_new = _assign_equivalence_class(
                            seed, all_equivalent_set, molecules_set, processed, molecule_to_class
                        )
                        if n_new:
                            pbar.update(n_new)

    for smiles in molecules:
        if smiles not in molecule_to_class:
            molecule_to_class[smiles] = smiles

    return molecule_to_class


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find equivalence classes via zero-energy bond opening/closing rules.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input CSV with molecules",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output mapping CSV (smiles, equivalent)",
    )
    parser.add_argument(
        "-r",
        "--rules",
        type=Path,
        required=True,
        help="Rules file (.toml CHO-style or .yml with rules: mapping)",
    )
    parser.add_argument(
        '-c',
        '--column',
        default='smiles',
        help='Column name containing SMILES (default: smiles)'
    )
    parser.add_argument(
        '-d',
        '--max-depth',
        type=int,
        default=10,
        help='Maximum depth for equivalence search (default: 10)'
    )
    parser.add_argument(
        '-u',
        '--unique',
        default=None,
        help='Also write unique molecules (equivalence class representatives) to this file'
    )
    parser.add_argument(
        '-j',
        '--jobs',
        type=int,
        default=1,
        metavar='N',
        help='Parallel workers, one per unclassified molecule (default: 1). Use 0 for all CPU cores.'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable molecule-classification progress bar'
    )
    
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1
    if not args.rules.is_file():
        print(f"Rules file not found: {args.rules}", file=sys.stderr)
        return 1

    rules, zero_energy_types = load_rules_and_zero_energy_types(args.rules)

    molecules: list[str] = []
    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or args.column not in reader.fieldnames:
            print(f"Column '{args.column}' not found in {args.input}", file=sys.stderr)
            return 1
        for row in reader:
            smiles = row[args.column].strip()
            if smiles:
                molecules.append(smiles)
    
    print(f'Loaded {len(molecules)} molecules')
    n_active = sum(1 for t in rules.values() if t in zero_energy_types)
    print(
        f'Using {n_active} zero-energy rules '
        f'({len(zero_energy_types)} famil{"y" if len(zero_energy_types) == 1 else "ies"})'
    )
    
    n_jobs = _resolve_jobs(args.jobs)
    if n_jobs > 1:
        print(f'Using up to {n_jobs} parallel worker(s) (one molecule per worker)')
    
    # Find equivalence classes
    equiv_map = find_equivalence_classes(
        molecules,
        rules,
        zero_energy_types,
        args.max_depth,
        jobs=args.jobs,
        progress=not args.no_progress,
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([args.column, "equivalent"])
        for smiles in molecules:
            equivalent = equiv_map.get(smiles, smiles)
            if isinstance(equivalent, list):
                equivalent_str = "|".join(sorted(equivalent))
            else:
                equivalent_str = equivalent
            writer.writerow([smiles, equivalent_str])
    print(f"Mapping file written to {args.output}")

    if args.unique:
        unique_representatives: set[str] = set()
        for representative in equiv_map.values():
            if isinstance(representative, list):
                unique_representatives.update(representative)
            else:
                unique_representatives.add(representative)
        Path(args.unique).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.unique).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["smiles"])
            for representative in sorted(unique_representatives):
                writer.writerow([representative])
        print(f"Wrote {len(unique_representatives)} unique molecules to {args.unique}")

    unique_equivalents = len(
        {
            tuple(sorted(rep)) if isinstance(rep, list) else rep
            for rep in equiv_map.values()
        }
    )
    print(f"Found {unique_equivalents} unique equivalence classes")
    return 0


if __name__ == "__main__":
    sys.exit(main())


