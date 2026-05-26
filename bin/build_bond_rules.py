#!/usr/bin/env python3
"""Build bond transformation rules from component specification file."""

import argparse
import os
import sys
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

_RULEGEN_DIR = Path(__file__).resolve().parent
if str(_RULEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_RULEGEN_DIR))

# SMILES/RDKit bond orders: single through triple only (no quadruple in this builder).
# [MAX_BOND] integers are max bond count per component (e.g. 4 for C) — not bond order;
# valence 4 with max bond order 3 covers cases such as triple bond plus a radical.
BOND_SYMBOLS = {0: '.', 1: '-', 2: '=', 3: '#'}
# Shorthand: bond symbol -> S/D/T/Z (single, double, triple, zero/none)
BOND_TO_SHORT = {'-': 'S', '=': 'D', '#': 'T', '.': 'Z'}
# Allowed bond-order changes in one rule: only adjacent steps (#<->=, =<->-, -<->.).
MAX_BOND_ORDER_STEP = 1

# [params].radicals_change_limit: max |reactant − product| total in |^...| slots.
DEFAULT_RADICALS_CHANGE_LIMIT = 2
# [params].charge_change_limit: max |Σ formal charge(reactants) − Σ formal charge(products)|.
DEFAULT_CHARGE_CHANGE_LIMIT = 2
# [params].allow_charge_radical_mix: when false, post-filter drops |^…| that break map
# pairing on charge-redistributing rules ({0,2}, {1,3}, or {0,1,2,3} per clause).
DEFAULT_ALLOW_CHARGE_RADICAL_MIX = False
DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_REORDER = True
DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_CLEAVAGE = False
DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_HYDROGEN = False

# Hot-path patterns (used heavily in rule generation).
_RE_COMP_PAIRED = re.compile(r'\[([A-Za-z]+)([+-]?\d+)(?:;H0)?:\d+\]')
_RE_MAP_IDX = re.compile(r':(\d+)')
_RE_BRACKET_CHARGE_MAP = re.compile(r'\[([A-Za-z]+)([+-]?\d+)(?:;H0)?:(\d+)\]')
# Bracket hydrogen: [H+0:1], [H;H0:1], [#1+0:1], etc.
_RE_BRACKET_HYDROGEN = re.compile(r"\[(?:#1|H)(?:[+\d\-]|;|:|\])")


def _normalize_component(comp: str) -> str:
    """Ensure component has an index: [O+0] -> [O+0:1], [O+0:1] unchanged."""
    comp = comp.strip()
    if not comp.startswith('[') or not comp.endswith(']'):
        return comp
    if re.search(r':\d+\]$', comp):
        return comp
    return comp[:-1] + ':1]'


def _parse_optional_int(value) -> Optional[int]:
    """None or "" or "none" -> None (no limit); else int(value)."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ('', 'none'):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_radicals_change_limit(params: dict) -> int:
    """[params].radicals_change_limit (legacy: change_limit, max_valence_net_change)."""
    if not isinstance(params, dict):
        return DEFAULT_RADICALS_CHANGE_LIMIT
    for key in ('radicals_change_limit', 'change_limit', 'max_valence_net_change'):
        v = _parse_optional_int(params.get(key))
        if v is not None:
            return max(0, v)
    return DEFAULT_RADICALS_CHANGE_LIMIT


def _parse_charge_change_limit(params: dict) -> int:
    """[params].charge_change_limit; 0 requires exact Σ formal charge match."""
    if not isinstance(params, dict):
        return DEFAULT_CHARGE_CHANGE_LIMIT
    v = _parse_optional_int(params.get('charge_change_limit'))
    if v is not None:
        return max(0, v)
    return DEFAULT_CHARGE_CHANGE_LIMIT


def _parse_bool_param(params: dict, key: str, default: bool = False) -> bool:
    """Parse a [params] boolean (true/false, yes/no, 1/0)."""
    if not isinstance(params, dict):
        return default
    v = params.get(key)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('true', '1', 'yes', 'on'):
            return True
        if s in ('false', '0', 'no', 'off'):
            return False
    return default


def _parse_allow_charge_radical_mix(params: dict) -> bool:
    """[params].allow_charge_radical_mix — mixed |^…| on charge-changing steps."""
    return _parse_bool_param(
        params, 'allow_charge_radical_mix', DEFAULT_ALLOW_CHARGE_RADICAL_MIX
    )


def _parse_allow_multi_radical_per_slot_flags(params: dict) -> Tuple[bool, bool, bool]:
    """Return (reorder, cleavage, hydrogen) allow_multi_radical_per_slot flags."""
    if not isinstance(params, dict):
        return (
            DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_REORDER,
            DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_CLEAVAGE,
            DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_HYDROGEN,
        )
    return (
        _parse_bool_param(
            params,
            "allow_multi_radical_per_slot_reorder",
            DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_REORDER,
        ),
        _parse_bool_param(
            params,
            "allow_multi_radical_per_slot_cleavage",
            DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_CLEAVAGE,
        ),
        _parse_bool_param(
            params,
            "allow_multi_radical_per_slot_hydrogen",
            DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_HYDROGEN,
        ),
    )


def _rule_contains_hydrogen(rule: str) -> bool:
    """True if rule SMARTS names a hydrogen centre ([H…] or [#1…])."""
    return bool(_RE_BRACKET_HYDROGEN.search(rule))


def _rule_is_cleavage_transition(rule: str) -> bool:
    """True when the rule is a bonded -↔. step (dot on reactant or product side)."""
    if ">>" not in rule:
        return False
    left, rest = rule.split(">>", 1)
    lb = _bond_order_from_side(left)
    rb = _bond_order_from_side(rest.split("|")[0])
    if lb is None or rb is None:
        return False
    return lb == 0 or rb == 0


def _rule_has_bond_break(rule: str) -> bool:
    """True when the rule contains a dot bond between fragments (].[ in SMARTS)."""
    body = rule.split("|", 1)[0]
    return "].[" in body


def _partition_rules_by_transition(rules: List[str]) -> Tuple[List[str], List[str]]:
    """Split rules into (#-= reorder) vs (-↔. cleavage/association)."""
    reorder_rules: List[str] = []
    cleavage_rules: List[str] = []
    for rule in rules:
        if _rule_is_cleavage_transition(rule):
            cleavage_rules.append(rule)
        else:
            reorder_rules.append(rule)
    return reorder_rules, cleavage_rules


def _valence_higher_radicals_redistributed(valence_str: Optional[str]) -> bool:
    """True when every ^n clause with n>1 lists at least two valence slot indices.

    Allows |^1:0,1;^2:2,3| and |^2:2,3|; rejects |^2:2| and |^1:0,1,2;^2:3| (orphan ^2:3).
    """
    if not valence_str or "^" not in valence_str:
        return True
    for m in re.finditer(r"\^(\d+):([\d,]+)", valence_str):
        n = int(m.group(1))
        if n <= 1:
            continue
        indices = [x.strip() for x in m.group(2).split(",") if x.strip()]
        if len(indices) < 2:
            return False
    return True


def _filter_rules_by_multi_radical_per_slot(
    rules: List[str],
    allow_multi_radical_per_slot: bool,
    *,
    allow_multi_radical_per_slot_hydrogen: bool = DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_HYDROGEN,
) -> List[str]:
    """Post-process: drop |^…| tails that pile ^n>1 on one centre in a single clause."""
    out: List[str] = []
    for r in rules:
        allow = allow_multi_radical_per_slot
        if _rule_contains_hydrogen(r) and not allow_multi_radical_per_slot_hydrogen:
            allow = False
        if allow or _valence_higher_radicals_redistributed(_valence_part_from_rule(r)):
            out.append(r)
    return out


def _rule_has_per_map_charge_redistribution(rule: str) -> bool:
    """True if any shared map number has different formal charge left vs right of >>."""
    if ">>" not in rule:
        return False
    left, right = rule.split(">>", 1)
    right = right.split("|")[0]
    lc_by_m: Dict[int, int] = {}
    rc_by_m: Dict[int, int] = {}
    for m in _RE_BRACKET_CHARGE_MAP.finditer(left):
        q = m.group(2)
        lc_by_m[int(m.group(3))] = int(q.replace("+", "")) if q else 0
    for m in _RE_BRACKET_CHARGE_MAP.finditer(right):
        q = m.group(2)
        rc_by_m[int(m.group(3))] = int(q.replace("+", "")) if q else 0
    for mp in (1, 2):
        if mp in lc_by_m and mp in rc_by_m and lc_by_m[mp] != rc_by_m[mp]:
            return True
    return False


def _valence_clause_ok_for_charge_change(slots: Tuple[int, ...]) -> bool:
    """One ^n:… segment is valid when charge redistributes (map 0↔2, 1↔3 conserved)."""
    if not slots:
        return True
    s = set(slots)
    if s == {0, 1, 2, 3}:
        return True
    if s == {0, 2} or s == {1, 3}:
        return True
    return False


def _valence_ok_when_charge_changes(
    valence_str: Optional[str],
    rule: str,
    allow_charge_radical_mix: bool,
) -> bool:
    """When charge redistributes and mixing is off, each ^ clause must conserve map pairing.

    Allowed per clause: {0,2}, {1,3}, or {0,1,2,3}. So |^1:0,2;^2:1,3| is fine (^1 on
    map :1, ^2 on map :2). Forbidden: |^1:3| (orphan product), |^1:0,1| (reactant-only).
    """
    if allow_charge_radical_mix or not valence_str or "^" not in valence_str:
        return True
    if not _rule_has_per_map_charge_redistribution(rule):
        return True
    for m in re.finditer(r"\^(\d+):([\d,]+)", valence_str):
        slots = tuple(
            int(x) for x in m.group(2).split(",") if x.strip()
        )
        if not _valence_clause_ok_for_charge_change(slots):
            return False
    return True


def _filter_rules_by_charge_radical_mix(
    rules: List[str], allow_charge_radical_mix: bool
) -> List[str]:
    """Post-process: drop charge-redistributing rules whose |^…| breaks map pairing."""
    if allow_charge_radical_mix:
        return rules
    return [
        r
        for r in rules
        if _valence_ok_when_charge_changes(_valence_part_from_rule(r), r, False)
    ]


def _parse_toml_input(data: dict) -> Tuple[
    List[Tuple[str, int]],
    Optional[int],
    Optional[int],
    str,
    Dict[str, int],
    int,
    int,
    bool,
    bool,
    bool,
    bool,
]:
    """Parse TOML: [params], [MAX_BOND], optional [MAX_RADICAL] per-component caps."""
    entries: List[Tuple[str, int]] = []
    limit_radicals: Optional[int] = None
    limit_abs_charges: Optional[int] = None
    name = ""
    radicals_change_limit = DEFAULT_RADICALS_CHANGE_LIMIT
    charge_change_limit = DEFAULT_CHARGE_CHANGE_LIMIT
    allow_charge_radical_mix = DEFAULT_ALLOW_CHARGE_RADICAL_MIX
    allow_multi_radical_per_slot_reorder = DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_REORDER
    allow_multi_radical_per_slot_cleavage = DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_CLEAVAGE
    allow_multi_radical_per_slot_hydrogen = DEFAULT_ALLOW_MULTI_RADICAL_PER_SLOT_HYDROGEN

    params = data.get('params', {})
    if isinstance(params, dict):
        name = str(params.get('name', '')).strip()
        limit_radicals = _parse_optional_int(params.get('limit_radicals'))
        limit_abs_charges = _parse_optional_int(params.get('limit_abs_charges'))
        radicals_change_limit = _parse_radicals_change_limit(params)
        charge_change_limit = _parse_charge_change_limit(params)
        allow_charge_radical_mix = _parse_allow_charge_radical_mix(params)
        (
            allow_multi_radical_per_slot_reorder,
            allow_multi_radical_per_slot_cleavage,
            allow_multi_radical_per_slot_hydrogen,
        ) = _parse_allow_multi_radical_per_slot_flags(params)

    max_bond_by_element = _parse_component_int_table(data.get('MAX_BOND'))
    if not max_bond_by_element:
        raise ValueError('Bond-rules TOML must define [MAX_BOND] with at least one component.')
    for bkey in sorted(max_bond_by_element):
        comp = _normalize_component(f'[{bkey}]')
        entries.append((comp, max_bond_by_element[bkey]))

    max_radical_by_element = _parse_component_int_table(data.get('MAX_RADICAL'))

    return (
        entries,
        limit_radicals,
        limit_abs_charges,
        name,
        max_radical_by_element,
        radicals_change_limit,
        charge_change_limit,
        allow_charge_radical_mix,
        allow_multi_radical_per_slot_reorder,
        allow_multi_radical_per_slot_cleavage,
        allow_multi_radical_per_slot_hydrogen,
    )


def parse_input_file(filepath: str) -> Tuple[
    List[Tuple[str, int]],
    Optional[int],
    Optional[int],
    str,
    Dict[str, int],
    int,
    int,
    bool,
    bool,
    bool,
    bool,
]:
    """Parse bond-rules input from TOML (e.g. bond_rules_input.toml)."""
    try:
        import toml
    except ImportError:
        raise ImportError("TOML input requires the 'toml' package. Install with: pip install toml")
    with open(filepath, 'r') as f:
        data = toml.load(f)
    return _parse_toml_input(data)


def _component_table_key(raw_key: str) -> Optional[str]:
    """Normalize a TOML component key (e.g. \"[O+0]\") to component_base_canonical (e.g. O+0)."""
    sk = str(raw_key).strip()
    if not sk:
        return None
    if not sk.startswith('['):
        sk = f'[{sk}]'
    comp = _normalize_component(sk)
    b = extract_component_base(comp)
    return component_base_canonical(b) if b else None


def _parse_component_int_table(section) -> Dict[str, int]:
    """Parse [MAX_BOND] / [MAX_RADICAL] tables keyed by component brackets."""
    out: Dict[str, int] = {}
    if not isinstance(section, dict):
        return out
    for k, v in section.items():
        bkey = _component_table_key(str(k))
        if not bkey:
            continue
        try:
            out[bkey] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def extract_charge(component: str) -> int:
    match = re.search(r'([+-]?\d+)(?:;H0)?:', component)
    if match:
        charge_str = match.group(1)
        return int(charge_str.replace('+', ''))
    return 0


def format_charge(charge: int) -> str:
    return f"+{charge}" if charge >= 0 else str(charge)


def _charge_to_shorthand(charge: int) -> str:
    """Format charge for shorthand: 0 -> 00, +1 -> 1p, -1 -> 1m, +2 -> 2p (so zeros line up)."""
    if charge == 0:
        return '00'
    if charge > 0:
        return f"{charge}p"
    return f"{abs(charge)}m"


def rule_to_shorthand(rule: str) -> str:
    """Build shorthand: OO__TD__1p1p1p1p__r0011.
    a1a2 = elements. Bonds = left+right (S,D,T,Z). Four charges concatenated. r + 4 digits for radical per atom."""
    if '>>' not in rule:
        return rule
    left, right_rest = rule.split('>>', 1)
    right_parts = right_rest.split('|')
    right_side = right_parts[0]
    valence_part = right_parts[1].strip() if len(right_parts) >= 2 and right_parts[1].strip().startswith('^') else ''

    comps_left = _RE_COMP_PAIRED.findall(left)
    comps_right = _RE_COMP_PAIRED.findall(right_side)
    bond_match_left = re.search(r'\]([-=#.])\[', left)
    bond_match_right = re.search(r'\]([-=#.])\[', right_side)
    bond_left = BOND_TO_SHORT.get(bond_match_left.group(1), 'Z') if bond_match_left else 'Z'
    bond_right = BOND_TO_SHORT.get(bond_match_right.group(1), 'Z') if bond_match_right else 'Z'

    if len(comps_left) < 2 or len(comps_right) < 2:
        return rule

    e1, e2 = comps_left[0][0], comps_left[1][0]
    a1a2 = f"{e1}{e2}"

    c1_left = int(comps_left[0][1].replace('+', '')) if comps_left[0][1] else 0
    c2_left = int(comps_left[1][1].replace('+', '')) if comps_left[1][1] else 0
    c1_right = int(comps_right[0][1].replace('+', '')) if comps_right[0][1] else 0
    c2_right = int(comps_right[1][1].replace('+', '')) if comps_right[1][1] else 0

    charges_block = (
        _charge_to_shorthand(c1_left) + _charge_to_shorthand(c2_left) +
        _charge_to_shorthand(c1_right) + _charge_to_shorthand(c2_right)
    )
    bonds_block = f"{bond_left}{bond_right}"

    # Radical: 4 positions = left0, left1, right0, right1. Each position gets its radical count from ^n:indices.
    radical = [0, 0, 0, 0]
    if valence_part:
        for part in re.findall(r'\^(\d+):([\d,]+)', valence_part):
            n = int(part[0])
            indices = [int(x) for x in part[1].split(',') if x.strip()]
            for idx in indices:
                if 0 <= idx <= 3:
                    radical[idx] = n
    radical_block = 'r' + ''.join(str(r) for r in radical)

    return f"{a1a2}__{bonds_block}__{charges_block}__{radical_block}"


def partner_shorthand_key(shorthand: str) -> str:
    """Shorthand key for the partner rule (other bond-order direction, sides swapped).

    Used by check_rule_pairs; generation emits both transitions via bond_left/bond_right loops.
    """
    parts = shorthand.split('__')
    if len(parts) < 4:
        return shorthand
    a1a2, bonds, charges = parts[0], parts[1], parts[2]
    rpart = '__'.join(parts[3:])
    m = re.match(r'^(r\d{4})(.*)$', rpart)
    if not m:
        return shorthand
    digits, suffix = m.group(1)[1:], m.group(2)
    if len(bonds) == 2:
        bonds = bonds[1] + bonds[0]
    if len(charges) == 8:
        charges = charges[4:8] + charges[0:4]
    if len(digits) == 4:
        rpart = 'r' + digits[2:4] + digits[0:2] + suffix
    return f'{a1a2}__{bonds}__{charges}__{rpart}'


def _join_rhs_and_valence(rhs: str, valence: str) -> str:
    """Append valence notation (|...|) after product SMILES. Space before | when rhs ends with ]."""
    if not valence:
        return rhs
    tail = rhs.rstrip()
    if tail.endswith(']') and valence.startswith('|'):
        return tail + ' ' + valence
    return rhs + valence


def _valence_part_from_radical_vector(radical: Tuple[int, int, int, int]) -> str:
    """Build |^n:i,j| from positional slots 0–3 (reactant0, reactant1, product0, product1)."""
    valence_to_indices: Dict[int, List[int]] = {}
    for pos, n in enumerate(radical):
        if n > 0:
            valence_to_indices.setdefault(n, []).append(pos)
    if not valence_to_indices:
        return ""
    parts = [
        f"{n}:{','.join(map(str, sorted(idxs)))}"
        for n, idxs in sorted(valence_to_indices.items())
    ]
    return "|^" + ";^".join(parts) + "|"


def normalize_valence_indices_in_rule(rule: str) -> str:
    """Sort ^ clauses by n and indices in each ^n:i,j,... segment (|^1:3,2| == |^1:2,3|)."""
    if ">>" not in rule or "|^" not in rule:
        return rule
    left, rest = rule.split(">>", 1)
    try:
        bar = rest.index("|^")
    except ValueError:
        return rule
    rhs = rest[:bar].rstrip()
    vp = rest[bar:].strip()
    if not (vp.startswith("|^") and vp.endswith("|")):
        return rule
    inner = vp[1:-1]
    chunks: List[Tuple[int, str]] = []
    for m in re.finditer(r"\^(\d+):([\d,]+)", inner):
        n = int(m.group(1))
        nums = sorted(int(x.strip()) for x in m.group(2).split(",") if x.strip())
        chunks.append((n, f"{n}:{','.join(map(str, nums))}"))
    if not chunks:
        return rule
    chunks.sort(key=lambda x: x[0])
    new_vp = "|^" + ";^".join(c for _, c in chunks) + "|"
    return f"{left}>>{_join_rhs_and_valence(rhs, new_vp)}"


def extract_component_data(comp: str) -> tuple:
    # Allow optional ;H0 before :digit] (e.g. [O+1;H0:1])
    match = re.search(r'\[([A-Za-z]+)([+-]?\d+)(?:;H0)?:(\d+)\]', comp)
    if match:
        e, c_str, _ = match.groups()
        return (e, int(c_str.replace('+', '')) if c_str else 0)
    return None


def normalize_rule(rule: str) -> str:
    """Assign canonical map numbers :1/:2 to both sides of a rule.

    The left-side pair is sorted ascending by (element, charge) to pick a
    canonical ordering.  The **same permutation** is then applied to the
    right-side pair so that map :N always refers to the same atom on both
    sides of >>.  Valence slot indices (0=reactant0, 1=reactant1,
    2=product0, 3=product1) are swapped accordingly.
    """
    if ">>" not in rule:
        return rule

    left, right = rule.split(">>", 1)
    valence_part = ""
    if "|^" in right:
        idx = right.index("|^")
        valence_part = right[idx:]
        right = right[:idx].rstrip()

    # Parse both sides into ordered component lists.
    def _parse_two(side: str) -> Optional[Tuple[str, str, str]]:
        """Return (comp1, separator, comp2) or None."""
        m_bond = re.match(r"^(\[[^\]]+\])([-=#.])(\[[^\]]+\])$", side.strip())
        if m_bond:
            return m_bond.group(1), m_bond.group(2), m_bond.group(3)
        m_dot = re.match(r"^(\[[^\]]+\])\.(\[[^\]]+\])$", side.strip())
        if m_dot:
            return m_dot.group(1), ".", m_dot.group(2)
        return None

    lp = _parse_two(left)
    rp = _parse_two(right)
    if lp is None or rp is None:
        return rule

    lc1, lsep, lc2 = lp
    rc1, rsep, rc2 = rp

    ld1 = extract_component_data(lc1)
    ld2 = extract_component_data(lc2)
    if ld1 is None or ld2 is None:
        return rule

    # Assign :1/:2 in input order; fragment reorder + valence remap for duplicates is done in
    # _canonicalize_and_filter_rules (maps and |^...| slots stay tied).

    def _rebase(comp: str, map_num: int) -> str:
        base = extract_component_base(comp)
        if base is None:
            return comp
        return f"[{base}:{map_num}]"

    new_left = f"{_rebase(lc1, 1)}{lsep}{_rebase(lc2, 2)}"
    new_right = f"{_rebase(rc1, 1)}{rsep}{_rebase(rc2, 2)}"
    return f"{new_left}>>{_join_rhs_and_valence(new_right, valence_part)}"


# Bracket atoms with map numbers: [Elem+charge:map] or [Elem+charge;H0:map]
# Element is generic (H, O, C, N, Cl, ...), not limited to H/O.
_MAP_ATOM_P1 = re.compile(r"\[([A-Za-z]+)([+-][^:\];]+):(\d+)\]")
_MAP_ATOM_P2 = re.compile(r"\[([A-Za-z]+)([+-][^:\];]+);H0:(\d+)\]")


def fix_reaction_map_consistency(rule: str) -> str:
    """Ensure each map number keeps the same element as on the left side of >>.

    Left map :N defines the element identity (e.g. map 2 is hydrogen). If a product
    bracket uses a different element at that map (O-2 under an H map), it is replaced
    with the left component base (H-1). Formal charge on the product is kept when the
    element already matches (charge redistribution, e.g. O-1 -> O+0).
    """
    if ">>" not in rule:
        return rule
    lhs, rest = rule.split(">>", 1)
    meta = ""
    if "|" in rest:
        rhs, meta = rest.split("|", 1)
        meta = "|" + meta
    else:
        rhs = rest

    maps: Dict[str, str] = {}
    for p in (_MAP_ATOM_P1, _MAP_ATOM_P2):
        for m in p.finditer(lhs):
            mp = m.group(3)
            if mp not in maps:
                maps[mp] = m.group(1) + m.group(2)
    if not maps:
        return rule

    def _left_element(base: str) -> str:
        m = re.match(r"^([A-Za-z]+)", base)
        return m.group(1) if m else base

    def sub1(m):
        el, mid, mp = m.group(1), m.group(2), m.group(3)
        left_base = maps.get(mp)
        if left_base is None or el == _left_element(left_base):
            return m.group(0)
        return f"[{left_base}:{mp}]"

    def sub2(m):
        el, mid, mp = m.group(1), m.group(2), m.group(3)
        left_base = maps.get(mp)
        if left_base is None or el == _left_element(left_base):
            return m.group(0)
        return f"[{left_base};H0:{mp}]"

    rhs2 = _MAP_ATOM_P1.sub(sub1, rhs)
    rhs2 = _MAP_ATOM_P2.sub(sub2, rhs2)
    return lhs + ">>" + _join_rhs_and_valence(rhs2, meta)


def _bracket_map_num(bracket: str) -> Optional[str]:
    m = re.search(r";H0:(\d+)\]$", bracket)
    if m:
        return m.group(1)
    m = re.search(r":(\d+)\]$", bracket)
    return m.group(1) if m else None


def _set_bracket_map_num(bracket: str, map_num: int) -> str:
    if ";H0:" in bracket:
        return re.sub(r";H0:\d+\]", f";H0:{map_num}]", bracket)
    return re.sub(r":\d+\]$", f":{map_num}]", bracket)


def _remap_maps_in_string(s: str, old_to_new: dict) -> str:
    """Replace map numbers in bracket atoms (plain :N and ;H0:N)."""

    def repl_bracket(m):
        b = m.group(0)
        mn = _bracket_map_num(b)
        if mn is None or mn not in old_to_new:
            return b
        new_m = old_to_new[mn]
        return _set_bracket_map_num(b, int(new_m))

    return re.sub(r"\[[^\]]+\]", repl_bracket, s)


def _swap_valence_atom_indices(valence_part: str) -> str:
    """Swap atom indices 0<->1 and 2<->3 in ^n:i,j,... segments (reactant vs product blocks)."""
    if not valence_part or "^" not in valence_part:
        return valence_part

    def swap_idx(x: int) -> int:
        return {0: 1, 1: 0, 2: 3, 3: 2}.get(x, x)

    def repl(m):
        n, idx_str = m.group(1), m.group(2)
        parts = [str(swap_idx(int(x.strip()))) for x in idx_str.split(",") if x.strip()]
        return f"^{n}:{','.join(parts)}"

    return re.sub(r"\^(\d+):([\d,]+)", repl, valence_part)


def _swap_valence_reactant_atom_indices(valence_part: str) -> str:
    """Swap indices 0<->1 only (reactant atoms) in ^n:i,j,... segments."""
    if not valence_part or "^" not in valence_part:
        return valence_part

    def swap_idx(x: int) -> int:
        return {0: 1, 1: 0}.get(x, x)

    def repl(m):
        n, idx_str = m.group(1), m.group(2)
        parts = [str(swap_idx(int(x.strip()))) for x in idx_str.split(",") if x.strip()]
        return f"^{n}:{','.join(parts)}"

    return re.sub(r"\^(\d+):([\d,]+)", repl, valence_part)


def _swap_valence_product_atom_indices(valence_part: str) -> str:
    """Swap indices 2<->3 only (product atoms) in ^n:i,j,... segments."""
    if not valence_part or "^" not in valence_part:
        return valence_part

    def swap_idx(x: int) -> int:
        return {2: 3, 3: 2}.get(x, x)

    def repl(m):
        n, idx_str = m.group(1), m.group(2)
        parts = [str(swap_idx(int(x.strip()))) for x in idx_str.split(",") if x.strip()]
        return f"^{n}:{','.join(parts)}"

    return re.sub(r"\^(\d+):([\d,]+)", repl, valence_part)


def _radical_vector_from_valence_part(valence_part: str) -> Tuple[int, int, int, int]:
    """Four valence slots: reactant maps 0,1 then product maps 2,3 (same convention as rule_to_shorthand)."""
    radical = [0, 0, 0, 0]
    if not valence_part or "^" not in valence_part:
        return tuple(radical)
    for part in re.findall(r"\^(\d+):([\d,]+)", valence_part):
        n = int(part[0])
        for idx in (int(x) for x in part[1].split(",") if x.strip()):
            if 0 <= idx <= 3:
                radical[idx] = n
    return tuple(radical)


def _canonicalize_valence_symmetric_slot_pair(
    valence_part: str, swap_fn
) -> str:
    """Pick one of two symmetric valence placements (swap_fn toggles equivalent slots)."""
    if not valence_part or "^" not in valence_part:
        return valence_part
    swapped = swap_fn(valence_part)
    v0 = _radical_vector_from_valence_part(valence_part)
    v1 = _radical_vector_from_valence_part(swapped)
    return swapped if v1 < v0 else valence_part


def _canonicalize_valence_symmetric_product_bond(valence_part: str) -> str:
    """When product centers are same element and charge, ^…:2 vs ^…:3 are notation duplicates.

    Only safe when there are NO reactant radicals (slots 0 and 1 both zero).
    When reactant slots carry radicals, slots 2 and 3 encode atom correspondence
    (which product atom tracks which reactant atom) and must not be swapped.
    """
    if not valence_part or "^" not in valence_part:
        return valence_part
    v = _radical_vector_from_valence_part(valence_part)
    if v[0] != 0 or v[1] != 0:
        return valence_part
    return _canonicalize_valence_symmetric_slot_pair(
        valence_part, _swap_valence_product_atom_indices
    )


def _canonicalize_valence_symmetric_pair(valence_part: str) -> str:
    """Canonical valence for an identical-component pair (same element and charge on both sides).

    The global symmetry of an identical pair is a simultaneous swap of both atoms:
    (r0, r1, p0, p1) ~ (r1, r0, p1, p0).  We keep whichever tuple is
    lexicographically smaller.  This correctly handles all mixed reactant/product
    cases without conflating distinct rules like |^1:1,2| and |^1:1,3|.
    """
    if not valence_part or "^" not in valence_part:
        return valence_part
    v = _radical_vector_from_valence_part(valence_part)
    r0, r1, p0, p1 = v
    swapped = (r1, r0, p1, p0)
    if v <= swapped:
        return valence_part
    return _valence_part_from_radical_vector(swapped) or valence_part


def _canonicalize_valence_symmetric_reactant_dots(valence_part: str) -> str:
    """When reactant dot centers match in element and charge, ^…:0 vs ^…:1 are duplicates.

    Only valid when product fragment charges are also identical; otherwise map slots
    0/1 track which reactant center becomes which product (e.g. O+0·O+0 → O−–O+).
    """
    return _canonicalize_valence_symmetric_slot_pair(
        valence_part, _swap_valence_reactant_atom_indices
    )


def _canonicalize_bond_side_symmetric(s: str) -> str:
    """For [A:1]-[B:2] or [A:1]=[B:2], reorder so (charge, bracket) is lexicographically ascending."""
    m = re.match(r"^(\[[^\]]+\])([-=#.])(\[[^\]]+\])$", s.strip())
    if not m:
        return s
    a, bond, b = m.group(1), m.group(2), m.group(3)
    da = extract_component_data(a)
    db = extract_component_data(b)
    if not da or not db or da[0] != db[0]:
        return s
    ca, cb = da[1], db[1]
    if (ca, a) <= (cb, b):
        return f"{a}{bond}{b}"
    return f"{_set_bracket_map_num(b, 1)}{bond}{_set_bracket_map_num(a, 2)}"


def _swap_bond_side_maps(s: str) -> str:
    """Swap the two components in a bonded pair: [A:1]bond[B:2] -> [B:1]bond[A:2]."""
    m = re.match(r"^(\[[^\]]+\])([-=#.])(\[[^\]]+\])$", s.strip())
    if not m:
        return s
    a, bond, b = m.group(1), m.group(2), m.group(3)
    return f"{_set_bracket_map_num(b, 1)}{bond}{_set_bracket_map_num(a, 2)}"


def _canonicalize_dot_side_symmetric(s: str) -> str:
    """For [A:1].[B:2] with same element on both fragments, sort by (charge, bracket) and renumber maps.

    When reactants are the same element pair (OO, HH, CC), fragment order is not distinguishing;
    this matches normalize_rule's dot-side sort for reactants_same.
    """
    if "." not in s:
        return s
    comps = re.findall(r"\[[^\]]+\]", s)
    if len(comps) != 2:
        return s
    comp_data = [(c, extract_component_data(c)) for c in comps]
    comp_data = [(c, d) for c, d in comp_data if d]
    if len(comp_data) != 2:
        return s
    if comp_data[0][1][0] != comp_data[1][1][0]:
        return s
    comp_data.sort(key=lambda x: (x[1][1], x[0]))
    return ".".join(
        f"[{extract_component_base(c)}:{i}]" for i, (c, _) in enumerate(comp_data, 1)
    )


def _map_positions_in_side(side: str) -> Tuple[int, int]:
    """First string positions of map :1 and :2 in *side*, or -1 if absent."""
    pos1, pos2 = -1, -1
    for pat in (":1]", ";H0:1]"):
        idx = side.find(pat)
        if idx != -1 and (pos1 == -1 or idx < pos1):
            pos1 = idx
    for pat in (":2]", ";H0:2]"):
        idx = side.find(pat)
        if idx != -1 and (pos2 == -1 or idx < pos2):
            pos2 = idx
    return pos1, pos2


def _swap_two_component_side(side: str) -> str:
    """Swap the two bracket groups on a dot or bonded side (maps on atoms unchanged)."""
    s = side.strip()
    m_dot = re.match(r"^(\[[^\]]+\])\.(\[[^\]]+\])$", s)
    if m_dot:
        return f"{m_dot.group(2)}.{m_dot.group(1)}"
    m_bond = re.match(r"^(\[[^\]]+\])([-=#.])(\[[^\]]+\])$", s)
    if m_bond:
        return f"{m_bond.group(3)}{m_bond.group(2)}{m_bond.group(1)}"
    return side


def _map_valence_by_side(
    left: str, right: str, radical: Tuple[int, int, int, int]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Valence per map number on reactant (left) and product (right) sides."""
    reactant: Dict[int, int] = {}
    product: Dict[int, int] = {}
    for i, comp in enumerate(re.findall(r"\[[^\]]+\]", left)[:2]):
        mn = _bracket_map_num(comp)
        if mn and radical[i]:
            reactant[int(mn)] = radical[i]
    for i, comp in enumerate(re.findall(r"\[[^\]]+\]", right)[:2]):
        mn = _bracket_map_num(comp)
        if mn and radical[i + 2]:
            product[int(mn)] = radical[i + 2]
    return reactant, product


def _radical_vector_for_layout(
    left: str,
    right: str,
    reactant_map_val: Dict[int, int],
    product_map_val: Dict[int, int],
) -> Tuple[int, int, int, int]:
    """Positional slots 0–3 from map-keyed valence and current component order."""
    v = [0, 0, 0, 0]
    for i, comp in enumerate(re.findall(r"\[[^\]]+\]", left)[:2]):
        mn = _bracket_map_num(comp)
        if mn and int(mn) in reactant_map_val:
            v[i] = reactant_map_val[int(mn)]
    for i, comp in enumerate(re.findall(r"\[[^\]]+\]", right)[:2]):
        mn = _bracket_map_num(comp)
        if mn and int(mn) in product_map_val:
            v[i + 2] = product_map_val[int(mn)]
    return tuple(v)


def _remap_map_valence_dict(
    d: Dict[int, int], old_to_new: Dict[str, str]
) -> Dict[int, int]:
    """Apply bracket map renumbering to map-keyed valence (e.g. 2→1, 1→2)."""
    out: Dict[int, int] = {}
    for m, v in d.items():
        new_m = int(old_to_new.get(str(m), str(m)))
        out[new_m] = v
    return out


def _valence_part_for_layout(left: str, right: str, valence_part: str) -> str:
    """Rebuild |^…| for the current left/right component order."""
    radical = _radical_vector_from_valence_part(valence_part)
    r_mv, p_mv = _map_valence_by_side(left, right, radical)
    return _valence_part_from_radical_vector(
        _radical_vector_for_layout(left, right, r_mv, p_mv)
    )


def ensure_map_reading_order(rule: str) -> str:
    """Ensure map :1 appears before map :2 in reading order on both sides of >>.

    Reorders components when needed; map numbers stay on the same atoms. Valence
    is keyed by map number per side (reactant vs product), then re-laid onto
    positional slots 0–3 after reordering.
    """
    if ">>" not in rule:
        return rule
    left, rest = rule.split(">>", 1)

    valence_part = ""
    if "|" in rest:
        rhs, valence_part = rest.split("|", 1)
        valence_part = "|" + valence_part
    else:
        rhs = rest

    radical = _radical_vector_from_valence_part(valence_part)
    reactant_mv, product_mv = _map_valence_by_side(left, rhs, radical)

    pos1_l, pos2_l = _map_positions_in_side(left)
    if pos1_l != -1 and pos2_l != -1 and pos2_l < pos1_l:
        left = _swap_two_component_side(left)

    pos1_r, pos2_r = _map_positions_in_side(rhs)
    if pos1_r != -1 and pos2_r != -1 and pos2_r < pos1_r:
        rhs = _swap_two_component_side(rhs)

    radical = _radical_vector_for_layout(left, rhs, reactant_mv, product_mv)
    new_vp = _valence_part_from_radical_vector(radical)
    rule_out = left + ">>" + _join_rhs_and_valence(rhs, new_vp)
    return normalize_valence_indices_in_rule(rule_out)


def canonicalize_symmetric_pair_rule(rule: str) -> str:
    """For same-element pairs (OO, HH, CC, …), canonicalize fragment order so duplicates merge.

    - **Reverse** (dot products on left): sort left fragments by descending charge (more positive
      first), then bracket; remap RHS. Matches ``[O+1:1].[O+0:2]`` not ``[O+0:1].[O+1:2]`` for OO.
      When left is already sorted, still canonicalize a same-element product bond (merges symmetric
      ;H0 on one O vs the other).
    - **Forward** (bond on left, dot products on right): when left bond is same-element, sort
      right fragments; order only matters when the two atoms differ (HO, OC, …). Always run
      ``_canonicalize_dot_side_symmetric`` when charges match so symmetric H0 placements merge.
    - **Forward** (bond on left, bond on right): same-element product bond (e.g. O−/O+ from O₂);
      sort product fragments like ``_canonicalize_bond_side_symmetric`` so O+−O− and O−−O+ merge.
    """
    if ">>" not in rule:
        return rule
    left, rest = rule.split(">>", 1)
    valence_part = ""
    if "|" in rest:
        rhs, valence_part = rest.split("|", 1)
        valence_part = "|" + valence_part
    else:
        rhs = rest

    # Case 1: reverse — dot products on left
    if "." in left:
        comps = re.findall(r"\[[^\]]+\]", left)
        if len(comps) != 2:
            return rule
        d1 = extract_component_data(comps[0])
        d2 = extract_component_data(comps[1])
        if not d1 or not d2:
            return rule
        e1, c1 = d1[0], d1[1]
        e2, c2 = d2[0], d2[1]
        if e1 != e2:
            return rule

        frag1, frag2 = comps[0], comps[1]
        # Different reactant charges: do not reorder dots (order encodes atom identity).
        if c1 != c2:
            rhs0 = rhs.strip()
            rm = re.match(r"^(\[[^\]]+\])([-=#.])(\[[^\]]+\])$", rhs0)
            if rm:
                pr1 = extract_component_data(rm.group(1))
                pr2 = extract_component_data(rm.group(3))
                if pr1 and pr2 and pr1[0] == pr2[0]:
                    # Product bond order encodes charge-swap (e.g. O−·O+ → O+0−O−1);
                    # do not sort by ascending charge when product charges differ.
                    if pr1[1] == pr2[1]:
                        rhs_c = _canonicalize_bond_side_symmetric(rhs0)
                        vp = (
                            _valence_part_for_layout(left.strip(), rhs_c, valence_part)
                            if rhs_c != rhs0
                            else valence_part
                        )
                    else:
                        rhs_c = rhs0
                        vp = valence_part
                    return left.strip() + ">>" + _join_rhs_and_valence(rhs_c, vp)
            return rule

        # Descending charge on reactants (same charge only), e.g. O+ before O+0.
        if (-c1, frag1) <= (-c2, frag2):
            # Left already canonical: still merge symmetric H0 on product bond (e.g. OO reverse).
            rhs0 = rhs.strip()
            rm = re.match(r"^(\[[^\]]+\])([-=#.])(\[[^\]]+\])$", rhs0)
            if rm:
                pr1 = extract_component_data(rm.group(1))
                pr2 = extract_component_data(rm.group(3))
                if pr1 and pr2 and pr1[0] == pr2[0]:
                    if pr1[1] == pr2[1]:
                        rhs_c = _canonicalize_bond_side_symmetric(rhs0)
                        vp = (
                            _valence_part_for_layout(left.strip(), rhs_c, valence_part)
                            if rhs_c != rhs0
                            else valence_part
                        )
                    else:
                        rhs_c = rhs0
                        vp = valence_part
                    # Canonicalize: global swap when both pairs identical. When product
                    # bond charges differ, reactant slots 0/1 are not interchangeable.
                    if c1 == c2 and pr1 and pr2 and pr1[1] == pr2[1]:
                        vp = _canonicalize_valence_symmetric_pair(vp)
                    return left.strip() + ">>" + _join_rhs_and_valence(rhs_c, vp)
            if c1 == c2:
                rm_dots = re.match(r"^(\[[^\]]+\])\.(\[[^\]]+\])$", rhs.strip())
                if rm_dots:
                    rp1 = extract_component_data(rm_dots.group(1))
                    rp2 = extract_component_data(rm_dots.group(2))
                    if rp1 and rp2 and rp1[1] == rp2[1]:
                        vp = _canonicalize_valence_symmetric_reactant_dots(valence_part)
                        if vp != valence_part:
                            return left.strip() + ">>" + _join_rhs_and_valence(rhs.strip(), vp)
            return rule

        m1 = _bracket_map_num(frag1)
        m2 = _bracket_map_num(frag2)
        if not m1 or not m2 or m1 == m2:
            return rule

        first, second = frag2, frag1
        new_left = f"{_set_bracket_map_num(first, 1)}.{_set_bracket_map_num(second, 2)}"
        old_to_new = {m2: "1", m1: "2"}

        radical = _radical_vector_from_valence_part(valence_part)
        r_mv, p_mv = _map_valence_by_side(left, rhs, radical)
        r_mv = _remap_map_valence_dict(r_mv, old_to_new)
        p_mv = _remap_map_valence_dict(p_mv, old_to_new)
        rhs2 = _remap_maps_in_string(rhs, old_to_new)
        rhs2_comps = re.findall(r'\[[^\]]+\]', rhs2) if rhs2 else []
        rhs2_pr1 = extract_component_data(rhs2_comps[0]) if len(rhs2_comps) > 0 else None
        rhs2_pr2 = extract_component_data(rhs2_comps[1]) if len(rhs2_comps) > 1 else None
        products_same = rhs2_pr1 and rhs2_pr2 and rhs2_pr1[1] == rhs2_pr2[1]
        if products_same:
            rhs2 = _canonicalize_bond_side_symmetric(rhs2)
        val2 = _valence_part_from_radical_vector(
            _radical_vector_for_layout(new_left, rhs2, r_mv, p_mv)
        )
        if c1 == c2 and products_same:
            val2 = _canonicalize_valence_symmetric_pair(val2)
        elif products_same:
            val2 = _canonicalize_valence_symmetric_product_bond(val2)
        return new_left + ">>" + _join_rhs_and_valence(rhs2, val2)

    # Forward: same-element bond on left (OO, CC, …)
    lm = re.match(r"^(\[[^\]]+\])([-=#.])(\[[^\]]+\])$", left.strip())
    if not lm:
        return rule
    dl = extract_component_data(lm.group(1))
    dr = extract_component_data(lm.group(3))
    if not dl or not dr or dl[0] != dr[0]:
        return rule

    # Case 2: dot products on right
    if "." not in rhs:
        # Case 3: bond on right (e.g. OO DS: = → − −); merge O+−O− with O−−O+
        rm = re.match(r"^(\[[^\]]+\])([-=#.])(\[[^\]]+\])$", rhs.strip())
        if not rm:
            return rule
        pr1 = extract_component_data(rm.group(1))
        pr2 = extract_component_data(rm.group(3))
        if not pr1 or not pr2 or pr1[0] != pr2[0]:
            return rule
        left0 = left.strip()
        rhs0 = rhs.strip()
        lhs_c = _canonicalize_bond_side_symmetric(left0)
        vp = valence_part
        if lhs_c != left0:
            vp = _swap_valence_reactant_atom_indices(vp)
        if dl[1] != dr[1]:
            # Different reactant charges: product map order encodes which
            # reactant atom becomes which product atom (charge-swapping vs
            # charge-preserving). Only canonicalize lhs; if it was swapped,
            # correspondingly swap rhs maps to maintain atom tracking.
            if lhs_c != left0:
                rhs0 = _swap_bond_side_maps(rhs0)
                vp = _swap_valence_product_atom_indices(vp)
            return lhs_c + ">>" + _join_rhs_and_valence(rhs0, vp)
        rhs_c = _canonicalize_bond_side_symmetric(rhs0)
        if rhs_c != rhs0:
            vp = _swap_valence_product_atom_indices(vp)
        # Collapse symmetric slots. Use global (simultaneous) swap when both pairs
        # are identical -- independent swaps would conflate physically distinct rules.
        if dl[1] == dr[1] and pr1[1] == pr2[1]:
            vp = _canonicalize_valence_symmetric_pair(vp)
        elif pr1[1] == pr2[1]:
            vp = _canonicalize_valence_symmetric_product_bond(vp)
        return lhs_c + ">>" + _join_rhs_and_valence(rhs_c, vp)

    rhs_comps = re.findall(r"\[[^\]]+\]", rhs)
    if len(rhs_comps) != 2:
        return rule
    # Same element, different bond charges (e.g. O+0–O+1): mirror O+1–O+0 is the same
    # charge-swap; canonicalize to lower formal charge on map :1 and remap products.
    if dl[1] != dr[1]:
        left0 = left.strip()
        rhs0 = rhs.strip()
        lhs_c = _canonicalize_bond_side_symmetric(left0)
        vp = valence_part
        if lhs_c != left0:
            comps_l = re.findall(r"\[[^\]]+\]", left0)
            if len(comps_l) >= 2:
                m1 = _bracket_map_num(comps_l[0])
                m2 = _bracket_map_num(comps_l[1])
                if m1 and m2 and m1 != m2:
                    old_to_new = {m1: "2", m2: "1"}
                    rhs2 = _remap_maps_in_string(rhs0, old_to_new)
                    radical = _radical_vector_from_valence_part(valence_part)
                    r_mv, p_mv = _map_valence_by_side(left0, rhs0, radical)
                    r_mv = _remap_map_valence_dict(r_mv, old_to_new)
                    p_mv = _remap_map_valence_dict(p_mv, old_to_new)
                    vp = _valence_part_from_radical_vector(
                        _radical_vector_for_layout(lhs_c, rhs2, r_mv, p_mv)
                    )
                    return lhs_c + ">>" + _join_rhs_and_valence(rhs2, vp)
        return rule
    f1, f2 = rhs_comps[0], rhs_comps[1]
    dc1 = extract_component_data(f1)
    dc2 = extract_component_data(f2)
    # Only sort when products are same element. Charge may differ (e.g. C+1 vs C-1)
    # because reactants are identical so :1/:2 are arbitrary -- sorting is safe as long
    # as product slots 2<->3 swap with the valence (handled below).
    if not dc1 or not dc2 or dc1[0] != dc2[0]:
        return rule
    left0 = left.strip()
    rhs0 = rhs.strip()
    lhs_c = _canonicalize_bond_side_symmetric(left0)
    rhs_c = _canonicalize_dot_side_symmetric(rhs0)
    vp = valence_part
    if lhs_c != left0:
        vp = _swap_valence_reactant_atom_indices(vp)
    if rhs_c != rhs0:
        vp = _swap_valence_product_atom_indices(vp)
    # dl[1]==dr[1] gated above (reactants identical). Use global swap when products
    # also identical; when product charges differ, slot order encodes atom identity.
    if dc1[1] == dc2[1]:
        vp = _canonicalize_valence_symmetric_pair(vp)
    return lhs_c + ">>" + _join_rhs_and_valence(rhs_c, vp)


def extract_component_base(component: str) -> Optional[str]:
    match = re.search(r'\[([^\]]+):\d+\]', component)
    return match.group(1) if match else None


def component_base_canonical(component_base: str) -> str:
    """Canonical component base for lookup (strips legacy ``;H0`` if present)."""
    return component_base.replace(';H0', '') if component_base else ''


def _build_bond_limit_lookup(entries: List[Tuple[str, int]]) -> Dict[str, int]:
    """Bond-limit per component base from [MAX_BOND] entries."""
    lookup: Dict[str, int] = {}
    for comp, bond_limit in entries:
        base = extract_component_base(comp)
        if not base:
            continue
        lookup[component_base_canonical(base)] = bond_limit
    return lookup


def get_bond_limit(
    component_base: str,
    entries: List[Tuple[str, int]],
    lookup: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    canonical = component_base_canonical(component_base)
    if lookup is not None:
        return lookup.get(canonical)
    for comp, bond_limit in entries:
        base = extract_component_base(comp)
        if base == component_base or component_base_canonical(base) == canonical:
            return bond_limit
    return None


def valence_net_abs_delta(valence_str: str) -> int:
    """|total reactant valence − total product valence| in the four-slot valence convention."""
    tr, tp = total_radicals_from_valence(valence_str)
    return abs(tr - tp)


def within_radicals_net_change(
    valence_str: str, max_net: int = DEFAULT_RADICALS_CHANGE_LIMIT
) -> bool:
    """True if |Σ reactant − Σ product| in |^...| slots does not exceed max_net."""
    return valence_net_abs_delta(valence_str) <= max_net


def formal_charge_sum_from_rule(rule: str) -> Tuple[int, int]:
    """(Σ formal charge on reactant side, Σ on product side) for a two-fragment rule."""
    if '>>' not in rule:
        return 0, 0
    left, right_rest = rule.split('>>', 1)
    right = right_rest.split('|')[0]
    sum_left = sum(extract_charge(c) for c in re.findall(r'\[[^\]]+\]', left))
    sum_right = sum(extract_charge(c) for c in re.findall(r'\[[^\]]+\]', right))
    return sum_left, sum_right


def formal_charge_net_abs_delta(rule: str) -> int:
    """|Σ formal charge(reactants) − Σ formal charge(products)|."""
    sl, sr = formal_charge_sum_from_rule(rule)
    return abs(sl - sr)


def within_formal_charge_net_change(
    rule: str, charge_change_limit: int = DEFAULT_CHARGE_CHANGE_LIMIT
) -> bool:
    """True if net formal-charge imbalance across >> is within charge_change_limit."""
    return formal_charge_net_abs_delta(rule) <= charge_change_limit


def _bond_order_from_side(side: str) -> Optional[int]:
    """Bond order 0–3 from a single bonded fragment like [A:1]-[B:2] or [A:1].[B:2]; None if unknown."""
    s = side.strip().split('|')[0]
    if re.search(r'\[[^\]]+\]\.\[[^\]]+\]', s):
        return 0
    m = re.search(r'\]([-=#.])(\[)', s)
    if not m:
        return None
    sym = m.group(1)
    for order, ch in BOND_SYMBOLS.items():
        if ch == sym:
            return order
    return None


def within_valence_radical_monotone(valence_str: str, rule: str) -> bool:
    """Radical totals in |^...| must follow bond-order and/or formal-charge flow.

    Charge-conserving cleavage (lb > rb): tp >= tr; formation (lb < rb): tp <= tr.
    When Σ formal charge changes, also allow the charge-coupled direction (reduction
    on cleavage: tp <= tr; oxidation on formation: tp >= tr). A rule passes if either
    bond-order or charge-coupled monotonicity holds.
    """
    if '>>' not in rule:
        return True
    left, right = rule.split('>>', 1)
    right_side = right.split('|')[0]
    lb = _bond_order_from_side(left)
    rb = _bond_order_from_side(right_side)
    if lb is None or rb is None:
        return True
    tr, tp = total_radicals_from_valence(valence_str)
    sum_left, sum_right = formal_charge_sum_from_rule(rule)
    dq = sum_right - sum_left

    if lb > rb:
        bond_ok = tp >= tr
    elif lb < rb:
        bond_ok = tp <= tr
    else:
        return True

    if dq == 0:
        return bond_ok

    if lb > rb and dq < 0:
        charge_ok = tp <= tr
    elif lb > rb:
        charge_ok = tp >= tr
    elif lb < rb and dq > 0:
        charge_ok = tp >= tr
    else:
        charge_ok = tp <= tr

    return bond_ok or charge_ok


def total_radicals_from_valence(valence_str: str) -> Tuple[int, int]:
    """From valence notation like |^1:0,1;^2:2,3| (or legacy ,^ between clauses) compute totals."""
    total_reactant = 0
    total_product = 0
    # Strip leading | and trailing | if present
    s = valence_str.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    for part in re.split(r'(?:,\s*|;\s*)\^', s):
        part = part.strip()
        if part.startswith('^'):
            part = part[1:]
        if ':' not in part:
            continue
        v_str, idx_str = part.split(':', 1)
        try:
            v = int(v_str.strip())
        except ValueError:
            continue
        indices = [int(x.strip()) for x in idx_str.split(',') if x.strip().isdigit()]
        for i in indices:
            if i in (0, 1):
                total_reactant += v
            elif i in (2, 3):
                total_product += v
    return total_reactant, total_product


def within_radical_limit(valence_str: str, limit_radicals: Optional[int]) -> bool:
    """True if total valence on reactant slots and on product slots are each <= limit_radicals (when set)."""
    if limit_radicals is None:
        return True
    total_reactant, total_product = total_radicals_from_valence(valence_str)
    return total_reactant <= limit_radicals and total_product <= limit_radicals


def _elements_two_fragment_side(side: str) -> Optional[Tuple[str, str]]:
    """Component bases (e.g. O+0, C+1) for the two fragments, ordered by ascending map number."""
    d: Dict[int, str] = {}
    for m in re.finditer(r'\[([^\]]+)\]', side.strip()):
        frag = f'[{m.group(1)}]'
        if not re.search(r':\d+\]$', frag):
            frag = _normalize_component(frag)
        base = extract_component_base(frag)
        if not base:
            continue
        mm = re.search(r':(\d+)\]$', frag)
        if not mm:
            continue
        d[int(mm.group(1))] = component_base_canonical(base)
    if len(d) < 2:
        return None
    m1, m2 = sorted(d.keys())
    return d[m1], d[m2]


def _valence_slot_elements(rule: str) -> Optional[Tuple[str, str, str, str]]:
    """(slot0..slot3) component bases matching valence indices 0,1 = reactants, 2,3 = products."""
    if '>>' not in rule:
        return None
    left, right_full = rule.split('>>', 1)
    right = right_full.split('|')[0]
    lp = _elements_two_fragment_side(left)
    rp = _elements_two_fragment_side(right)
    if lp is None or rp is None:
        return None
    return lp[0], lp[1], rp[0], rp[1]


def _valence_per_slot_max(valence_str: str) -> Dict[int, int]:
    """Maximum ^v per valence slot 0..3."""
    out = {0: 0, 1: 0, 2: 0, 3: 0}
    s = valence_str.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    for part in re.split(r'(?:,\s*|;\s*)\^', s):
        part = part.strip()
        if part.startswith('^'):
            part = part[1:]
        if ':' not in part:
            continue
        v_str, idx_str = part.split(':', 1)
        try:
            v = int(v_str.strip())
        except ValueError:
            continue
        for x in idx_str.split(','):
            x = x.strip()
            if not x.isdigit():
                continue
            i = int(x)
            if 0 <= i <= 3:
                out[i] = max(out[i], v)
    return out


def _max_radical_cap_for_component(base: Optional[str], caps: Optional[Dict[str, int]]) -> Optional[int]:
    """[MAX_RADICAL] cap for this component base (e.g. O+0), or legacy element-only key; None if not listed."""
    if not caps or not base:
        return None
    key = component_base_canonical(base)
    c = caps.get(key)
    if c is not None:
        return c
    m = re.match(r'^([A-Za-z]+)', key)
    if m:
        raw = m.group(1)
        sym = raw[0].upper() + (raw[1:].lower() if len(raw) > 1 else '')
        for candidate in (sym, sym.upper(), raw.upper()):
            c = caps.get(candidate)
            if c is not None:
                return c
    return None


def _valence_slot_loop_hi(
    component_base: Optional[str],
    caps: Optional[Dict[str, int]],
    limit_radicals: Optional[int],
) -> int:
    """Per-slot ^v upper bound for mixed |^...| loops ([MAX_RADICAL], then [params].limit_radicals)."""
    hi: Optional[int] = None
    if caps and component_base:
        c = _max_radical_cap_for_component(component_base, caps)
        if c is not None:
            hi = max(0, int(c))
    if limit_radicals is not None:
        lr = max(0, limit_radicals)
        hi = min(hi, lr) if hi is not None else lr
    return hi if hi is not None else 0


def within_per_atom_max_radical(
    valence_str: str,
    rule: str,
    caps: Optional[Dict[str, int]],
) -> bool:
    """True if each slot's valence v is <= [MAX_RADICAL] for that component type (when table present)."""
    if not caps:
        return True
    els = _valence_slot_elements(rule)
    if els is None:
        return True
    per = _valence_per_slot_max(valence_str)
    for i in range(4):
        v = per.get(i, 0)
        if v <= 0:
            continue
        sym = els[i]
        cap = _max_radical_cap_for_component(sym, caps)
        if cap is None:
            continue
        if v > cap:
            return False
    return True


def total_abs_charges_from_rule(rule: str) -> Tuple[int, int]:
    """From rule string 'left>>right' or 'left>>right|^...|' return (sum of |charge| on left, sum of |charge| on right)."""
    if '>>' not in rule:
        return 0, 0
    left, right_rest = rule.split('>>', 1)
    right = right_rest.split('|')[0]
    left_comps = re.findall(r'\[[^\]]+\]', left)
    right_comps = re.findall(r'\[[^\]]+\]', right)
    sum_abs_left = sum(abs(extract_charge(c)) for c in left_comps)
    sum_abs_right = sum(abs(extract_charge(c)) for c in right_comps)
    return sum_abs_left, sum_abs_right


def _abs_charge_sum_two(comp1: str, comp2: str) -> int:
    """Sum of |formal charge| for two fragment components (reactant or product side)."""
    return abs(_fragment_type_key(comp1)[1]) + abs(_fragment_type_key(comp2)[1])


def _pair_exceeds_abs_charge_limit(
    comp1: str, comp2: str, limit_abs_charges: Optional[int]
) -> bool:
    """True if this fragment pair can never satisfy limit_abs_charges (early prune)."""
    if limit_abs_charges is None:
        return False
    return _abs_charge_sum_two(comp1, comp2) > limit_abs_charges


def within_charge_limit(rule: str, limit_abs_charges: Optional[int]) -> bool:
    """True if sum of |charge| per atom on left and on right are each <= limit_abs_charges (when set)."""
    if limit_abs_charges is None:
        return True
    sum_abs_left, sum_abs_right = total_abs_charges_from_rule(rule)
    return sum_abs_left <= limit_abs_charges and sum_abs_right <= limit_abs_charges


def _rule_passes_limits(
    rule: str,
    valence_str: Optional[str],
    limit_radicals: Optional[int],
    limit_abs_charges: Optional[int],
    max_radical_by_element: Optional[Dict[str, int]] = None,
    radicals_change_limit: int = DEFAULT_RADICALS_CHANGE_LIMIT,
    charge_change_limit: int = DEFAULT_CHARGE_CHANGE_LIMIT,
    bond_limit_lookup: Optional[Dict[str, int]] = None,
    entries: Optional[List[Tuple[str, int]]] = None,
) -> bool:
    """True if rule passes valence shape, radical limit (when valence_str) and charge limit."""
    if not within_formal_charge_net_change(rule, charge_change_limit):
        return False
    if valence_str is not None:
        if not within_radicals_net_change(valence_str, radicals_change_limit):
            return False
        if not within_valence_radical_monotone(valence_str, rule):
            return False
        if not within_radical_limit(valence_str, limit_radicals):
            return False
        if not within_per_atom_max_radical(valence_str, rule, max_radical_by_element):
            return False
    if not within_charge_limit(rule, limit_abs_charges):
        return False
    # Reject rules where the bond order on either side exceeds a component's MAX_BOND.
    # fix_reaction_map_consistency can substitute a component that was excluded from
    # _pairs_for_bond_order, so this check must be applied after finalization.
    if bond_limit_lookup or entries:
        if ">>" in rule:
            left, rhs = rule.split(">>", 1)
            for side in (left, rhs.split("|")[0]):
                m = re.search(r'\]([-=#.])\[', side)
                if not m:
                    continue
                bo = {"-": 1, "=": 2, "#": 3, ".": 0}.get(m.group(1), 0)
                if bo == 0:
                    continue
                for cm in re.finditer(r'\[([A-Za-z]+[+-]?\d+)(?:;H0)?:\d+\]', side):
                    limit = get_bond_limit(cm.group(1), entries or [], bond_limit_lookup)
                    if limit is not None and bo > limit:
                        return False
    return True


def _default_jobs() -> int:
    n = os.cpu_count() or 1
    return max(1, n - 1)


def _prefer_process_pool() -> bool:
    """Use processes only when loaded as a package module (``python -m utils.ruleset_generator.build_bond_rules``)."""
    return __name__.startswith('utils.')


def _parallel_map(fn, items, jobs: int, chunksize: int = 1):
    """Map *fn* over *items* with process or thread pool depending on invocation."""
    if jobs <= 1:
        return [fn(x) for x in items]
    Executor = ProcessPoolExecutor if _prefer_process_pool() else ThreadPoolExecutor
    with Executor(max_workers=jobs) as executor:
        return list(executor.map(fn, items, chunksize=chunksize))


def _fragment_type_key(comp: str) -> Tuple[str, int]:
    """(element, formal_charge) for ordering / symmetry checks."""
    d = extract_component_data(comp)
    if d:
        return d
    m = re.search(r'\[([A-Za-z]+)([+-]?\d+)', comp)
    if m:
        c_str = m.group(2)
        return (m.group(1), int(c_str.replace('+', '')) if c_str else 0)
    return (comp.strip(), 0)


def _is_same_element_pair(comp_a: str, comp_b: str) -> bool:
    return _fragment_type_key(comp_a)[0] == _fragment_type_key(comp_b)[0]


def _skip_symmetric_valence_on_same_component(
    same_component: bool, v_first: int, v_second: int, *, charges_unchanged: bool
) -> bool:
    """Skip ordered (v_first, v_second) only when charges are unchanged on both sides.

    When reactants share a component type but products differ (e.g. O+0.O+0 -> O-1-O+1),
    ^1 on map slot 0 vs slot 1 are distinct and must not be collapsed.
    """
    return charges_unchanged and same_component and v_first > v_second


def _product_order_matches_reactant_elements(
    comp1_left: str, comp2_left: str, p1_right: str, p2_right: str
) -> bool:
    """True if product fragments at map :1/:2 use the same elements as reactants (after normalize_rule)."""
    d1l = extract_component_data(comp1_left)
    d2l = extract_component_data(comp2_left)
    d1p = extract_component_data(p1_right)
    d2p = extract_component_data(p2_right)
    if not d1l or not d2l or not d1p or not d2p:
        return True
    return d1p[0] == d1l[0] and d2p[0] == d2l[0]


def _skip_mirror_same_element_pair(comp_a: str, comp_b: str) -> bool:
    """Skip (b, a) when both fragments are the same element — one unordered pair is enough."""
    if not _is_same_element_pair(comp_a, comp_b):
        return False
    return _fragment_type_key(comp_a) > _fragment_type_key(comp_b)


def _skip_noncanonical_fragment_pair_order(comp_a: str, comp_b: str) -> bool:
    """Skip unordered duplicate (b,a): same element mirrors or hetero fragment reorder."""
    return _skip_mirror_same_element_pair(comp_a, comp_b) or (
        not _is_same_element_pair(comp_a, comp_b)
        and _fragment_type_key(comp_a) > _fragment_type_key(comp_b)
    )


def _mixed_valence_product_sum_bounds(
    reactant_sum: int,
    radicals_change_limit: int,
    *,
    cleavage: bool,
    mp1: int,
    mp2: int,
    limit_radicals: Optional[int],
    formal_charge_redistributes: bool = False,
) -> Tuple[int, int]:
    """p_sum = p1+p2 range for mixed |^...| loops.

    Charge-conserving: cleavage tp>=tr (p_lo=r), formation tp<=tr (p_hi=r).
    Formal-charge redistribution: symmetric band [r−limit, r+limit] on both sides.
    """
    r = reactant_sum
    cap = mp1 + mp2
    if formal_charge_redistributes:
        p_lo = max(0, r - radicals_change_limit)
        p_hi = min(r + radicals_change_limit, cap)
    elif cleavage:
        p_lo = max(0, r)
        p_hi = min(r + radicals_change_limit, cap)
    else:
        p_lo = max(0, r - radicals_change_limit)
        p_hi = min(r, cap)
    if limit_radicals is not None:
        p_hi = min(p_hi, limit_radicals)
    # Ensure p_lo never exceeds p_hi (can happen after limit_radicals tightens p_hi)
    p_lo = min(p_lo, p_hi)
    return p_lo, p_hi


def _valence_part_from_rule(rule: str) -> Optional[str]:
    if '>>' not in rule:
        return None
    rhs = rule.split('>>', 1)[1]
    if '|' not in rhs:
        return None
    vp = '|' + rhs.split('|', 1)[1]
    return vp if vp.startswith('|^') else None


def _map_valence_slot_indices(
    left_side: str, right_side_clean: str
) -> Optional[Tuple[int, int, int, int]]:
    """Return sorted (idx1_left, idx2_left, idx1_right, idx2_right) or None."""
    left_indices = _RE_MAP_IDX.findall(left_side)
    right_indices = _RE_MAP_IDX.findall(right_side_clean)
    if len(left_indices) < 2 or len(right_indices) < 2:
        return None
    idx1_right = int(right_indices[0]) + 1
    idx2_right = int(right_indices[1]) + 1
    idx1_right, idx2_right = min(idx1_right, idx2_right), max(idx1_right, idx2_right)
    idx1_left = int(left_indices[0]) - 1
    idx2_left = int(left_indices[1]) - 1
    idx1_left, idx2_left = min(idx1_left, idx2_left), max(idx1_left, idx2_left)
    return idx1_left, idx2_left, idx1_right, idx2_right


def _mixed_valence_part_from_counts(
    r1: int,
    r2: int,
    p1: int,
    p2: int,
    idx1_left: int,
    idx2_left: int,
    idx1_right: int,
    idx2_right: int,
) -> str:
    """Build |^n:i,j| from per-slot radical counts (empty string if all zero)."""
    valence_to_indices: Dict[int, List[int]] = {}
    if r1 > 0:
        valence_to_indices.setdefault(r1, []).append(idx1_left)
    if r2 > 0:
        valence_to_indices.setdefault(r2, []).append(idx2_left)
    if p1 > 0:
        valence_to_indices.setdefault(p1, []).append(idx1_right)
    if p2 > 0:
        valence_to_indices.setdefault(p2, []).append(idx2_right)
    if not valence_to_indices:
        return ""
    parts = [
        f"{v}:{','.join(map(str, sorted(idxs)))}"
        for v, idxs in sorted(valence_to_indices.items())
    ]
    return '|^' + ';^'.join(parts) + '|'


def _append_mixed_valence_combinations(
    rules: List[str],
    *,
    left_side: str,
    right_side_clean: str,
    cleavage: bool,
    formal_charge_redistributes: bool,
    idx1_left: int,
    idx2_left: int,
    idx1_right: int,
    idx2_right: int,
    same_component_left: bool,
    same_component_right: bool,
    charges_unchanged: bool,
    r1_hi: int,
    r2_hi: int,
    mp1: int,
    mp2: int,
    el_r: Optional[Tuple[str, str]],
    max_radical_by_element: Optional[Dict[str, int]],
    limit_radicals: Optional[int],
    radicals_change_limit: int,
    passes,
) -> None:
    """Enumerate mixed reactant/product |^…| assignments for one bond-order step."""
    L = limit_radicals
    for r1 in range(0, (min(r1_hi, L) + 1) if L is not None else (r1_hi + 1)):
        r2_top = min(r2_hi, L - r1) if L is not None else r2_hi
        if r2_top < 0:
            continue
        for r2 in range(0, r2_top + 1):
            R = r1 + r2
            p_lo, p_hi = _mixed_valence_product_sum_bounds(
                R,
                radicals_change_limit,
                cleavage=cleavage,
                mp1=mp1,
                mp2=mp2,
                limit_radicals=L,
                formal_charge_redistributes=formal_charge_redistributes,
            )
            if max_radical_by_element and el_r:
                cr0 = _max_radical_cap_for_component(el_r[0], max_radical_by_element)
                cr1 = _max_radical_cap_for_component(el_r[1], max_radical_by_element)
                if cr0 is not None and cr1 is not None:
                    p_hi = min(p_hi, cr0 + cr1)
                    if cleavage or formal_charge_redistributes:
                        p_lo = min(p_lo, p_hi)
            for p_sum in range(p_lo, p_hi + 1):
                p1_lo = max(0, p_sum - mp2)
                p1_hi = min(mp1, p_sum)
                if p1_lo > p1_hi:
                    continue
                for p1 in range(p1_lo, p1_hi + 1):
                    p2 = p_sum - p1
                    if (
                        same_component_left
                        and same_component_right
                        and (r1, r2, p1, p2) > (r2, r1, p2, p1)
                    ):
                        continue
                    if (
                        not same_component_left
                        and same_component_right
                        and _skip_symmetric_valence_on_same_component(
                            True, p1, p2, charges_unchanged=charges_unchanged
                        )
                    ):
                        continue
                    valence_str = _mixed_valence_part_from_counts(
                        r1, r2, p1, p2, idx1_left, idx2_left, idx1_right, idx2_right
                    )
                    if not valence_str:
                        continue
                    r_mix = f"{left_side}>>{_join_rhs_and_valence(right_side_clean, valence_str)}"
                    if passes(r_mix, valence_str):
                        rules.append(r_mix)


def _append_charge_redistribution_mixed_valence(
    rules: List[str],
    *,
    left_side: str,
    right_side_clean: str,
    left_comps: List[Tuple[str, str, str]],
    right_comps: List[Tuple[str, str, str]],
    cleavage: bool,
    charges_unchanged: bool,
    limit_radicals: Optional[int],
    max_radical_by_element: Optional[Dict[str, int]],
    radicals_change_limit: int,
    passes,
) -> None:
    """Mixed |^…| when formal charge redistributes across a bond-order step."""
    slot_indices = _map_valence_slot_indices(left_side, right_side_clean)
    if slot_indices is None:
        return
    idx1_left, idx2_left, idx1_right, idx2_right = slot_indices
    comp1_base_left = f"{left_comps[0][0]}{left_comps[0][1]}"
    comp2_base_left = f"{left_comps[1][0]}{left_comps[1][1]}"
    comp1_base_right = f"{right_comps[0][0]}{right_comps[0][1]}"
    comp2_base_right = f"{right_comps[1][0]}{right_comps[1][1]}"
    el_l = _elements_two_fragment_side(left_side)
    el_r = _elements_two_fragment_side(right_side_clean)
    caps = max_radical_by_element
    L = limit_radicals
    r1_hi = _valence_slot_loop_hi(el_l[0] if el_l else None, caps, L)
    r2_hi = _valence_slot_loop_hi(el_l[1] if el_l else None, caps, L)
    mp1 = _valence_slot_loop_hi(el_r[0] if el_r else None, caps, L)
    mp2 = _valence_slot_loop_hi(el_r[1] if el_r else None, caps, L)
    _append_mixed_valence_combinations(
        rules,
        left_side=left_side,
        right_side_clean=right_side_clean,
        cleavage=cleavage,
        formal_charge_redistributes=True,
        idx1_left=idx1_left,
        idx2_left=idx2_left,
        idx1_right=idx1_right,
        idx2_right=idx2_right,
        same_component_left=comp1_base_left == comp2_base_left,
        same_component_right=comp1_base_right == comp2_base_right,
        charges_unchanged=charges_unchanged,
        r1_hi=r1_hi,
        r2_hi=r2_hi,
        mp1=mp1,
        mp2=mp2,
        el_r=el_r,
        max_radical_by_element=max_radical_by_element,
        limit_radicals=L,
        radicals_change_limit=radicals_change_limit,
        passes=passes,
    )


def _pairs_for_bond_order(entries: List[Tuple[str, int]], bond_order: int) -> List[Tuple[str, str]]:
    """Fragment pairs at *bond_order* where each component's [MAX_BOND] allows that order."""
    return [
        (c1, c2)
        for c1, bl1 in entries
        for c2, bl2 in entries
        if bond_order <= bl1 and bond_order <= bl2
    ]


def _append_unchanged_bond_valence_rules(
    rules: List[str],
    *,
    cleavage: bool,
    bond_left: int,
    bond_right: int,
    left_side: str,
    right_side_clean: str,
    left_comps: List[Tuple[str, str, str]],
    right_comps: List[Tuple[str, str, str]],
    rule: str,
    limit_radicals: Optional[int],
    max_radical_by_element: Optional[Dict[str, int]],
    radicals_change_limit: int,
    passes,
) -> None:
    """Valence rules when formal charges are unchanged across a bond-order step (cleavage or formation)."""
    valence_step = (bond_left - bond_right) if cleavage else (bond_right - bond_left)
    slot_indices = _map_valence_slot_indices(left_side, right_side_clean)
    if slot_indices is None:
        if passes(rule):
            rules.append(rule)
        return
    idx1_left, idx2_left, idx1_right, idx2_right = slot_indices

    # Rule 1: valence on the side that gains radical character from the bond step
    if cleavage:
        rule1_vp = f"|^{valence_step}:{idx1_right},{idx2_right}|"
    else:
        rule1_vp = f"|^{valence_step}:{idx1_left},{idx2_left}|"
    rule1 = f"{left_side}>>{_join_rhs_and_valence(right_side_clean, rule1_vp)}"
    if passes(rule1, rule1_vp):
        rules.append(rule1)

    comp1_base_left = f"{left_comps[0][0]}{left_comps[0][1]}"
    comp2_base_left = f"{left_comps[1][0]}{left_comps[1][1]}"
    comp1_base_right = f"{right_comps[0][0]}{right_comps[0][1]}"
    comp2_base_right = f"{right_comps[1][0]}{right_comps[1][1]}"
    same_component_left = comp1_base_left == comp2_base_left
    same_component_right = comp1_base_right == comp2_base_right
    el_l = _elements_two_fragment_side(left_side)
    el_r = _elements_two_fragment_side(right_side_clean)
    caps = max_radical_by_element
    L = limit_radicals
    r1_hi = _valence_slot_loop_hi(el_l[0] if el_l else None, caps, L)
    r2_hi = _valence_slot_loop_hi(el_l[1] if el_l else None, caps, L)
    mp1 = _valence_slot_loop_hi(el_r[0] if el_r else None, caps, L)
    mp2 = _valence_slot_loop_hi(el_r[1] if el_r else None, caps, L)

    # Rule 2: mixed reactant/product valence
    _append_mixed_valence_combinations(
        rules,
        left_side=left_side,
        right_side_clean=right_side_clean,
        cleavage=cleavage,
        formal_charge_redistributes=False,
        idx1_left=idx1_left,
        idx2_left=idx2_left,
        idx1_right=idx1_right,
        idx2_right=idx2_right,
        same_component_left=same_component_left,
        same_component_right=same_component_right,
        charges_unchanged=True,
        r1_hi=r1_hi,
        r2_hi=r2_hi,
        mp1=mp1,
        mp2=mp2,
        el_r=el_r,
        max_radical_by_element=max_radical_by_element,
        limit_radicals=L,
        radicals_change_limit=radicals_change_limit,
        passes=passes,
    )

    # Rule 3: dot-side of the transition (cleavage → dots, formation ← dots)
    if cleavage and bond_right == 0:
        reactant_cap = min(r1_hi, r2_hi)
        max_product_valence = min(mp1, mp2)
        reactant_valence = 1
        while reactant_valence <= reactant_cap:
            product_valence = reactant_valence + 1
            if product_valence > max_product_valence:
                break
            rule3_vp = (
                f"|^{reactant_valence}:{idx1_left},{idx2_left};"
                f"^{product_valence}:{idx1_right},{idx2_right}|"
            )
            r3 = f"{left_side}>>{_join_rhs_and_valence(right_side_clean, rule3_vp)}"
            if passes(r3, rule3_vp):
                rules.append(r3)
            reactant_valence += 1
    elif not cleavage and bond_left == 0:
        product_cap = min(mp1, mp2)
        max_reactant_valence = min(r1_hi, r2_hi)
        product_valence = 1
        while product_valence <= product_cap:
            reactant_valence = product_valence + 1
            if reactant_valence > max_reactant_valence:
                break
            rule3_vp = (
                f"|^{reactant_valence}:{idx1_left},{idx2_left};"
                f"^{product_valence}:{idx1_right},{idx2_right}|"
            )
            r3 = f"{left_side}>>{_join_rhs_and_valence(right_side_clean, rule3_vp)}"
            if passes(r3, rule3_vp):
                rules.append(r3)
            product_valence += 1


def _generate_rules_for_bond_pair(
    bond_left: int,
    bond_right: int,
    left_pairs: List[Tuple[str, str]],
    right_pairs: List[Tuple[str, str]],
    entries: List[Tuple[str, int]],
    bond_limit_lookup: Dict[str, int],
    limit_radicals: Optional[int],
    limit_abs_charges: Optional[int],
    max_radical_by_element: Optional[Dict[str, int]],
    radicals_change_limit: int,
    charge_change_limit: int,
) -> List[str]:
    """Generate all rules for one (reactant bond, product bond) transition."""
    rules: List[str] = []
    bond_left_symbol = BOND_SYMBOLS[bond_left]
    bond_right_symbol = BOND_SYMBOLS[bond_right]

    def passes(rule: str, valence_str: Optional[str] = None) -> bool:
        return _rule_passes_limits(
            rule,
            valence_str,
            limit_radicals,
            limit_abs_charges,
            max_radical_by_element,
            radicals_change_limit,
            charge_change_limit,
        )

    for comp1_left, comp2_left in left_pairs:
        total_charge_left = extract_charge(comp1_left) + extract_charge(comp2_left)

        # Canonical left order (for mapping when reactants differ)
        data1 = extract_component_data(comp1_left)
        data2 = extract_component_data(comp2_left)
        reactants_same = (data1 == data2)
        if data1 and data2:
            e1, c1 = data1
            e2, c2 = data2
            if (e1, c1) > (e2, c2):
                e1, e2, c1, c2 = e2, e1, c2, c1
            r1_charge, r2_charge = c1, c2
        else:
            r1_charge = extract_charge(comp1_left)
            r2_charge = extract_charge(comp2_left)

        if _pair_exceeds_abs_charge_limit(comp1_left, comp2_left, limit_abs_charges):
            continue

        if _skip_noncanonical_fragment_pair_order(comp1_left, comp2_left):
            continue

        for comp1_right, comp2_right in right_pairs:
            if reactants_same and _skip_mirror_same_element_pair(comp1_right, comp2_right):
                continue
            if not reactants_same and _skip_noncanonical_fragment_pair_order(
                comp1_right, comp2_right
            ):
                continue

            if _pair_exceeds_abs_charge_limit(comp1_right, comp2_right, limit_abs_charges):
                continue

            total_charge_right = extract_charge(comp1_right) + extract_charge(comp2_right)
            if abs(total_charge_right - total_charge_left) > charge_change_limit:
                continue

            c1_right = extract_charge(comp1_right)
            c2_right = extract_charge(comp2_right)

            # Charge-conserving: product order must give per-atom charge change at most 1
            order_12_ok = (abs(c1_right - r1_charge) <= 1 and abs(c2_right - r2_charge) <= 1)
            order_21_ok = (abs(c2_right - r1_charge) <= 1 and abs(c1_right - r2_charge) <= 1)
            if not order_12_ok and not order_21_ok:
                continue
            products_same_element = _is_same_element_pair(comp1_right, comp2_right)
            if (
                order_12_ok
                and order_21_ok
                and not reactants_same
                and not products_same_element
            ):
                right_orders = [(comp1_right, comp2_right), (comp2_right, comp1_right)]
            elif order_12_ok:
                right_orders = [(comp1_right, comp2_right)]
            else:
                right_orders = [(comp2_right, comp1_right)]

            for p1_right, p2_right in right_orders:
                if not _product_order_matches_reactant_elements(
                    comp1_left, comp2_left, p1_right, p2_right
                ):
                    continue
                # Create base rule and normalize it
                rule = f"{comp1_left}{bond_left_symbol}{comp2_left}>>{p1_right}{bond_right_symbol}{p2_right}"
                rule = normalize_rule(rule)

                # Extract components from normalized rule
                left_side, right_side = rule.split('>>', 1)
                right_side_clean = right_side.split('|')[0]

                left_comps = _RE_COMP_PAIRED.findall(left_side)
                right_comps = _RE_COMP_PAIRED.findall(right_side_clean)

                if len(left_comps) >= 2 and len(right_comps) >= 2:
                    def _pair_charge(comp_data: Tuple[str, str, str]) -> int:
                        c_str = comp_data[1]
                        return int(c_str.replace('+', '')) if c_str else 0

                    c1_left, c2_left = _pair_charge(left_comps[0]), _pair_charge(left_comps[1])
                    c1_right, c2_right = _pair_charge(right_comps[0]), _pair_charge(right_comps[1])
                    charges_unchanged = (c1_left == c1_right and c2_left == c2_right)

                    if bond_left > bond_right and charges_unchanged:
                        _append_unchanged_bond_valence_rules(
                            rules,
                            cleavage=True,
                            bond_left=bond_left,
                            bond_right=bond_right,
                            left_side=left_side,
                            right_side_clean=right_side_clean,
                            left_comps=left_comps,
                            right_comps=right_comps,
                            rule=rule,
                            limit_radicals=limit_radicals,
                            max_radical_by_element=max_radical_by_element,
                            radicals_change_limit=radicals_change_limit,
                            passes=passes,
                        )
                    elif bond_right > bond_left and charges_unchanged:
                        _append_unchanged_bond_valence_rules(
                            rules,
                            cleavage=False,
                            bond_left=bond_left,
                            bond_right=bond_right,
                            left_side=left_side,
                            right_side_clean=right_side_clean,
                            left_comps=left_comps,
                            right_comps=right_comps,
                            rule=rule,
                            limit_radicals=limit_radicals,
                            max_radical_by_element=max_radical_by_element,
                            radicals_change_limit=radicals_change_limit,
                            passes=passes,
                        )
                    elif bond_left > bond_right:
                        if passes(rule):
                            rules.append(rule)
                        if not charges_unchanged:
                            _append_charge_redistribution_mixed_valence(
                                rules,
                                left_side=left_side,
                                right_side_clean=right_side_clean,
                                left_comps=left_comps,
                                right_comps=right_comps,
                                cleavage=True,
                                charges_unchanged=False,
                                limit_radicals=limit_radicals,
                                max_radical_by_element=max_radical_by_element,
                                radicals_change_limit=radicals_change_limit,
                                passes=passes,
                            )
                    elif bond_right > bond_left:
                        if passes(rule):
                            rules.append(rule)
                        if not charges_unchanged:
                            _append_charge_redistribution_mixed_valence(
                                rules,
                                left_side=left_side,
                                right_side_clean=right_side_clean,
                                left_comps=left_comps,
                                right_comps=right_comps,
                                cleavage=False,
                                charges_unchanged=charges_unchanged,
                                limit_radicals=limit_radicals,
                                max_radical_by_element=max_radical_by_element,
                                radicals_change_limit=radicals_change_limit,
                                passes=passes,
                            )
                else:
                    if passes(rule):
                        rules.append(rule)

    return rules


def _unpack_generate_bond_pair(args: Tuple) -> List[str]:
    return _generate_rules_for_bond_pair(*args)


def _generate_rules_worker_args(
    entries: List[Tuple[str, int]],
    cap: int,
    bond_limit_lookup: Dict[str, int],
    limit_radicals: Optional[int],
    limit_abs_charges: Optional[int],
    max_radical_by_element: Optional[Dict[str, int]],
    radicals_change_limit: int,
    charge_change_limit: int,
) -> List[Tuple]:
    pairs_cache = {bo: _pairs_for_bond_order(entries, bo) for bo in range(cap + 1)}
    args = []
    for bond_left in range(cap + 1):
        for bond_right in range(cap + 1):
            if bond_left == bond_right:
                continue
            if abs(bond_left - bond_right) != MAX_BOND_ORDER_STEP:
                continue
            args.append(
                (
                    bond_left,
                    bond_right,
                    pairs_cache[bond_left],
                    pairs_cache[bond_right],
                    entries,
                    bond_limit_lookup,
                    limit_radicals,
                    limit_abs_charges,
                    max_radical_by_element,
                    radicals_change_limit,
                    charge_change_limit,
                )
            )
    return args


def generate_rules(
    entries: List[Tuple[str, int]],
    limit_radicals: Optional[int] = None,
    limit_abs_charges: Optional[int] = None,
    max_radical_by_element: Optional[Dict[str, int]] = None,
    radicals_change_limit: int = DEFAULT_RADICALS_CHANGE_LIMIT,
    charge_change_limit: int = DEFAULT_CHARGE_CHANGE_LIMIT,
    jobs: Optional[int] = None,
) -> List[str]:
    """Generate all transformation rules (each bond-order transition both ways)."""
    max_valence = max(bond_limit for _, bond_limit in entries) if entries else 0
    max_bond_order = max(BOND_SYMBOLS.keys())
    cap = min(max_valence, max_bond_order)
    bond_limit_lookup = _build_bond_limit_lookup(entries)
    worker_jobs = jobs if jobs is not None else _default_jobs()
    task_args = _generate_rules_worker_args(
        entries,
        cap,
        bond_limit_lookup,
        limit_radicals,
        limit_abs_charges,
        max_radical_by_element,
        radicals_change_limit,
        charge_change_limit,
    )
    if worker_jobs <= 1 or len(task_args) < 2:
        rules: List[str] = []
        for a in task_args:
            rules.extend(_generate_rules_for_bond_pair(*a))
        return rules
    rules: List[str] = []
    for chunk in _parallel_map(_unpack_generate_bond_pair, task_args, worker_jobs):
        rules.extend(chunk)
    return rules


def _light_normalize_rule(rule: str) -> str:
    """Map/element fix and sorted |^...| clauses (no fragment reorder)."""
    return normalize_valence_indices_in_rule(fix_reaction_map_consistency(rule))


def _canonical_rule_for_dedup(rule: str) -> str:
    """One representative per equivalence class: fragment order + valence slots move together."""
    return ensure_map_reading_order(canonicalize_symmetric_pair_rule(rule))


def _dedup_shorthand_key(shorthand: str) -> str:
    """Canonical shorthand for duplicate detection (forward/reverse partners share one key)."""
    partner = partner_shorthand_key(shorthand)
    return min(shorthand, partner)


def _rule_dedup_key(rule: str) -> str:
    """Stable key for duplicate detection (BUILD_BOND_RULES_BASICS § finalize examples)."""
    return _dedup_shorthand_key(rule_to_shorthand(rule))


def _canonicalize_and_filter_rules(
    raw_rules: List[str],
    limit_radicals: Optional[int] = None,
    limit_abs_charges: Optional[int] = None,
    max_radical_by_element: Optional[Dict[str, int]] = None,
    radicals_change_limit: int = DEFAULT_RADICALS_CHANGE_LIMIT,
    charge_change_limit: int = DEFAULT_CHARGE_CHANGE_LIMIT,
    entries: Optional[List[Tuple[str, int]]] = None,
) -> List[str]:
    """Normalize, canonicalize once, filter limits, dedupe by shorthand equivalence key."""
    _bll = _build_bond_limit_lookup(entries) if entries else None
    seen_keys: Set[str] = set()
    out: List[str] = []

    def _accept(rule: str) -> bool:
        return _rule_passes_limits(
            rule,
            _valence_part_from_rule(rule),
            limit_radicals,
            limit_abs_charges,
            max_radical_by_element,
            radicals_change_limit,
            charge_change_limit,
            bond_limit_lookup=_bll,
            entries=entries,
        )

    for raw in raw_rules:
        light = _light_normalize_rule(raw)
        if not _accept(light):
            continue
        canon = _canonical_rule_for_dedup(light)
        if not _accept(canon):
            continue
        key = _rule_dedup_key(canon)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(canon)

    return out


def _process_rule_batch(
    raw_rules: List[str],
    *,
    allow_multi_radical_per_slot: bool,
    allow_multi_radical_per_slot_hydrogen: bool,
    allow_charge_radical_mix: bool,
    limit_radicals: Optional[int],
    limit_abs_charges: Optional[int],
    max_radical_by_element: Optional[Dict[str, int]],
    radicals_change_limit: int,
    charge_change_limit: int,
    entries: List[Tuple[str, int]],
) -> List[str]:
    """Filter, canonicalize, and dedupe one transition family."""
    batch = _filter_rules_by_charge_radical_mix(raw_rules, allow_charge_radical_mix)
    batch = _filter_rules_by_multi_radical_per_slot(
        batch,
        allow_multi_radical_per_slot,
        allow_multi_radical_per_slot_hydrogen=allow_multi_radical_per_slot_hydrogen,
    )
    return _canonicalize_and_filter_rules(
        batch,
        limit_radicals=limit_radicals,
        limit_abs_charges=limit_abs_charges,
        max_radical_by_element=max_radical_by_element or None,
        radicals_change_limit=radicals_change_limit,
        charge_change_limit=charge_change_limit,
        entries=entries,
    )


def _rule_smarts_for_cho_export(rule: str) -> str:
    """Normalized SMARTS string for CHO TOML (sorted |^…|, map-consistent)."""
    return normalize_valence_indices_in_rule(rule)


def _format_cho_toml(sections: Dict[str, List[str]]) -> Tuple[str, int]:
    """Build CHO-style TOML ([unimolecular], rule_000001=\"…\"); return (text, count)."""
    section_order = ("unimolecular", "bond opening", "bond closing")
    ordered = [s for s in section_order if sections.get(s)] + sorted(
        k for k in sections if k not in section_order and sections[k]
    )
    lines: List[str] = []
    total = 0
    for section in ordered:
        rules = sections[section]
        if not rules:
            continue
        lines.append(f"[{section}]")
        for i, rule in enumerate(rules, 1):
            smarts = _rule_smarts_for_cho_export(rule)
            escaped = smarts.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'rule_{i:06d}="{escaped}"')
            total += 1
    text = "\n".join(lines)
    return (text + "\n") if text else "", total


def _cleavage_section_for_rule(rule: str) -> str:
    """Classify a cleavage rule as bond opening (lb>rb) or bond closing (lb<rb)."""
    left, rest = rule.split(">>", 1)
    lb = _bond_order_from_side(left)
    rb = _bond_order_from_side(rest.split("|")[0])
    if lb is not None and rb is not None and lb > rb:
        return "bond opening"
    return "bond closing"


def _format_rules_toml(
    rules: List[str],
    name_display: str,
    extra: List[str],
    *,
    transition_label: str,
) -> Tuple[str, int, int, int]:
    """Build TOML text; return (text, emitted_count, n_forward, n_reverse)."""
    shorthand_count: Dict[str, int] = {}
    rules_by_pair: Dict[str, List[str]] = {}
    emitted_normalized_rules: Set[str] = set()
    n_forward = 0
    n_reverse = 0

    def add_rule(rule: str, buckets: dict) -> None:
        nonlocal n_forward, n_reverse
        rule_n = normalize_valence_indices_in_rule(rule)
        if rule_n in emitted_normalized_rules:
            return
        emitted_normalized_rules.add(rule_n)
        left, right_rest = rule_n.split(">>", 1)
        lb = _bond_order_from_side(left)
        rb = _bond_order_from_side(right_rest.split("|")[0])
        if lb is not None and rb is not None:
            if lb < rb:
                n_forward += 1
            elif lb > rb:
                n_reverse += 1
        base_short = rule_to_shorthand(rule_n)
        key_base = base_short
        if key_base in shorthand_count:
            shorthand_count[key_base] += 1
            key = f"{key_base}_{shorthand_count[key_base]}"
        else:
            shorthand_count[key_base] = 1
            key = key_base
        rule_escaped = rule_n.replace("\\", "\\\\").replace('"', '\\"')
        line = f'{key}="{rule_escaped}"'
        a1a2 = base_short.split("__")[0] if "__" in base_short else ""
        buckets.setdefault(a1a2, []).append(line)

    for rule in rules:
        add_rule(rule, rules_by_pair)

    sections: List[str] = []
    for pair in sorted(rules_by_pair.keys()):
        if rules_by_pair[pair]:
            sections.append(f"[{pair}]")
            sections.extend(rules_by_pair[pair])

    suffix = " ; ".join(extra)
    header = (
        f'["Reaction Rules {name_display} ({transition_label})'
        + (f" ; {suffix}" if suffix else "")
        + '"]'
    )
    return header + "\n" + "\n".join(sections), len(emitted_normalized_rules), n_forward, n_reverse


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description='Build bond transformation rules from component TOML.')
    parser.add_argument('input', help='Input TOML (e.g. bond_rules_input.toml)')
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='Directory for cho_rules.toml and cho_equivalence.toml (default: input file directory)',
    )
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=None,
        metavar='N',
        help='Parallel workers (default: CPU count - 1). Use -j 1 to disable.',
    )
    parser.epilog = (
        'Writes cho_rules.toml (all rules) and cho_equivalence.toml (no .][ bond-break rules). '
        'Parallelism uses threads when run as a script; pass -j 1 to disable.'
    )
    args = parser.parse_args(argv)

    input_file = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_file.resolve().parent
    jobs = args.jobs if args.jobs is not None else _default_jobs()

    (
        entries,
        limit_radicals,
        limit_abs_charges,
        name,
        max_radical_by_element,
        radicals_change_limit,
        charge_change_limit,
        allow_charge_radical_mix,
        allow_multi_radical_per_slot_reorder,
        allow_multi_radical_per_slot_cleavage,
        allow_multi_radical_per_slot_hydrogen,
    ) = parse_input_file(str(input_file))
    if not entries:
        print('No valid entries found in input file.')
        sys.exit(1)

    raw_rules = generate_rules(
        entries,
        limit_radicals=limit_radicals,
        limit_abs_charges=limit_abs_charges,
        max_radical_by_element=max_radical_by_element or None,
        radicals_change_limit=radicals_change_limit,
        charge_change_limit=charge_change_limit,
        jobs=jobs,
    )
    raw_reorder, raw_cleavage = _partition_rules_by_transition(raw_rules)

    common_finalize = dict(
        allow_multi_radical_per_slot_hydrogen=allow_multi_radical_per_slot_hydrogen,
        allow_charge_radical_mix=allow_charge_radical_mix,
        limit_radicals=limit_radicals,
        limit_abs_charges=limit_abs_charges,
        max_radical_by_element=max_radical_by_element,
        radicals_change_limit=radicals_change_limit,
        charge_change_limit=charge_change_limit,
        entries=entries,
    )
    rules_reorder = _process_rule_batch(
        raw_reorder,
        allow_multi_radical_per_slot=allow_multi_radical_per_slot_reorder,
        **common_finalize,
    )
    rules_cleavage = _process_rule_batch(
        raw_cleavage,
        allow_multi_radical_per_slot=allow_multi_radical_per_slot_cleavage,
        **common_finalize,
    )

    rules_equivalence = [r for r in rules_reorder if not _rule_has_bond_break(r)]
    n_equiv_skipped = len(rules_reorder) - len(rules_equivalence)

    cho_rules_sections: Dict[str, List[str]] = {"unimolecular": list(rules_reorder)}
    opening: List[str] = []
    closing: List[str] = []
    for rule in rules_cleavage:
        if _cleavage_section_for_rule(rule) == "bond opening":
            opening.append(rule)
        else:
            closing.append(rule)
    if opening:
        cho_rules_sections["bond opening"] = opening
    if closing:
        cho_rules_sections["bond closing"] = closing

    cho_rules_path = output_dir / "cho_rules.toml"
    cho_equiv_path = output_dir / "cho_equivalence.toml"
    rules_text, n_rules = _format_cho_toml(cho_rules_sections)
    equiv_text, n_equiv = _format_cho_toml({"unimolecular": rules_equivalence})

    output_dir.mkdir(parents=True, exist_ok=True)
    cho_rules_path.write_text(rules_text, encoding="utf-8")
    cho_equiv_path.write_text(equiv_text, encoding="utf-8")

    msg = (
        f"Wrote {n_rules} rules to {cho_rules_path} "
        f"(unimolecular: {len(rules_reorder)}, bond opening: {len(opening)}, "
        f"bond closing: {len(closing)})"
    )
    msg += (
        f"; {n_equiv} equivalence rules to {cho_equiv_path}"
    )
    if n_equiv_skipped:
        msg += f" (skipped {n_equiv_skipped} reorder rule(s) with ].[ bond break)"
    print(msg)


if __name__ == '__main__':
    main()
