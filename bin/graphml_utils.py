"""Shared helpers for GraphML I/O and SMILES canonicalization."""

from __future__ import annotations

import bz2
import re
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO

import numpy as np
from rdkit import Chem

GRAPHML_NS = "{http://graphml.graphdrawing.org/xmlns}"
NODE_OPEN_RE = re.compile(r'<node\s+id="([^"]+)"')
EDGE_OPEN_RE = re.compile(r'<edge\s+source="([^"]+)"\s+target="([^"]+)"')
DATA_D0_RE = re.compile(r'<data key="d0">(.*?)</data>')
DATA_D1_RE = re.compile(r'<data key="d1">(.*?)</data>')

WEIGHTED_MARKER = "_wk"


def is_weighted_graphml(path: Path) -> bool:
    """True if the filename denotes a weighted (wk) network output."""
    return WEIGHTED_MARKER in path.name and path.name.endswith((".graphml", ".graphml.bz2"))


def open_graphml(path: str | Path, mode: str = "rb") -> BinaryIO | TextIO:
    """Open a GraphML file, transparently handling ``.bz2`` compression."""
    path = Path(path)
    if path.suffix == ".bz2":
        if "t" in mode:
            return bz2.open(path, mode, encoding="utf-8")
        return bz2.open(path, mode)
    if "t" in mode:
        return path.open(mode, encoding="utf-8")
    return path.open(mode)


def iter_graphml_lines(path: str | Path) -> Iterator[str]:
    """Yield decoded lines from a GraphML file."""
    with open_graphml(path, "rb") as handle:
        for line_bytes in handle:
            yield line_bytes.decode("utf-8", errors="replace")


def unescape_xml(text: str) -> str:
    return text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def escape_xml_for_graphml(text: str, escape_gt: bool) -> str:
    """Escape text for GraphML; optionally encode ``>`` as ``&gt;``."""
    text = text.replace("&", "&amp;").replace("<", "&lt;")
    return text.replace(">", "&gt;") if escape_gt else text


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_multimol_smiles(smiles: str) -> str | None:
    parts = smiles.split(".")
    canonical_parts: list[str] = []
    for part in parts:
        canonical = canonicalize_smiles(part)
        if canonical is None:
            return None
        canonical_parts.append(canonical)
    canonical_parts.sort()
    return ".".join(canonical_parts)


def compute_edge_weight(source_energy: float, target_energy: float) -> float:
    """KARC-style edge weight from source and target node energies."""
    alpha = 1.0
    kappa = 5.0
    energy_diff = target_energy - source_energy
    sigmoid = (1.0 - np.exp(-kappa * energy_diff**2)) / (
        1.0 + np.exp(-kappa * energy_diff**2)
    )
    return float(np.sqrt(energy_diff**2 + alpha**2 * sigmoid))


def count_nodes_and_edges(path: str | Path) -> tuple[int, int]:
    nodes = edges = 0
    for line in iter_graphml_lines(path):
        if NODE_OPEN_RE.search(line):
            nodes += 1
        elif EDGE_OPEN_RE.search(line):
            edges += 1
    return nodes, edges


def default_weighted_output(input_path: Path) -> Path:
    """Exhaustive RN: ``*_reversable`` → ``*_weighted.graphml.bz2``."""
    name = input_path.name
    if name.endswith("_reversable.graphml.bz2"):
        return input_path.with_name(
            name.replace("_reversable.graphml.bz2", "_weighted.graphml.bz2")
        )
    if name.endswith(".graphml.bz2"):
        stem = name[: -len(".graphml.bz2")]
        return input_path.with_name(f"{stem}_weighted.graphml.bz2")
    return input_path.with_name(f"{input_path.stem}_weighted.graphml.bz2")


def default_wk_output(input_path: Path) -> Path:
    """Bounded RN: append ``_wk`` before ``.graphml.bz2``."""
    name = input_path.name
    if name.endswith(".graphml.bz2"):
        stem = name[: -len(".graphml.bz2")]
        if stem.endswith(WEIGHTED_MARKER):
            return input_path
        return input_path.with_name(f"{stem}{WEIGHTED_MARKER}.graphml.bz2")
    if name.endswith(".graphml"):
        stem = name[: -len(".graphml")]
        if stem.endswith(WEIGHTED_MARKER):
            return input_path.with_name(f"{stem}.graphml.bz2")
        return input_path.with_name(f"{stem}{WEIGHTED_MARKER}.graphml.bz2")
    return input_path.with_name(f"{input_path.stem}{WEIGHTED_MARKER}.graphml.bz2")


def default_reversable_output(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".graphml.bz2"):
        stem = name[: -len(".graphml.bz2")]
        return input_path.with_name(f"{stem}_reversable.graphml.bz2")
    return input_path.with_name(f"{input_path.stem}_reversable.graphml.bz2")


def default_molecules_csv(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".graphml.bz2"):
        stem = name[: -len(".graphml.bz2")]
        return input_path.with_name(f"{stem}_molecules.csv")
    return input_path.with_name(f"{input_path.stem}_molecules.csv")


def strip_xml_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[-1]
    return tag
