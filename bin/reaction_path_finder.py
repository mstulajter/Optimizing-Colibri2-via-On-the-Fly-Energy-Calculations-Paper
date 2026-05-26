#!/usr/bin/env python3
"""
Find reaction paths in a weighted GraphML network.

Uses Dijkstra for the lowest-cost path, then BFS for all simple paths up to
(shortest length + delta). Writes a ranked report to a text file.

Authors:
    Miko Stulajter

Version 2.0.0

Usage:
    reaction_path_finder.py -gfile NETWORK.graphml[.bz2] -r REACTANTS -p PRODUCTS \\
        [--delta N] [--hide-paths] [--verbose]
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

from graphml_utils import canonicalize_multimol_smiles, open_graphml

RDLogger.DisableLog("rdApp.warning")


def _parse_formula(formula: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
        elem, num = match.group(1), match.group(2)
        counts[elem] += int(num) if num else 1
    return dict(counts)


def _merge_formulas(formulas: list[dict[str, int]]) -> str:
    total: dict[str, int] = defaultdict(int)
    for formula in formulas:
        for elem, count in formula.items():
            total[elem] += count
    order = ["C", "H"] + sorted(k for k in total if k not in ("C", "H"))
    parts = []
    for elem in order:
        if elem in total and total[elem] > 0:
            parts.append(f"{elem}{total[elem]}" if total[elem] > 1 else elem)
    return "".join(parts)


def molecular_formula_from_smiles(smiles: str) -> str:
    from rdkit import Chem

    parts = [p.strip() for p in smiles.split(".")]
    formulas = []
    for part in parts:
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return "unknown"
        formulas.append(_parse_formula(rdMolDescriptors.CalcMolFormula(mol)))
    if not formulas:
        return "unknown"
    return _merge_formulas(formulas)


def make_output_path(rn: str, delta: int) -> str:
    tag = random.randint(10_000_000, 99_999_999)
    return f"{rn}_reaction_path_delta{delta}_{tag}.txt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lowest-cost path (Dijkstra) plus paths within length cutoff (BFS).",
    )
    parser.add_argument("-gfile", required=True, type=Path, help="GraphML file")
    parser.add_argument("-r", required=True, help="Reactant SMILES")
    parser.add_argument("-p", required=True, help="Product SMILES")
    parser.add_argument(
        "--delta",
        type=int,
        default=0,
        help="Extra steps beyond shortest path (default: 0)",
    )
    parser.add_argument(
        "--hide-paths",
        action="store_true",
        help="Omit Path1, Path2, ... lines from output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print BFS progress",
    )
    return parser.parse_args()


def are_equivalent_smiles(smiles1: str, smiles2: str) -> bool:
    canon1 = canonicalize_multimol_smiles(smiles1)
    canon2 = canonicalize_multimol_smiles(smiles2)
    return canon1 is not None and canon2 is not None and canon1 == canon2


def find_node_for_label(node_labels: dict[str, str], label: str) -> str:
    if label in node_labels.values():
        for node, lbl in node_labels.items():
            if lbl == label:
                return node

    for perm in {".".join(p) for p in itertools.permutations(label.split("."))}:
        for node, lbl in node_labels.items():
            if lbl == perm:
                return node

    for node, lbl in node_labels.items():
        if are_equivalent_smiles(label, lbl):
            return node

    print(f"ERROR: Label '{label}' not found in the graph.", file=sys.stderr)
    sys.exit(1)


def get_edge_weight(edge_data: dict, weight_key: str | None) -> float:
    if weight_key:
        weight = edge_data.get(weight_key)
        if weight is not None:
            try:
                weight_val = float(weight)
                if weight_val == weight_val and abs(weight_val) != float("inf"):
                    return weight_val
            except (ValueError, TypeError):
                pass
    return 1.0


def compute_path_cost(
    path_nodes: list[str], graph: nx.Graph, weight_key: str | None
) -> float:
    total = 0.0
    for i in range(len(path_nodes) - 1):
        total += get_edge_weight(graph.edges[path_nodes[i], path_nodes[i + 1]], weight_key)
    return total


def check_weight_key(graph: nx.Graph) -> str | None:
    for _, _, edge_data in graph.edges(data=True):
        if "weight" in edge_data:
            try:
                float(edge_data["weight"])
                return "weight"
            except (ValueError, TypeError):
                pass
    return None


def check_rule_key(graph: nx.Graph) -> str | None:
    for _, _, edge_data in graph.edges(data=True):
        if "rule" in edge_data:
            return "rule"
    return None


def crop_graph_to_max_length(
    graph: nx.Graph, source: str, target: str, max_length: int
) -> nx.Graph:
    source_nodes = set()
    if source in graph:
        source_nodes = set(
            nx.single_source_shortest_path_length(graph, source, cutoff=max_length)
        )
    target_nodes = set()
    if target in graph:
        target_nodes = set(
            nx.single_source_shortest_path_length(graph, target, cutoff=max_length)
        )
    return graph.subgraph(source_nodes | target_nodes).copy()


def bfs_paths_up_to_length(
    graph: nx.Graph,
    source: str,
    target: str,
    max_length: int,
    weight_key: str | None,
    node_labels: dict[str, str],
    verbose: bool = False,
) -> list[tuple[float, str, list[str]]]:
    path_costs: list[tuple[float, str, list[str]]] = []
    seen_paths: set[tuple[str, ...]] = set()
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])
    paths_found = 0

    while queue:
        node, path_tuple = queue.popleft()
        path_list = list(path_tuple)
        steps = len(path_list) - 1

        if node == target:
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                cost = compute_path_cost(path_list, graph, weight_key)
                path_str = " --> ".join(node_labels[n] for n in path_list)
                path_costs.append((cost, path_str, path_list))
                paths_found += 1
                if verbose and paths_found % 100 == 0:
                    print(f"  BFS found {paths_found} paths so far...", flush=True)
            continue

        if steps >= max_length:
            continue

        path_set = set(path_list)
        for neighbor in graph.neighbors(node):
            if neighbor not in path_set:
                queue.append((neighbor, path_tuple + (neighbor,)))

    path_costs.sort(key=lambda item: (item[0], item[1]))
    return path_costs


def extract_edge_info(
    graph: nx.Graph,
    path_nodes: list[str],
    rule_key: str | None,
    weight_key: str | None,
) -> list[dict[str, object]]:
    edges = []
    for i in range(len(path_nodes) - 1):
        edge_data = graph.edges[path_nodes[i], path_nodes[i + 1]]
        edges.append(
            {
                "rule": edge_data.get(rule_key, "") if rule_key else "",
                "weight": get_edge_weight(edge_data, weight_key),
            }
        )
    return edges


def analyze_paths(
    graph: nx.Graph,
    path_costs: list[tuple[float, str, list[str]]],
    rule_key: str | None,
    weight_key: str | None,
) -> list[dict[str, object]] | None:
    if not path_costs:
        return None

    cost_groups: dict[float, list[tuple[str, list[str]]]] = defaultdict(list)
    for cost, path_str, path_nodes in path_costs:
        cost_groups[cost].append((path_str, path_nodes))

    results = []
    for cost in sorted(cost_groups):
        paths_in_group = cost_groups[cost]
        all_edges = []
        path_costs_list = []
        path_lengths_list = []
        for path_str, path_nodes in paths_in_group:
            edges = extract_edge_info(graph, path_nodes, rule_key, weight_key)
            all_edges.append((path_str, edges))
            path_costs_list.append(cost)
            path_lengths_list.append(len(edges))
        results.append(
            {
                "cost": cost,
                "paths": all_edges,
                "path_costs": path_costs_list,
                "path_lengths": path_lengths_list,
            }
        )
    return results


def format_duplicity(path_costs: list[float]) -> str:
    if not path_costs:
        return "None"
    cost_counts: dict[float, int] = defaultdict(int)
    for cost in path_costs:
        cost_counts[cost] += 1
    return ", ".join(
        f"{cost:.9f} ({count}x)"
        for cost, count in sorted(cost_counts.items(), reverse=True)
    )


def format_length_duplicity(path_lengths: list[int]) -> str:
    if not path_lengths:
        return "None"
    length_counts: dict[int, int] = defaultdict(int)
    for length in path_lengths:
        length_counts[length] += 1
    return ", ".join(
        f"Length {length} ({count}x)"
        for length, count in sorted(length_counts.items(), reverse=True)
    )


def print_results(
    analysis_results: list[dict[str, object]] | None, hide_paths: bool = False
) -> None:
    if not analysis_results:
        print("No paths found.")
        return

    for rank, result in enumerate(analysis_results, 1):
        cost = result["cost"]
        paths = result["paths"]
        path_costs = result["path_costs"]
        path_lengths = result["path_lengths"]
        print(f"{rank}: Cost: {cost:.9f}")
        if not hide_paths:
            for index, (path_str, _) in enumerate(paths, 1):
                print(f"    Path{index}: {path_str}")
        print("  Edge steps:")
        seen_rules: set[str] = set()
        for _path_str, edges in paths:
            for edge in edges:
                rule = edge["rule"]
                weight = edge["weight"]
                if rule and weight != 0.0 and rule not in seen_rules:
                    print(f"    {rule} --> ")
                    seen_rules.add(rule)
        print(f"  Duplicity:\n    {format_duplicity(path_costs)}")
        print(f"  Path Length Duplicity:\n    {format_length_duplicity(path_lengths)}\n")


def load_graphml(path: Path) -> nx.Graph:
    if not path.is_file():
        print(f"Graph file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix == ".bz2":
        with open_graphml(path, "rt") as handle:
            return nx.read_graphml(handle)
    with path.open(encoding="utf-8") as handle:
        return nx.read_graphml(handle)


def _run(args: argparse.Namespace, outpath: str, rn: str) -> None:
    print(f"# Output file: {outpath}\n")

    graph = load_graphml(args.gfile)
    node_labels = {node: data.get("smiles", node) for node, data in graph.nodes(data=True)}
    rule_key = check_rule_key(graph)
    if not rule_key:
        print("WARNING: No rule key found in graph. Using empty rules.")
        rule_key = None
    weight_key = check_weight_key(graph)

    reactant_node = find_node_for_label(node_labels, args.r)
    product_node = find_node_for_label(node_labels, args.p)

    print(f"Reactant: {args.r} (node: {reactant_node})")
    print(f"Product: {args.p} (node: {product_node})")

    try:
        path_nodes = nx.dijkstra_path(
            graph, reactant_node, product_node, weight=weight_key
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print("No path found between reactant and product.")
        return

    path_length_steps = len(path_nodes) - 1
    path_cost = compute_path_cost(path_nodes, graph, weight_key)
    print(f"\nDijkstra shortest path: {path_length_steps} steps, cost = {path_cost:.9f}")
    print(f"  Path: {' --> '.join(node_labels[n] for n in path_nodes)}\n")

    cutoff = path_length_steps + args.delta
    print(f"BFS cutoff depth: {path_length_steps} + {args.delta} = {cutoff} steps\n")

    cropped = crop_graph_to_max_length(graph, reactant_node, product_node, cutoff)
    if cropped.number_of_nodes() < graph.number_of_nodes():
        print(
            f"Graph optimization: {graph.number_of_nodes()} -> "
            f"{cropped.number_of_nodes()} nodes\n"
        )

    path_costs = bfs_paths_up_to_length(
        cropped,
        reactant_node,
        product_node,
        cutoff,
        weight_key,
        node_labels,
        verbose=args.verbose,
    )
    print(f"BFS found {len(path_costs)} path(s) with length <= {cutoff}\n")

    if not path_costs:
        print("No paths found within cutoff.")
        return

    print_results(analyze_paths(graph, path_costs, rule_key, weight_key), args.hide_paths)


def main() -> int:
    args = parse_arguments()
    rn = molecular_formula_from_smiles(args.r)
    outpath = make_output_path(rn, args.delta)

    with open(outpath, "w", encoding="utf-8") as outfile:
        real_stdout = sys.stdout
        sys.stdout = outfile
        try:
            _run(args, outpath, rn)
        finally:
            sys.stdout = real_stdout

    print(f"Output saved to {os.path.abspath(outpath)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
