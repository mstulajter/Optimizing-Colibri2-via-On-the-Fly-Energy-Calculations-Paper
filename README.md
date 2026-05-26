Python code, Colibri2 launch scripts, and reaction networks for the paper ‘Optimizing Colibri2 via On-the-Fly Energy Calculations: Applications to Atmospheric Radical Chemistry’. This includes generating reaction networks with Colibri2, merging molecular energies, making networks reversible, and adding karc edge weights. Networks are stored in GraphML format (`*.graphml.bz2`).

The repository includes:
- **Exhaustive RN** — explore full stoichiometry chemical space
- **Bounded RN** — families of step- and ΔE-bounded networks
- shared Python tools in `bin/` (equivalence, energy merge, path finding, GraphML helpers)
- CHO rule and equivalence TOML files (generated in `bin/`, preprocessed with Colibri2 `rulesetpreprocess.py`, then copied into each study’s `bin/`)

## Repository Layout

**`bin/`** — shared Python scripts (used by both studies)

- `bin/build_bond_rules.py` — generate `cho_rules.toml` and `cho_equivalence.toml` from `bond_rules_input.toml`
- `bin/bond_rules_input.toml` — component valence / radical limits for rule generation
- `bin/cho_rules.toml`, `bin/cho_equivalence.toml` — raw CHO rule sets (pre-Colibri2 preprocess)
- `bin/graphml_utils.py` — GraphML I/O, SMILES canonicalization, KARC edge-weight formula
- `bin/find_equivalent_mols.py` — equivalence classes from bond opening/closing rules
- `bin/extract_mols.py` — fragment SMILES from GraphML
- `bin/merge_energies.py` — merge computed energies with equivalence mapping
- `bin/make_reversable.py` — add reverse edges to exported networks
- `bin/reaction_path_finder.py` — Dijkstra + BFS path search on weighted graphs

**Exhaustive RN/**

- `Exhaustive RN/bin/add_edge_weights.py` — weights from external `smiles,energy` CSV → `*_graph_weighted.graphml.bz2`
- `Exhaustive RN/bin/initialize3`, `submit3`, `compute3`, `network3`, `export3`, `collect3` — Colibri2 v3 wrappers
- `Exhaustive RN/bin/settings.toml`, `cho_rules_processed.toml`, `cho_equivalence_processed.toml`
- `Exhaustive RN/<stoich>/` — `*_graph_reversable.graphml.bz2`, `*_graph_weighted.graphml.bz2`, `*_molecules_merged_energies.csv`

**Bounded RN/**

- `Bounded RN/bin/add_edge_weights.py` — batch weighting from node `energy` in GraphML; adds reverse edges → `*_wk.graphml.bz2`
- `Bounded RN/bin/build_master_equivalence.py` — master SMILES list + project-wide equivalence
- `Bounded RN/bin/graph_summary_table.py` — per-stoichiometry summary tables
- `Bounded RN/bin/initialize5`, `proton5`, `network5`, `equivalence5`, `compute5`, `reaper5`, `export5`, `pipeline5` — Colibri2 v5 wrappers
- `Bounded RN/bin/env.params5`, `settings.toml`, CHO TOML files
- `Bounded RN/<stoich>/` — many `*_steps<N>_dE<B>_wk.graphml.bz2` networks

**Outputs only from Bounded RN post-processing** (written under `Bounded RN/` when you run those scripts):

- `bounded_master_molecules.csv`, `bounded_master_molecules_equivalent.csv`, `bounded_master_molecules_unique.csv`
- `<stoich>_graph_weight_counts.csv`, `<stoich>_graph_summary_table.csv`

Exhaustive RN does not generate these; it keeps energies and weighted graphs inside each `<stoich>/` folder.

`add_edge_weights.py` is **not** shared: the two copies differ (CSV-based vs in-graph energies, single file vs batch, `_weighted` vs `_wk` naming).

## Requirements

Core dependencies include:
- `numpy`, `pandas`, `rdkit`, `openbabel`, `toml`
- `networkx`, `networkit` (reversible graphs and path finding)
- `tqdm` (optional)
- Colibri2, PostgreSQL, and Redis for network generation runs

Run shared tools as `python3 bin/<script>.py` from the repo root, or `python3 ../bin/<script>.py` from inside a study folder. Study-specific scripts stay in `Exhaustive RN/bin/` or `Bounded RN/bin/`.

## CHO reaction rules

Generate raw rule TOML from component specs:

```bash
python3 bin/build_bond_rules.py bin/bond_rules_input.toml
```

This writes `cho_rules.toml` (all rules) and `cho_equivalence.toml` (bond-order reorder only) in the same directory as the input file (by default `bin/`).

Files are then run through **`rulesetpreprocess.py`** in Colibri2 to produce the processed rule sets used by network generation (e.g. `cho_rules_processed.toml`, `cho_equivalence_processed.toml` in each study’s `bin/`).

## Workflow — Exhaustive RN

### 0) Set variables

```bash
PORT_SQL=1090
PORT_REDIS=1091

SET_FILE="/absolute/path/to/Exhaustive RN/bin/settings.toml"
RULES_FILE="/absolute/path/to/Exhaustive RN/bin/cho_rules_processed.toml"
EQUIV_FILE="/absolute/path/to/Exhaustive RN/bin/cho_equivalence_processed.toml"
INITIAL_FLASK="..."

RN="O3"
```

### 1) Build and export network with Colibri2

```bash
cd "Exhaustive RN/bin"
./initialize3 "$PORT_SQL" "$PORT_REDIS" "$SET_FILE" "$INITIAL_FLASK"
./network3 "$PORT_SQL" "$PORT_REDIS" "$SET_FILE" "$RULES_FILE"
./export3 "$PORT_SQL" "$SET_FILE" "../${RN}/${RN}.graphml.bz2"
```

### 2) Molecule extraction, equivalence, and energy merge

```bash
cd "Exhaustive RN"
python3 ../bin/extract_mols.py -i "${RN}/${RN}.graphml.bz2" -o "${RN}/${RN}_molecules.csv"

python3 ../bin/find_equivalent_mols.py \
  -i "${RN}/${RN}_molecules.csv" \
  -r bin/cho_equivalence_processed.toml \
  -o "${RN}/${RN}_molecules_equivalent.csv" \
  -u "${RN}/${RN}_molecules_unique.csv"

# bin/submit3, bin/compute3, bin/collect3 ...

python3 ../bin/merge_energies.py \
  "${RN}/${RN}_molecules_unique_energies.csv" \
  "${RN}/${RN}_molecules_equivalent.csv" \
  "${RN}/${RN}_molecules_merged_energies.csv"
```

### 3) Reversible weighted graph

```bash
python3 ../bin/make_reversable.py \
  "${RN}/${RN}.graphml.bz2" \
  -o "${RN}/${RN}_graph_reversable.graphml.bz2"

python3 bin/add_edge_weights.py \
  "${RN}/${RN}_graph_reversable.graphml.bz2" \
  "${RN}/${RN}_molecules_merged_energies.csv" \
  -o "${RN}/${RN}_graph_weighted.graphml.bz2"
```

### 4) Reaction path search

```bash
python3 ../bin/reaction_path_finder.py \
  -gfile "${RN}/${RN}_graph_weighted.graphml.bz2" \
  -r "reactant_smiles" \
  -p "product_smiles" \
  --delta 0
```

## Workflow — Bounded RN

Configure `Bounded RN/bin/env.params5`, then Colibri2 v5, e.g.:

```bash
cd "Bounded RN/bin"
./pipeline5
```

Post-process (node energies already in GraphML):

```bash
cd "Bounded RN"
python3 bin/add_edge_weights.py
python3 bin/build_master_equivalence.py -j 0
python3 bin/graph_summary_table.py
```

Reaction paths on a weighted network:

```bash
python3 ../bin/reaction_path_finder.py \
  -gfile "${RN}/${RN}_steps12_dE4_wk.graphml.bz2" \
  -r "reactant_smiles" \
  -p "product_smiles" \
  --delta 0
```

Network names: `*_steps<N>_dE<B>_wk.graphml.bz2` (`_wk` = weighted + reversible).

## Input/Output Notes

- GraphML, usually `.graphml.bz2`.
- Nodes: `smiles`; Bounded weighted graphs also have `energy` on nodes.
- Edges: `rule`; weighted graphs add `weight` (KARC-style from node energies).
- Exhaustive RN weights from a merged energy CSV; Bounded RN uses energies stored on nodes during Colibri2 export.
