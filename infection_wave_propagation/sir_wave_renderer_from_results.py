"""
Infection Wave Propagation Model - 2nd Iteration

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

BASE_DIR = Path(__file__).resolve().parent
GRAPH_DIR = BASE_DIR / "grafuri_naturale_si_generate_curatate"
RESULTS_DIR = BASE_DIR / "simulation_results"
RAW_RESULTS = RESULTS_DIR / "raw_results.csv"
OUTPUT_DIR = BASE_DIR / "infection_wave_2nd_iteration_outputs"

NETWORKS: Dict[str, str] = {
    "graf_generat_100_99": "99muchii_100noduri_generat_output.csv",
    "graf_real_100_99": "99muchii_100noduri_natural_output.csv",
    "graf_generat_275_100": "275muchii_100noduri_generat_output.csv",
    "graf_real_100_275": "275muchii_100noduri_natural_output.csv",
    "graf_real_100_1000": "1000muchii_100node_natural_output.csv",
    "graf_generat_100_1000": "1000muchii_100noduri_generat_output.csv",
}

SELECTED_PROBABILITIES = [0.01, 0.10, 0.50]


SELECTED_SOURCES: Dict[str, int] = {
    "graf_generat_100_99": 79,
    "graf_real_100_99": 1,
    "graf_generat_275_100": 59,
    "graf_real_100_275": 1,
    "graf_real_100_1000": 3,
    "graf_generat_100_1000": 34,
}


def p_to_folder(p: float) -> str:
    
    if math.isclose(p, 0.01):
        return "Prob_0.01"
    return f"Prob_{p:g}"


def p_to_file_part(p: float) -> str:
    
    if math.isclose(p, 0.01):
        return "0.01"
    return f"{p:g}"


def load_graph(network: str) -> nx.Graph:

    if network not in NETWORKS:
        raise ValueError(f"Unknown network: {network}. Available: {list(NETWORKS)}")

    graph_path = GRAPH_DIR / NETWORKS[network]
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    graph = nx.Graph()
    with graph_path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    for line in lines[1:]:  # same behavior as runner.py
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                graph.add_edge(int(parts[0]), int(parts[1]))
            except ValueError:
                continue

    return graph


def choose_representative_run(network: str, p: float, source: int) -> int:
    
    raw = pd.read_csv(RAW_RESULTS)
    subset = raw[(raw["retea"] == network) & (raw["p"].round(4) == round(p, 4)) & (raw["sursa"] == source)]
    if subset.empty:
        raise ValueError(f"No raw_results rows for network={network}, p={p}, source={source}")

    mean_value = subset["nr_infectati"].mean()
    closest = subset.iloc[(subset["nr_infectati"] - mean_value).abs().argsort().iloc[0]]
    return int(closest["run_id"])


def find_nomenclator_file(network: str, p: float, run_id: int) -> Path:
    prob_folder = p_to_folder(p)
    prob_part = p_to_file_part(p)
    folder = RESULTS_DIR / network / prob_folder
    if not folder.exists():
        raise FileNotFoundError(f"Probability folder not found: {folder}")

    matches = list(folder.glob(f"Graf_experiment_*_p{prob_part}_run{run_id}.csv"))
    if not matches:
        raise FileNotFoundError(f"No detailed result file for {network}, p={p}, run={run_id} in {folder}")
    return matches[0]


def load_infection_ticks(nomenclator_file: Path, source: int) -> Dict[int, int]:
    
    df = pd.read_csv(nomenclator_file)
    col = f"Sursa_{source}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in {nomenclator_file.name}")

    infection_tick: Dict[int, int] = {source: 0}
    for _, row in df.iterrows():
        tick_label = str(row["Clock Tick"]).replace("t", "")
        try:
            tick = int(tick_label)
        except ValueError:
            continue

        cell = row[col]
        if pd.isna(cell) or str(cell).strip() == "":
            continue

        for value in str(cell).split(","):
            value = value.strip()
            if value:
                infection_tick[int(float(value))] = tick

    return infection_tick


def render_wave(network: str, p: float, source: int, run_id: int | str = "auto", output_dir: Path = OUTPUT_DIR) -> Path:
    graph = load_graph(network)
    if source not in graph.nodes:
        raise ValueError(f"Source node {source} is not present in network {network}")

    if run_id == "auto":
        run_id = choose_representative_run(network, p, source)
    run_id = int(run_id)

    nomenclator_file = find_nomenclator_file(network, p, run_id)
    infection_tick = load_infection_ticks(nomenclator_file, source)
    max_tick = max(infection_tick.values()) if infection_tick else 0

    pos = nx.spring_layout(graph, seed=42, k=0.55)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"wave_{network}_p{p_to_file_part(p)}_source{source}_run{run_id}.gif"

    fig, ax = plt.subplots(figsize=(9, 7))

    def draw(frame: int):
        ax.clear()
        susceptible: List[int] = []
        infected_now: List[int] = []
        recovered: List[int] = []

        for node in graph.nodes:
            tick = infection_tick.get(node)
            if tick is None or tick > frame:
                susceptible.append(node)
            elif tick == frame:
                infected_now.append(node)
            else:
                recovered.append(node)

        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.18, width=0.7)
        nx.draw_networkx_nodes(graph, pos, nodelist=susceptible, node_color="lightgray", node_size=80, ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=recovered, node_color="orange", node_size=90, ax=ax)
        nx.draw_networkx_nodes(graph, pos, nodelist=infected_now, node_color="red", node_size=130, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=6, ax=ax)

        ax.set_title(
            f"Infection Wave Propagation - {network}\n"
            f"p={p}, source={source}, run={run_id}, tick={frame}/{max_tick}, "
            f"infected total={len([n for n, t in infection_tick.items() if t <= frame])}",
            fontsize=11,
        )
        ax.axis("off")

    frames = list(range(0, max_tick + 1))
    if len(frames) == 1:
        frames = [0, 0]  # Pillow needs at least a small animation sequence

    animation = FuncAnimation(fig, draw, frames=frames, interval=900, repeat=True)
    animation.save(out_path, writer=PillowWriter(fps=1))
    plt.close(fig)

    print(f"[OK] Saved {out_path.name}")
    return out_path


def render_batch() -> List[Path]:
    outputs: List[Path] = []
    for network in NETWORKS:
        source = SELECTED_SOURCES[network]
        for p in SELECTED_PROBABILITIES:
            outputs.append(render_wave(network=network, p=p, source=source, run_id="auto"))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SIR infection wave from existing project simulation results.")
    parser.add_argument("--batch", action="store_true", help="Render 18 GIFs: 6 networks x 3 probabilities.")
    parser.add_argument("--network", choices=list(NETWORKS.keys()), help="Network name from project results.")
    parser.add_argument("--p", type=float, help="Probability used in the project, e.g. 0.01, 0.1, 0.5.")
    parser.add_argument("--source", type=int, help="Source/start node.")
    parser.add_argument("--run-id", default="auto", help="Existing run_id or 'auto' for representative run.")
    args = parser.parse_args()

    if args.batch:
        render_batch()
        return

    if args.network is None or args.p is None or args.source is None:
        parser.error("Use --batch or provide --network, --p and --source.")

    render_wave(network=args.network, p=args.p, source=args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
