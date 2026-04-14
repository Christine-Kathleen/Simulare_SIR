import csv
import networkx as nx
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr


# LOAD GRAPH
def load_graph_from_csv(file_path):
    G = nx.Graph()
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            G.add_edge(u, v)
    return G


# CENTRALITIES
def compute_centralities(G):
    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    katz = nx.katz_centrality(G, alpha=0.1)

    degree = dict(sorted(degree.items()))
    closeness = dict(sorted(closeness.items()))
    katz = dict(sorted(katz.items()))

    return degree, closeness, katz


# SAVE CENTRALITIES
def save_centrality(data, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Node", "Value"])
        for node, value in data.items():
            writer.writerow([node, value])


# LOAD STOCHASTIC + AGGREGATE
def load_stochastic(file_path, graph_name, total_nodes):
    data = defaultdict(list)

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["retea"] != graph_name:
                continue

            node = int(row["sursa"])
            infected = int(row["nr_infectati"])
            data[node].append(infected)

    result = []
    for node in range(total_nodes):
        if node in data:
            avg = sum(data[node]) / len(data[node])
        else:
            avg = 0
        result.append(avg)

    return result


# MAIN
def main():
    print("=== START FULL PIPELINE ===")

    graph_file = "grafuri_naturale_si_generate_curatate/99muchii_100noduri_generat_output.csv"
    graph_name = "graf_generat_100_99"

    # Load graph
    G = load_graph_from_csv(graph_file)
    print("Graph loaded:", G.number_of_nodes(), "nodes")

    # Centralities
    degree, closeness, katz = compute_centralities(G)

    save_centrality(degree, "degree_sorted.csv")
    save_centrality(closeness, "closeness_sorted.csv")
    save_centrality(katz, "katz_sorted.csv")

    print("Centralities computed and saved")

    # Convert to lists
    degree_values = list(degree.values())
    closeness_values = list(closeness.values())
    katz_values = list(katz.values())

    # Load stochastic
    stochastic_file = "simulation_results/raw_results.csv"

    stochastic_values = load_stochastic(
        stochastic_file,
        graph_name,
        G.number_of_nodes()
    )

    print("Stochastic loaded:", len(stochastic_values))

    # Check sizes
    if len(degree_values) != len(stochastic_values):
        print("ERROR: Dimensiuni diferite!")
        print("Centrality:", len(degree_values))
        print("Stochastic:", len(stochastic_values))
        return

    # Correlations
    results = []

    k, _ = kendalltau(degree_values, stochastic_values)
    s, _ = spearmanr(degree_values, stochastic_values)
    print(f"Degree -> Kendall: {k}, Spearman: {s}")
    results.append(["Degree", k, s])

    k, _ = kendalltau(closeness_values, stochastic_values)
    s, _ = spearmanr(closeness_values, stochastic_values)
    print(f"Closeness -> Kendall: {k}, Spearman: {s}")
    results.append(["Closeness", k, s])

    k, _ = kendalltau(katz_values, stochastic_values)
    s, _ = spearmanr(katz_values, stochastic_values)
    print(f"Katz -> Kendall: {k}, Spearman: {s}")
    results.append(["Katz", k, s])

    # Save results
    with open("correlations_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Kendall", "Spearman"])
        for row in results:
            writer.writerow(row)

    print("=== FINAL GATA ===")


if __name__ == "__main__":
    main()