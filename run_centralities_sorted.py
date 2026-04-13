import csv
import networkx as nx

from C6_degree_centrality_module import degree_centrality
from C6_closeness_centrality_module import closeness_centrality
from C6_katz_centrality_module import katz_centrality


def load_graph_from_csv(file_path):
    G = nx.Graph()
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            G.add_edge(u, v)
    return G


def save_sorted_results(results, filename):
    sorted_results = dict(sorted(results.items()))

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Node", "Value"])
        for node, value in sorted_results.items():
            writer.writerow([node, value])


def main():
    print("START")

    graph_file = "graph5.csv"

    G = load_graph_from_csv(graph_file)
    print("Graph loaded:", G.number_of_nodes(), "nodes")

    deg = degree_centrality(G)
    print("Degree done")

    clo = closeness_centrality(G)
    print("Closeness done")

    katz = katz_centrality(G)[0]
    print("Katz done")

    save_sorted_results(deg, "degree_sorted.csv")
    save_sorted_results(clo, "closeness_sorted.csv")
    save_sorted_results(katz, "katz_sorted.csv")

    print("FINAL GATA")


if __name__ == "__main__":
    main()