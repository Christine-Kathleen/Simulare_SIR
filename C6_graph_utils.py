import csv
import networkx as nx

def read_graph_nx(path):
    G = nx.Graph()

    with open(path, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            a = int(row[0])
            b = int(row[1])
            
            G.add_edge(a, b)

    return G

def read_graph_dict(path):
    graf = {}

    with open(path, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            a = int(row[0])
            b = int(row[1])

            if a not in graf:
                graf[a] = set()

            if b not in graf:
                graf[b] = set()

            graf[a].add(b)
            graf[b].add(a)

    return graf

# def export_to_csv(result, filename):
#     with open(filename, "w", newline="") as f:
#         writer = csv.writer(f)

#         for node, score in result.items():
#             writer.writerow([node, round(score, 6)])

def export_to_csv(result, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        for node in sorted(result.keys()): #sortare dupa cheie (adica dupa nod, si nu dupa scorul nodului)
            writer.writerow([round(result[node], 6)])
