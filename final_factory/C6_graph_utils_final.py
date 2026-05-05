import csv
import networkx as nx
import os

def load_graph_from_csv(path):
    """Încarcă un graf din CSV, ignorând headerele și gestionând erorile de format."""
    G = nx.Graph()
    if not os.path.exists(path):
        print(f"⚠️ Fișier negăsit: {path}")
        return None
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                # Corecție: accesăm elementele rândului prin index [0] și [1]
                u, v = int(row[0]), int(row[1])
                G.add_edge(u, v)
            except (ValueError, IndexError):
                continue
    return G

def export_sorted_scores(data, filename):
    """Salvează doar scorurile, sortate strict după ID-ul nodului (1, 2, 10...)."""
    # Ne asigurăm că folderul destinație există înainte de salvare
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Sortare numerică: Nodul 2 vine înaintea lui 10
        for node in sorted(data.keys(), key=lambda x: int(x)):
            writer.writerow([round(data[node], 6)])
    print(f"✅ Rezultat salvat: {filename}")