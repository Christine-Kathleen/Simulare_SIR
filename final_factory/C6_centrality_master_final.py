import networkx as nx
# Actualizat pentru a importa din fișierul _final
from .C6_graph_utils_final import load_graph_from_csv, export_sorted_scores

class CentralityMaster:
    """Clasă unificată pentru calculul centralităților matematice."""

    @staticmethod
    def compute_all(graph_path, output_prefix):
        G = load_graph_from_csv(graph_path)
        if G is None or G.number_of_nodes() == 0: 
            return

        print(f"🧐 Analizăm: {graph_path} ({G.number_of_nodes()} noduri)")

        # 1. Degree Centrality
        print("   -> Calculăm Degree...")
        deg = nx.degree_centrality(G)
        export_sorted_scores(deg, f"{output_prefix}_degree_output.csv")

        # 2. Closeness Centrality
        print("   -> Calculăm Closeness...")
        clo = nx.closeness_centrality(G)
        export_sorted_scores(clo, f"{output_prefix}_closeness_output.csv")

        # 3. Katz Centrality (Alpha dinamic pentru stabilitate)
        print("   -> Calculăm Katz...")
        try:
            # Calculăm lambda_max pentru a seta un alpha valid (0.9/lambda_max)
            # Folosim numpy pentru a extrage spectrul adiacenței (necesar pentru Katz)
            adj_matrix = nx.to_numpy_array(G)
            import numpy as np
            l_max = max(abs(np.linalg.eigvals(adj_matrix)))
            
            alpha = 0.9 / l_max if l_max > 0 else 0.1
            katz = nx.katz_centrality(G, alpha=alpha, normalized=True)
            export_sorted_scores(katz, f"{output_prefix}_katz_output.csv")
        except Exception as e:
            print(f"   ⚠️ Eroare Katz la {graph_path}: {e}")

if __name__ == "__main__":
    pass

