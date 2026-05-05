import networkx as nx
from collections import deque, defaultdict

class GraphPreprocessor:
  

    @staticmethod
    def get_max_component(G):
        """
        Extrage componenta conexa maxima (programCuratare.py).
        Garanteaza ca simularea SIR nu se blocheaza in 'insule' izolate.
        """
        if len(G) == 0:
            return G
            
        # Gasim componentele conexe folosind NetworkX (bazat pe BFS/DFS)
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        max_comp_nodes = components[0]
        
        return G.subgraph(max_comp_nodes).copy()

    @staticmethod
    def remove_cycles_dsu(G):
        """
        Elimina ciclurile folosind Union-Find (DSU).
        Transforma graful intr-un arbore (Spanning Tree).
        """
        parent = {node: node for node in G.nodes()}

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i
                return True
            return False

        T = nx.Graph()
        T.add_nodes_from(G.nodes())
        
        # Parcurgem muchiile si le adaugam doar daca nu inchid un ciclu
        for u, v in G.edges():
            if union(u, v):
                T.add_edge(u, v)
        
        return T

    @staticmethod
    def cleanup_pipeline(G, make_tree=False):
        """Pipeline complet de curatare."""
        if G is None: return None
        
        # 1. Pastram doar componenta maxima
        G_clean = GraphPreprocessor.get_max_component(G)
        
        # 2. Daca se cere specific un arbore (pentru grafuri sintetice tip tree)
        if make_tree:
            G_clean = GraphPreprocessor.remove_cycles_dsu(G_clean)
            
        return G_clean
