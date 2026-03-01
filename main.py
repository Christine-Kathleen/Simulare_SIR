import networkx as nx
import matplotlib.pyplot as plt
import csv
import random

#Echipa care a lucrat la codul de generare de graph-uri: Deva Dan, Hirceaga Andreea, Pascu Lorena, Racz Christine

# Verificari
def check(v, m):
    muchii_max = v * (v - 1) // 2

    if v > 100:
        print("Numarul maxim de varfuri este 100.")
        return False

    if m > 5000:
        print("Numarul maxim de muchii este 5000.")
        return False

    if m > muchii_max:
        print("Prea multe muchii pentru un graf neorientat.")
        return False

    # if m < v - 1:
    #     print("Graful nu poate fi conex (m < v - 1).")
    #     return False

    return True

def gnm_conex_neorientat(n, m, seed=None):
    if m < n - 1:
        raise ValueError("Pentru graf conex trebuie m >= n-1")

    if seed is not None:
        random.seed(seed)
    # creare graf gol, cu noduri de la 0 la n
    G = nx.Graph()
    G.add_nodes_from(range(n))


    nodes = list(G.nodes())
    random.shuffle(nodes)
    # fiecare nod se leaga de unul deja conectat
    for i in range(1, n):
        u = nodes[i]
        v = random.choice(nodes[:i])
        G.add_edge(u, v)

    # adaugare muchii suplimentare pana la m
    while G.number_of_edges() < m:
        u = random.randrange(n)
        v = random.randrange(n)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G

def main():
    # introducere numar varfuri si muchii
    v = int(input("Varfuri: "))
    m = int(input("Muchii: "))

    if check(v, m):
        # generare graf
        # G = nx.gnm_random_graph(v, m)
        G = gnm_conex_neorientat(v, m)

        # salvare muchii in CSV
        with open("graph5.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for u, w in G.edges():
                writer.writerow([u, w])

        print("Graful a fost salvat in graph.csv")

        if nx.is_tree(G):
            print("Graful NU are cicluri")
        else:
            print("Graful ARE cicluri")

        # desen graf
        nx.draw(G, with_labels=True)
        plt.show()


if __name__ == "__main__":
    main()