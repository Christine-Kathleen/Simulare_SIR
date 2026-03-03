import networkx as nx
import matplotlib.pyplot as plt
import csv
from collections import defaultdict, deque

def cheie_sortare(x):
    # incearca sortare numerica daca nodurile sunt numere
    try:
        return int(x)
    except ValueError:
        return x

def citire_graf(nume_fisier):
    graf = defaultdict(set)
    
    with open(nume_fisier, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            u = row[0].strip()
            v = row[1].strip()
            
            # Eliminare loop
            if u == v:
                continue
            
            # Transformare in neorientat
            graf[u].add(v)
            graf[v].add(u)
    
    return graf

def componente_conexe(graf):
    vizitat = set()
    componente = []

    for nod in graf:
        if nod not in vizitat:
            comp = []
            coada = deque([nod])
            vizitat.add(nod)

            while coada:
                x = coada.popleft()
                comp.append(x)

                for vecin in graf[x]:
                    if vecin not in vizitat:
                        vizitat.add(vecin)
                        coada.append(vecin)

            componente.append(comp)

    return componente

def pastreaza_componenta_maxima(graf):
    comps = componente_conexe(graf)
    comp_max = max(comps, key=len)

    graf_nou = defaultdict(set)
    for nod in comp_max:
        for vecin in graf[nod]:
            if vecin in comp_max:
                graf_nou[nod].add(vecin)

    return graf_nou

def numara(graf):
    noduri = len(graf)
    muchii = sum(len(graf[nod]) for nod in graf) // 2
    return noduri, muchii

def salveaza_graf(graf, nume_fisier):
    with open(nume_fisier, 'w', newline='') as f:
        writer = csv.writer(f)
        scrise = set()

        # sortam nodurile crescator
        for u in sorted(graf.keys(), key=cheie_sortare):
            for v in sorted(graf[u], key=cheie_sortare):
                if (v, u) not in scrise:
                    writer.writerow([u, v])
                    scrise.add((u, v))

def afiseaza_graf(graf):
    G = nx.Graph()

    # adaugare muchii
    for u in graf:
        for v in graf[u]:
            if u <= v:   # evitam dublurile
                G.add_edge(u, v)

    plt.figure()
    pos = nx.spring_layout(G)  # layout automat (forta elastica)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        font_size=10
    )

    plt.title("Graful curatat (componenta maxima)")
    plt.show()

input_file = "arbore_100noduri_natural.csv"

graf = citire_graf(input_file)

graf_curatat = pastreaza_componenta_maxima(graf)

noduri, muchii = numara(graf_curatat)
output_file = f"{muchii}muchii_{noduri}noduri_natural_output.csv"
print("Numar noduri:", noduri)
print("Numar muchii:", muchii)
try:
    salveaza_graf(graf_curatat, output_file)
    print("Graful a fost salvat cu succes in", output_file)
    afiseaza_graf(graf_curatat)
except Exception as e:
    print("Eroare la salvarea grafului curatat.")
    print("Detalii eroare:", e)
