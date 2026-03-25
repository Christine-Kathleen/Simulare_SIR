from collections import deque
from C6_graph_utils import read_graph_dict, export_to_csv

#    Echipa de centralitati (degree si closeness): Butyka Ana, Ous Andreea, Saracsan Sorana, Stefan Natalia, Zanfir Denis 
#    Echipa de centralitati (Katz) : Plugar Mihai si 

def closeness_centrality(graf):
    centrality = {}
    n = len(graf)

    for start in graf:
        distante = {start: 0}
        coada = deque([start])

        while coada:
            nod = coada.popleft()
            for vecin in graf[nod]:
                if vecin not in distante:
                    distante[vecin] = distante[nod] + 1
                    coada.append(vecin)

        suma_dist = sum(distante.values())

        if suma_dist > 0 and n > 1:
            centrality[start] = (n - 1) / suma_dist
        else:
            centrality[start] = 0.0

    return centrality

def compute_closeness_from_csv(path, output_file=None):
    graf = read_graph_dict(path)
    result = closeness_centrality(graf)

    if output_file:
        export_to_csv(result, output_file)

    return {
        "result": result
    }
