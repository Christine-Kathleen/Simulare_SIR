from math import sqrt
import networkx as nx
from C6_graph_utils import read_graph_nx, export_to_csv

def katz_centrality(G: nx.Graph, alpha=0.1, beta=1.0, max_iter=1000, tol=1.0e-6, 
                        nstart=None, normalized=True, weight = 'weight'):
    if len(G) == 0:
        return {}

    nnodes = G.number_of_nodes()

    if nstart is None:
        x = dict([(n, 0) for n in G]) #"x(0)" => se pleaca de la un vect de 0
    else:
        x = nstart

    try:
        b = dict.fromkeys(G, float(beta))
    except (TypeError, ValueError, AttributeError):
        b = beta

        if set(beta) != set(G):
            raise nx.NetworkXError('beta dictionary''must have a value for every node')

    for i in range(max_iter): #implementare metoda iterativa "x(k + 1) = alpha * A * x(k) + beta"
        xlast = x #salvarea val anterioare => "x(k)"
        x = dict.fromkeys(xlast, 0) #resetare vect (pt pregatirea "x(k + 1)")

        #implementare "A * x" => se parcurge fiecare nod "n", apoi fiecare vecin "nbr" si se adauga contributia lui "n"
        #catre vecin (inmultirea matrice - vector)
        for n in x: 
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
        
        for n in x: #implementare formula completa pt "x = alpha * A * x + beta"
            x[n] = alpha * x[n] + b[n]

        err = sum([abs(x[n] - xlast[n]) for n in x]) #cond de oprire (de convergenta) => "|x(k + 1) - x(k)|"

        #se ver daca "x(k) aproximativ egal cu x(k - 1)", adica daca sol s-a stabilizat; "err < nnodes * tol" se continua 
        #pana cand diferentele devin foarte mici 
        if err < nnodes * tol: 
            if normalized:
                try:
                    s = 1.0 / sqrt(sum(v ** 2 for v in x.values()))
                except ZeroDivisionError:
                    s = 1.0
            else:
                s = 1
            
            for n in x:
                x[n] *= s

            return x, i

    #daca nu converge (posibil ca "alpha" sa fie prea mare)
    raise nx.NetworkXError('Power iteration failed to converge in ''%d iterations.' % max_iter)

def compute_alpha(G, factor=0.9):
    valoare_proprie_maxima = max(nx.adjacency_spectrum(G)).real
    alpha = factor / valoare_proprie_maxima
    
    return alpha, valoare_proprie_maxima

def compute_katz_from_csv(path, alpha_factor=0.9, output_file=None):
    G = read_graph_nx(path)
    alpha, _ = compute_alpha(G, alpha_factor)
    result, _ = katz_centrality(G, alpha=alpha)

    if output_file:
        export_to_csv(result, output_file)

    return {
        "result": result
    }
