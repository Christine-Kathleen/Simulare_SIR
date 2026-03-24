from C6_graph_utils import read_graph_dict, export_to_csv

def degree_centrality(graf):
    centrality = {}
    n = len(graf)

    for nod in graf:
        if n > 1:
            centrality[nod] = len(graf[nod]) / (n - 1)
        else:
            centrality[nod] = 0.0

    return centrality

def compute_degree_from_csv(path, output_file=None):
    graf = read_graph_dict(path)
    result = degree_centrality(graf)
    max_node = max(result, key=result.get)

    if output_file:
        export_to_csv(result, output_file)

    return {
        "result": result,
        "max_node": max_node
    }
