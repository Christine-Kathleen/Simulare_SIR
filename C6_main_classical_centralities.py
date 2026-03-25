from C6_katz_centrality_module import compute_katz_from_csv
from C6_degree_centrality_module import compute_degree_from_csv
from C6_closeness_centrality_module import compute_closeness_from_csv

#    Echipa de centralitati: Butyka Ana, Ous Andreea, Plugar Mihai, Saracsan Sorana, Stefan Natalia, Zanfir Denis 


def make_output_name(prefix, input_path):
    filename = input_path.split("/")[-1]
    
    return f"{prefix}_{filename}"

paths = [
    "Simulare_SIR/grafuri_naturale_si_generate_curatate/99muchii_100noduri_generat_output.csv",
    "Simulare_SIR/grafuri_naturale_si_generate_curatate/99muchii_100noduri_natural_output.csv",
    "Simulare_SIR/grafuri_naturale_si_generate_curatate/275muchii_100noduri_generat_output.csv",
    "Simulare_SIR/grafuri_naturale_si_generate_curatate/275muchii_100noduri_natural_output.csv",
    "Simulare_SIR/grafuri_naturale_si_generate_curatate/1000muchii_100node_natural_output.csv",
    "Simulare_SIR/grafuri_naturale_si_generate_curatate/1000muchii_100noduri_generat_output.csv"
]

for path in paths:
    compute_degree_from_csv(path=path, output_file=make_output_name("degree", path))
    compute_katz_from_csv(path=path, output_file=make_output_name("katz", path))
    compute_closeness_from_csv(path=path, output_file=make_output_name("closeness", path))
