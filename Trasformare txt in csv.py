import csv

def curata_si_converteste(input_txt, output_csv):
    with open(input_txt, 'r') as fin, open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)

        for linie in fin:
            linie = linie.strip()

            # sarim peste linii goale
            if not linie:
                continue

            # eliminam caracterele nedorite
            linie = linie.replace('[', '').replace(']', '').replace(';', '')

            # daca are virgula -> format [1,2]
            if ',' in linie:
                parti = linie.split(',')

            # daca are spatii -> format 64 67 2256
            else:
                parti = linie.split()

            # pastram doar primele doua valori (fara ponderi)
            if len(parti) >= 2:
                u = parti[0].strip()
                v = parti[1].strip()

                writer.writerow([u, v])


# utilizare
input_txt = "raport_graf_100.txt"
output_csv = "raport_graf_100.csv"

curata_si_converteste(input_txt, output_csv)

print("Conversie + curatare realizata cu succes!")