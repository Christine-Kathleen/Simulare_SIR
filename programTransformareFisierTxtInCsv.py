import csv


def TransformIntoCsv(file):
    with open(file,"r") as infile,open("CSVfile","w", newline="") as outfile:
        writer=csv.writer(outfile)
        for line in infile:
            clean=line.strip().replace("[","").replace("]","").replace(";","")

            if clean:
                row=clean.split(",")
                writer.writerow(row)

    print("CSV generat cu succes")



if __name__=="__main__":
    TransformIntoCsv("muchii_100_orientat.txt")
