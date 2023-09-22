import csv

data = {
    "A": {"precedentes": [], "t_otimista": 2, "t_pessimista": 8, "t_provavel": 5},
    "B": {"precedentes": ["A"], "t_otimista": 3, "t_pessimista": 10, "t_provavel": 6},
    "C": {"precedentes": ["A"], "t_otimista": 1, "t_pessimista": 4, "t_provavel": 5},
    "D": {"precedentes": ["B"], "t_otimista": 4, "t_pessimista": 6, "t_provavel": 8},
    "E": {"precedentes": ["B"], "t_otimista": 8, "t_pessimista": 12, "t_provavel": 10},
    "F": {"precedentes": ["C"], "t_otimista": 3, "t_pessimista": 6, "t_provavel": 5},
    "G": {"precedentes": ["D", "E"], "t_otimista": 7, "t_pessimista": 11, "t_provavel": 8},
    "H": {"precedentes": ["F"], "t_otimista": 3, "t_pessimista": 6, "t_provavel": 5}
}

# Especifica o nome do arquivo CSV de saída
output_file = "dados.csv"

# Cria o arquivo CSV e escreve os dados nele
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Escreve o cabeçalho do CSV
    writer.writerow(["atividade", "precedentes", "t_otimista", "t_pessimista", "t_provavel"])
    
    # Escreve os dados das atividades
    for atividade, info in data.items():
        precedentes = ', '.join(info["precedentes"])
        t_otimista = info["t_otimista"]
        t_pessimista = info["t_pessimista"]
        t_provavel = info["t_provavel"]
        writer.writerow([atividade, precedentes, t_otimista, t_pessimista, t_provavel])

print(f'Os dados foram escritos no arquivo CSV: {output_file}')
