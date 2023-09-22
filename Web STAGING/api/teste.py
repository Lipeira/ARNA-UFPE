#pandas 

import pandas as pd

atividades_json = {
   "A":{
      "precedentes":[
         
      ],
      "t_otimista":2,
      "t_pessimista":8,
      "t_provavel":5,
      "t_esperado":5.0,
      "dp":1.0,
      "variance":1.0
   },
   "B":{
      "precedentes":[
         "A"
      ],
      "t_otimista":3,
      "t_pessimista":10,
      "t_provavel":6,
      "t_esperado":6.166666666666667,
      "dp":1.1666666666666667,
      "variance":1.3611111111111114
   },
   "C":{
      "precedentes":[
         "A"
      ],
      "t_otimista":1,
      "t_pessimista":4,
      "t_provavel":5,
      "t_esperado":4.166666666666667,
      "dp":0.5,
      "variance":0.25
   },
   "D":{
      "precedentes":[
         "B"
      ],
      "t_otimista":4,
      "t_pessimista":6,
      "t_provavel":8,
      "t_esperado":7.0,
      "dp":0.3333333333333333,
      "variance":0.1111111111111111
   },
   "E":{
      "precedentes":[
         "B"
      ],
      "t_otimista":8,
      "t_pessimista":12,
      "t_provavel":10,
      "t_esperado":10.0,
      "dp":0.6666666666666666,
      "variance":0.4444444444444444
   },
   "F":{
      "precedentes":[
         "C"
      ],
      "t_otimista":3,
      "t_pessimista":6,
      "t_provavel":5,
      "t_esperado":4.833333333333333,
      "dp":0.5,
      "variance":0.25
   },
   "G":{
      "precedentes":[
         "D",
         "E"
      ],
      "t_otimista":7,
      "t_pessimista":11,
      "t_provavel":8,
      "t_esperado":8.333333333333334,
      "dp":0.6666666666666666,
      "variance":0.4444444444444444
   },
   "H":{
      "precedentes":[
         "F"
      ],
      "t_otimista":3,
      "t_pessimista":6,
      "t_provavel":5,
      "t_esperado":4.833333333333333,
      "dp":0.5,
      "variance":0.25
   }
}

# Create a list of dictionaries where each dictionary has the activity name as a key and its data as values
data_list = [{"Atividade": activity, **values} for activity, values in atividades_json.items()]

# Create a DataFrame from the list of dictionaries, and set the index to be the default numeric index
df = pd.DataFrame(data_list).reset_index(drop=True)

#def gauss(): 

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


t_programado = 41
tempo_total = 40

soma_var = 0

#df.iterrows() itera sobre cada linha ou index do dataframe
for index, row in df.iterrows():
    soma_var += row['variance']

z = (t_programado - tempo_total) / (soma_var**(1/2))
z = f"{z:.1f}"
z = float(z)

mi = tempo_total
pct_projeto_programado = 50/100

area = norm.cdf(z)
print(f"Área: {area:.3f}")

# Cria um conjunto de valores x no intervalo [tempo_total - 4, tempo_total + 4] para o tempo do projeto
x = np.linspace(tempo_total - 5, tempo_total + 5, 1000)

# Calcula a probabilidade acumulada até o valor Z
area = norm.cdf(z)

# Cria uma figura e um eixo
fig, ax = plt.subplots()

# Plota a curva gaussiana
ax.plot(x, norm.pdf(x, tempo_total), label='Distribuição Normal')

# Preenche a área sob a curva até o valor de Z
ax.fill_between(x, 0, norm.pdf(x, tempo_total), where=(x <= z + tempo_total), alpha=0.3, label=f'Probabilidade = {area:.5f}')

# Adiciona uma linha vermelha no valor médio (símbolo μ)
ax.axvline(x=tempo_total, color='red', linestyle='dashed', label=f'Valor Médio (μ = {tempo_total:.2f})')

# Configurações do gráfico
ax.set_title('Distribuição Gaussiana com Valor Médio')
ax.set_xlabel('Tempo do Projeto')
ax.set_ylabel('Densidade de Probabilidade')
ax.legend()
plt.savefig('grafico_gaussiano.png')