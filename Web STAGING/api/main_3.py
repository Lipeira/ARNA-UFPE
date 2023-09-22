import json
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, session
from functools import wraps
import os
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment
from openpyxl.drawing.image import Image as XlImage
from graphviz import Digraph
import networkx as nx
import io
import pandas as pd
import csv
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__, static_folder='static')

''
# Decorator para verificar a autenticação do usuário
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Lógica de verificação da autenticação do usuário
        if not is_user_authenticated():
            # Redirecionar o usuário para a página de login se não estiver autenticado
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def is_user_authenticated():
    return session.get('authenticated', False)

def authenticate_user():
    session['authenticated'] = True

from flask import session, redirect, url_for

@app.route('/logout')
def logout():
    # Desautentica o usuário
    deauthenticate_user()
    return redirect(url_for('login'))

def deauthenticate_user():
    session.pop('authenticated', None)

def realizar_analise(atividades, Tipo):

    if Tipo == 'PERT':
        def is_edge_in_critical_path(u, v, caminho_critico):
            return (u, v) in zip(caminho_critico[:-1], caminho_critico[1:])
        def is_node_in_critical_path(node, caminho_critico):
            return node in caminho_critico

        # Definir os dados das atividades
        def plot_activities(atividades):
            # Criar o grafo direcionado
            G = nx.DiGraph()

            #Acrescentar T_esperado
            for act in atividades:
                atividades[act]['t_esperado'] = (atividades[act]['t_otimista'] + 4 * atividades[act]['t_provavel'] + atividades[act]['t_pessimista']) / 6

            for act in atividades:
                atividades[act]['dp'] = (atividades[act]['t_pessimista'] - atividades[act]['t_otimista'])/6

            for act in atividades:
                atividades[act]['variance'] = ((atividades[act]['t_pessimista'] - atividades[act]['t_otimista'])/6)**2

            atividades_df = pd.DataFrame.from_dict(atividades, orient='index') 

            # Adicionar os nós e as arestas ao grafo
            for _, info in atividades_df.iterrows():
                duracao = info.get('t_esperado', 0)
                G.add_node(info.name, duracao=duracao)
                precedentes = info.get('precedentes', [])
                for precedente in precedentes:
                    G.add_edge(precedente, info.name, weight=duracao)

            # Adicionar os nós e as arestas ao grafo
            G.add_node("Start", duration=0)  # Nó de início

            for _, info in atividades_df.iterrows():
                duracao = info.get('t_esperado', 0)
                G.add_node(info.name, duracao=duracao)
                precedentes = info.get('precedentes', [])

                # Adicione as arestas entre o nó de início e as atividades iniciais
                if not precedentes:
                    G.add_edge("Start", info.name, weight=duracao)

                for precedente in precedentes:
                    G.add_edge(precedente, info.name, weight=duracao)

            # Encontre as atividades que não são precedentes de outras
            atividades_finais = atividades_df[~atividades_df.index.isin([atividade for info in atividades_df['precedentes'] for atividade in info])]

            # Adicione as arestas das atividades finais ao nó "End"
            for atividade_final in atividades_finais.index:
                G.add_edge(atividade_final, "End", weight=0)

            # Calcular o caminho crítico
            caminho_critico = nx.dag_longest_path(G)

            # Desenhar o grafo usando Graphviz
            dot = Digraph()
            dot.attr(rankdir='LR')  # Configurar o layout da esquerda para a direita
            dot.attr('graph', ranksep='1.5')  # Adicionar o atributo ranksep aqui (aumentar o valor para aumentar o espaço entre os níveis)
            dot.attr('graph', nodesep='0.75')  # Adicionar o atributo nodesep aqui (aumentar o valor para aumentar o espaço entre os nós no mesmo nível)

            # Adicionar os nós e as arestas ao grafo
            for node, data in G.nodes(data=True):
                label = f"{node}"
                dot.node(node, label=label, shape='circle', ports='we')  # Adicionar atributo ports='we' aqui

            edge_number = 1  # Inicializa o contador de numeração das arestas
            edges_df = pd.DataFrame(columns=['Aresta', 'Atividade_Saida', 'Atividade_Chegada'])  # Cria um DataFrame vazio
            for u, v, d in G.edges(data=True):
                edge_label = f'a{edge_number}'  # Nome da aresta
                atividade_saida = u  # Nome do nó de saída da aresta
                atividade_chegada = v  # Nome do nó de chegada da aresta

                edges_df.loc[edge_number - 1] = [edge_label, atividade_saida, atividade_chegada]

                if v == "End" and is_edge_in_critical_path(u, v, caminho_critico):
                    dot.edge(u, v, color='red')
                elif u == caminho_critico[-1] and v == "End":
                    dot.edge(u, v, color='red')
                elif is_edge_in_critical_path(u, v, caminho_critico):
                    dot.edge(u + ":e", v + ":w", color='red')
                else:
                    dot.edge(u + ":e", v + ":w",)

                edge_number += 1

            dot.attr(label='PERT Network', labelloc='bottom')

            # Tempo Total Atividades
            caminho_critico_nos = [node for node in caminho_critico if is_node_in_critical_path(node, caminho_critico)]
            caminho_critico_nos = caminho_critico_nos[1:]

            tempo_total_critico = 0
            for numb in caminho_critico_nos:
                tempo_total_critico += atividades_df.loc [numb, 't_esperado']
            
            caminho_critico_string = " --> ".join(caminho_critico_nos)
            # Salvar a imagem do grafo em formato PNG
            static_dir = os.path.join(os.getcwd(), 'static')
            image_path = os.path.join(static_dir, 'rede_atividades')
            dot.render(image_path, view=False, format='png')
            atividades_df.to_html(static_dir + "/dataframe.html")
            atividades_df.to_excel(static_dir + "/dataframe.xlsx")

            return caminho_critico_string, tempo_total_critico

        caminho_critico, tempo_total = plot_activities(atividades)
        print("Caminho Crítico:", caminho_critico)
        print("Tempo Total Crítico:", tempo_total)
        return caminho_critico, tempo_total
            
    elif Tipo == 'CPM':
            def is_edge_in_critical_path(u, v, caminho_critico):
                return (u, v) in zip(caminho_critico[:-1], caminho_critico[1:])

            def calcular_es_ls(G):
                es = {}
                ls = {}

                # Calcular Early Start (ES)
                for node in nx.topological_sort(G):
                    if not G.in_edges(node):
                        es[node] = 0
                    else:
                        es[node] = max(es[pred] + G.edges[pred, node]['weight'] for pred in G.predecessors(node))

                # Calcular Late Start (LS)
                for node in reversed(list(nx.topological_sort(G))):
                    if not G.out_edges(node):
                        ls[node] = es[node]
                    else:
                        ls[node] = min(ls[succ] - G.edges[node, succ]['weight'] for succ in G.successors(node))

                return es, ls

            # Definir os dados das atividades

            def plot_activities(atividades):
                # Criar o grafo direcionado
                G = nx.DiGraph()
                

                # Adicionar os nós e as arestas ao grafo
                for atividade, info in atividades.items():
                    duracao = info.get('duracao', 0)
                    G.add_node(atividade, duracao=duracao)
                    precedentes = info.get('precedentes', [])
                    for precedente in precedentes:
                        G.add_edge(precedente, atividade, weight=duracao)



                # Calcular o caminho crítico
                caminho_critico = nx.dag_longest_path(G)

                # Calcular Early Start (ES) e Late Start (LS)
                es, ls = calcular_es_ls(G)

                # Calcular Early Finish (ES) e Late Finish (LS)
                ef = {node: es[node] + G.nodes[node]['duracao'] for node in G.nodes()}
                lf = {node: ls[node] + G.nodes[node]['duracao'] for node in G.nodes()}

                # Calcular a folga (float) de cada atividade
                folga = {node: ls[node] - es[node] for node in G.nodes()}


                # Calcular a posição dos nós usando o layout planar com scale=2.0
                pos = nx.planar_layout(G, scale=7.0)

                # Desenhar o grafo usando Graphviz
                dot = Digraph()
                dot.attr(rankdir='LR')  # Configurar o layout da esquerda para a direita
                dot.attr('graph', ranksep='1.5')  # Adicionar o atributo ranksep aqui (aumentar o valor para aumentar o espaço entre os níveis)
                dot.attr('graph', nodesep='0.75')  # Adicionar o atributo nodesep aqui (aumentar o valor para aumentar o espaço entre os nós no mesmo nível)

                # Adicionar os nós e as arestas ao grafo
                for node, data in G.nodes(data=True):
                    label = f"{node} | Peso: {data['duracao']} | ES: {es[node]} / EF: {ef[node]} | LS: {ls[node]} / LF: {lf[node]} | Folga: {folga[node]} "
                    dot.node(node, label=label, shape='record', ports='we')  # Adicionar atributo ports='we' aqui

                initial_nodes = [node for node in G.nodes() if not G.in_edges(node)]
                final_nodes = [node for node in G.nodes() if not G.out_edges(node)]

                dot.node('start', label='start', shape='rectangle')  # Adicionar o nó "start"
                dot.node('end', label='end', shape='rectangle')  # Adicionar o nó "end"

                for initial_node in initial_nodes:
                    dot.edge('start', initial_node + ':w')  # Conectar "start" aos nós iniciais

                for final_node in final_nodes:
                    dot.edge(final_node + ':e', 'end')  # Conectar os nós finais ao nó "end"

                label = "Tarefa|Peso|ES e EF|LS e LF|Folga"

                # Criar um subgrafo com o mesmo rank
                with dot.subgraph() as same_rank:
                    same_rank.attr(rank='same')
                    same_rank.node('start')  # Mover o nó "start" para este subgrafo
                    # Adicionar o nó de legenda com atributo "pos" para posicioná-lo abaixo do nó "start"
                    same_rank.node('legend', label=label, shape='record', pos='0,1!', pin='true') 


                for u, v, d in G.edges(data=True):
                    if is_edge_in_critical_path(u, v, caminho_critico):
                        dot.edge(u + ":e", v + ":w", color='red')  # Modificar u para u + ":e" e v para v + ":w" aqui
                    else:
                        dot.edge(u + ":e", v + ":w")  # Modificar u para u + ":e" e v para v + ":w" aqui


                dot.attr(label='CPM Network', labelloc='bottom')

                # Salvar a imagem do grafo em formato PNG
                static_dir = os.path.join(os.getcwd(), 'static')
                image_path = os.path.join(static_dir, 'rede_atividades')
                dot.render(image_path, view=False, format='png')

            if Tipo == 'CPM':
                plot_activities(atividades)
            elif Tipo == 'PERT':
                caminho_critico, tempo_total, edges_df = plot_activities(atividades)
                return caminho_critico, tempo_total, edges_df

app.secret_key = 'ALDmofmognmg15448d8'
# ------------ ROTAS -----------------

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():

    if is_user_authenticated():
        return redirect(url_for('home'))

    # Lógica de autenticação do usuário
    if request.method == 'POST':
        email = request.form['email']
        senha = request.form['senha']
        
        if email == 'admin' and senha == '123456':
            # Autenticar o usuário
            authenticate_user() 
            return redirect(url_for('home'))
        else:
            mensagem_erro = 'Usuário ou senha incorretos'
            return render_template('login.html', erro=mensagem_erro)

    return render_template('login.html')


@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/PERT')
@login_required
def PERT():
    return render_template('homePERT.html')


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if request.method == 'POST':
        csv_file = request.files['csv_file']  # Obter o arquivo CSV do formulário
        if csv_file:  # Se um arquivo CSV foi enviado
            csv_content = csv_file.read().decode('utf-8')
            try:
                atividades_from_csv = json.loads(csv_content)
            except json.JSONDecodeError:
                erro = "O arquivo CSV deve conter um JSON válido."
                return render_template("home.html", erro=erro)
            atividades_json = atividades_from_csv
            
        else:
            atividades = request.form.get("atividades")
            # Verificar se o campo de atividades está vazio
            if not atividades:
                erro = "O campo de atividades não pode estar vazio."
                return render_template("home.html", erro=erro)
            
            try:
                atividades_json = json.loads(atividades)
            except json.JSONDecodeError:
                erro = "O campo de atividades deve ser um JSON válido."
                return render_template("home.html", erro=erro)

        results = realizar_analise(atividades_json, 'CPM')
        session['analysis_results'] = results
        return redirect(url_for('result'))
    

@app.route('/analyzePERT', methods=['POST'])
@login_required
def analyzePERT():
    if request.method == 'POST':
        csv_file = request.files['csv_file']  # Obter o arquivo CSV do formulário
        json_file = request.files['json_file']


        if csv_file:  # Se um arquivo CSV foi enviado
            csv_content = csv_file.read().decode('utf-8')

            # Crie uma lista para armazenar os dados do CSV
            atividades_from_csv = {}

            # Use um leitor CSV para ler as linhas do arquivo CSV
            csv_reader = csv.reader(csv_content.splitlines())
            
            # Pule a primeira linha (cabeçalho)
            next(csv_reader)
            
            for row in csv_reader:
                # Extraia os dados de cada linha
                atividade, precedentes, t_otimista, t_pessimista, t_provavel = row
                
                # Converta os valores numéricos para inteiros ou floats, conforme necessário
                t_otimista = int(t_otimista)
                t_pessimista = int(t_pessimista)
                t_provavel = int(t_provavel)
                
                # Crie um dicionário para representar cada atividade e seus dados
                atividade_data = {
                    "precedentes": precedentes.split(', ') if precedentes else [],
                    "t_otimista": t_otimista,
                    "t_pessimista": t_pessimista,
                    "t_provavel": t_provavel
                }
                
                # Adicione o dicionário à lista de atividades
                atividades_from_csv[atividade] = (atividade_data)
            
            atividades_json = atividades_from_csv

        elif json_file:  # Se um arquivo JSON foi enviado
            input(json_file)
            json_content = json_file.read().decode('utf-8')
            input(json_content)
            try:
                data_from_json = json.loads(json_content)
                input(data_from_json)
                atividades_json = data_from_json
            except json.JSONDecodeError:
                erro = "O arquivo JSON não é válido."
                return render_template("home.html", erro=erro)
            
        else:
            atividades = request.form.get("atividades")
            # Verificar se o campo de atividades está vazio
            if not atividades:
                erro = "O campo de atividades não pode estar vazio."
                return render_template("home.html", erro=erro)
            
            try:
                atividades_json = json.loads(atividades)

            except json.JSONDecodeError:
                erro = "O campo de atividades deve ser um JSON válido."
                return render_template("homePERT.html", erro=erro)

        caminho_critico, tempo_total= realizar_analise(atividades_json, 'PERT')
        session['caminho_critico'] = caminho_critico
        session['tempo_total'] = tempo_total
        session['atividades_json'] = atividades_json

        return render_template('resultPERT.html', caminho_critico=caminho_critico, tempo_total=tempo_total)

@app.route('/result')
@login_required
def result():
    return render_template('result.html')


@app.route('/resultPERT')
@login_required
def resultPERT():
    caminho_critico = session.get('caminho_critico')
    tempo_total = session.get('tempo_total')
    edges_df = session.get('edges_df')

    return render_template('resultPERT.html')

@app.route('/download_png')
@login_required
def download_png():
    # Baixar o arquivo PNG do Amazon S3
    """ s3.download_file(BUCKET_NAME, 'rede_atividades.png', 'static/rede_atividades.png') """
    return send_from_directory('static', 'rede_atividades.png', as_attachment=True)

@app.route("/download_pdf_PERT")
@login_required
def download_pdf():
    session.get()
    return send_from_directory('static', 'report_PERT.pdf', as_attachment=True)

@app.route("/download_xls")
@login_required
def download_xls():
    # Baixar o arquivo XLSX do Amazon S3
    """ s3.download_file(BUCKET_NAME, 'report.xlsx', 'static/report.xlsx') """
    return send_from_directory('static', 'dataframe.xlsx', as_attachment=True)


@app.route('/help')
@login_required
def help():
    return render_template('help.html')

@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html')

@app.route('/calculo_tempo', methods=['POST'])
@login_required
def calculo_tempo():
    if request.method == 'POST':
        entrada = request.form.get("entrada")
        saida = request.form.get("saida")
        atividades_json = session.get('atividades_json')
        # Função para encontrar um caminho entre dois nós (entrada e saída)
        def encontrar_caminho(no_atual, caminho):
            caminho.append(no_atual)
            if no_atual == saida:
                caminhos.append(list(caminho))
            else:
                for no in atividades_json[no_atual]['precedentes']:
                    encontrar_caminho(no, caminho)
            caminho.pop()

        # Verifique se os nós de entrada e saída são válidos
        if entrada not in atividades_json or saida not in atividades_json:
            return "Nós de entrada ou saída inválidos."
        
        return tempo_total
    

@app.route('/gauss', methods=['POST'])
@login_required
def gauss():
    t_programado = float(request.form.get("t_programado"))
    tempo_total = float(session.get('tempo_total'))
    atividades_json = session.get('atividades_json')
    df = pd.DataFrame.from_dict(atividades_json, orient='index') 

    """ AAAAAAAAAAAAA """

    # encontrar a tabela Z

    med_z = 0  #média da tabela Z
    sd_z = 1  #desvio padrão da tabela Z
    med_pj = tempo_total
    sd_pj = mx_10['prj_sd'].item() #desvio padrão do projeto
    tm_x = t_programado

    if tm_x > med_pj + 4 * sd_pj:
        tm_x = med_pj + 4 * sd_pj
    elif tm_x < med_pj - 4 * sd_pj:
        tm_x = med_pj - 4 * sd_pj
    else:
        tm_x

    sd_x = (tm_x - med_pj)/sd_pj #encontrar o DP pretendido

    #gráfico da tabela Z
    ax_x_z = np.linspace(med_z - 4 * sd_z, med_z + 4 * sd_z, 5000)
    polinomio = 30  # escolher o polinômio da função
    f_z = np.poly1d(np.polyfit(ax_x_z, stats.norm.pdf(ax_x_z, med_z, sd_z), polinomio))  # função polinomial da curva normal
    int_z = intg.quad(f_z, 0,sd_x)  # integral da curva normal, que é a probabilidade da tabela Z, a partir do DP pretendido a na equação de Z
    plt.figure(figsize=(10, 5))
    plt.plot(ax_x_z, stats.norm.pdf(ax_x_z, med_z, sd_z))
    #pintar a área abaixo da curva
    plt.fill_between(ax_x_z,stats.norm.pdf(ax_x_z, med_z, sd_z),where=[(ax_x_z > -4) and (ax_x_z < med_z+sd_x) for ax_x_z in ax_x_z], alpha = 0.4)
    plt.xlabel('Curva normal da Tabela Z', fontsize='12') #etiqueta do gráfico
    plt.gca().yaxis.set_major_locator(NullLocator())  #retira marcação do eixo y
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))  #inclui marcação menor do eixo x
    plt.show()
    print('Tabela Z: DvP '+str(round(sd_x,3))+' / P '+str(round(int_z[0], 4)))

    # gráfico do tempo real
    plt.figure(figsize=(10, 5))
    ax_x = np.linspace(med_pj - 4 * sd_pj, med_pj + 4 * sd_pj, 5000)
    plt.plot(ax_x, stats.norm.pdf(ax_x, med_pj, sd_pj))
    plt.axvline(x=med_pj, ymin=0.05, ymax=0.95, color='red', lw=2, ls='solid' ) #linha da média
    #plt.axvline(x=tm_x, ymin=0.05, ymax=0.95, color='red', lw=2.5, ls='solid') #linha 1sd
    plt.text(med_pj + (med_pj * 0.01),0.02,str(round(med_pj,2)) + ' med',rotation=90,color='red') #texto da média
    plt.text(tm_x - (tm_x * 0.01),-0.001,str(tm_x),color='red') #texto do tempo pretendido
    #pintar a área abaixo da cura
    plt.fill_between(ax_x,stats.norm.pdf(ax_x, med_pj, sd_pj),where=[(ax_x > med_pj - 4*sd_pj) and (ax_x < tm_x) for ax_x in ax_x], alpha = 0.4)
    plt.xlabel('Tempo para conclusão do projeto',fontsize = '12')
    plt.gca().yaxis.set_major_locator(NullLocator()) #retira marcação do eixo y
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1)) #inclui marcação menor do eixo x
    plt.show()

    print('Tempo médio do projeto de '+str(round(med_pj,4))+' UT, com DvP '+str(round(sd_pj,4))+'.')
    print('A probabilidade do projeto encerrar até '+str(round(tm_x,2))+' UT é '+str(round(0.5+int_z[0], 4))+'.')

    # calcular a integral inversa na mão, usando os dados da tabela Z

    ZZ = {'sdz':(3.9,1.6450,1.2815,1.0365,0.8415,0.6745,0.5245,0.3853,0.2533,0.1256,0.0000,
                -0.1256,-0.2533,-0.3853,-0.5245,-0.6745,-0.8415,-1.0365,-1.2815,-1.6450,-3.9),
        'prob':(0.9999,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,
                0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05,0.0001)}

    tb_z = pd.DataFrame(ZZ)
    tb_z['p_time'] = tb_z.apply(lambda x: x['sdz']*sd_pj+med_pj,axis=1)

    # calcular a normal inversa com scipy

    norm.pdf(0.0001, loc=med_pj, scale=sd_pj)

    tb_z['p_time2'] = tb_z.apply(lambda x: norm.ppf(x['prob'], loc=med_pj, scale=sd_pj),axis=1)
    tb_z['p_time2']

    """ AAAAAAAAAAAAAA """










    soma_var = 0
    #df.iterrows() itera sobre cada linha ou index do dataframe
    for index, row in df.iterrows():
        soma_var += row['variance']

    static_dir = os.path.join(os.getcwd(), 'static')
    plt.savefig(static_dir + '/grafico_gaussiano.png')
    return send_from_directory('static', 'grafico_gaussiano.png', as_attachment=True)

if __name__ == '__main__':
    app.run()