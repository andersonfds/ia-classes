import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

# carregando dataset
df = pd.read_csv("ds/playtennis.csv")

# convertendo os dados nao-numericos em numericos (exigencia da DT)
# cria um mapeamento para cada dado nas colunas e sobreescreve os valores
d = {'ensolarado': 2, 'nublado': 0, 'chuvoso': 1}
df['aparencia'] = df['aparencia'].map(d)

d = {'quente': 1, 'moderada': 2, 'fria': 0}
df['temperatura'] = df['temperatura'].map(d)

d = {'normal': 1, 'alta': 0}
df['umidade'] = df['umidade'].map(d)

d = {'fraco': 0, 'forte': 1}
df['vento'] = df['vento'].map(d)

d = {'nao': 0, 'sim': 1}
df['joga'] = df['joga'].map(d)

# separando as colunas de variaveis e classes
features = ['aparencia', 'temperatura', 'umidade', 'vento']
x = df[features]
y = df['joga']

# cria o modelo (classificador)
model = DecisionTreeClassifier()

# treina modelo
model = model.fit(x, y)

# cria e salva o diagrama da arvore de decisao
dot = tree.export_graphviz(model, out_file=None,
                           feature_names=features,
                           class_names=['nao', 'sim'],
                           filled=True)
graph = graphviz.Source(dot, format="png")
graph.render("dt")

# faz uma previsao de um exemplo existente (d5), resultado: sim (1)
# aparencia: chuvoso (1); temperatura: fria (0); umidade: normal (1); vento: fraco (0)
print(model.predict([[1, 0, 1, 0]]))

# faz uma previsao de um novo exemplo
# aparencia: ensolarado (2); temperatura: quente (1); umidade: normal (1); vento: forte (1)
print(model.predict([[2, 1, 1, 1]]))