# -*- coding: utf-8 -*-
# Importar as bibliotecas necessárias para este projeto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visuals as vs  # Supplementary code
from sklearn.cross_validation import ShuffleSplit

# Executar o conjunto de dados de imóveis de Boston
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# Êxito
print "O conjunto de dados de imóveis de Boston tem {} pontos com {} variáveis em cada.".format(*data.shape)
# print data.head(30)

# TODO: Preço mínimo dos dados
minimum_price = min(prices)

# TODO: Preço máximo dos dados
maximum_price = max(prices)

# TODO: Preço médio dos dados
mean_price = np.mean(prices)

# TODO: Preço mediano dos dados
median_price = np.median(prices)

# TODO: Desvio padrão do preço dos dados
std_price = np.std(prices)

# Mostrar as estatísticas calculadas
print "Estatísticas para os dados dos imóveis de Boston:\n"
print "Preço mínimo: ${:,.2f}".format(minimum_price)
print "Preço máximo: ${:,.2f}".format(maximum_price)
print "Preço médio: ${:,.2f}".format(mean_price)
print "Preço mediano: ${:,.2f}".format(median_price)
print "Desvio padrão dos preços: ${:,.2f}".format(std_price)

plt.plot(data['RM'], data['MEDV'], 'ro', alpha=0.5)
plt.xlabel('RM')
plt.ylabel('MEDV')
# plt.show()
plt.plot(data['LSTAT'], data['MEDV'], 'ro', alpha=0.5)
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
# plt.show()
plt.plot(data['PTRATIO'], data['MEDV'], 'ro', alpha=0.5)
plt.xlabel('PTRATIO')
plt.ylabel('MEDV')
# plt.show()

# question 2
from sklearn.metrics import r2_score


def performance_metric(y_true, y_predict):
    """ Calcular e retornar a pontuação de desempenho entre
        valores reais e estimados baseado na métrica escolhida. """
    score = r2_score(y_true, y_predict)
    # Devolver a pontuação
    return score


score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "O coeficiente de determinação, R^2, do modelo é {:.3f}.".format(score)

# TODO: Importar 'train_test_split'
from sklearn.cross_validation import train_test_split

# TODO: Misturar e separar os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

# Êxito
print "Separação entre treino e teste feita com êxito."

# question 3
# Criar curvas de aprendizagem para tamanhos de conjunto de treinamento variável e profundidades máximas
vs.ModelLearning(features, prices)

# TODO: Importar 'make_scorer', 'DecisionTreeRegressor' e 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV


def fit_model(X, y):
    """ Desempenhar busca em matriz sobre o parâmetro the 'max_depth' para uma
        árvore de decisão de regressão treinada nos dados de entrada [X, y]. """

    # Gerar conjuntos de validação-cruzada para o treinamento de dados
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)

    # TODO: Gerar uma árvore de decisão de regressão de objeto
    regressor = DecisionTreeRegressor()

    # TODO: Gerar um dicionário para o parâmetro 'max_depth' com um alcance de 1 a 10
    params = {'max_depth': range(1, 11)}

    # TODO: Transformar 'performance_metric' em uma função de pontuação utilizando 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Gerar o objeto de busca em matriz
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)

    # Ajustar o objeto de busca em matriz com os dados para calcular o modelo ótimo
    grid = grid.fit(X, y)

    # Devolver o modelo ótimo depois de realizar o ajuste dos dados
    return grid.best_estimator_

# Ajustar os dados de treinamento para o modelo utilizando busca em matriz
reg = fit_model(X_train, y_train)

# Produzir valores para 'max_depth'
print "O parâmetro 'max_depth' é {} para o modelo ótimo.".format(reg.get_params()['max_depth'])

# Gerar uma matriz para os dados do cliente
client_data = [[5, 17, 15],  # Cliente 1
               [4, 32, 22],  # Cliente 2
               [8, 3, 12]]  # Cliente 3

# Mostrar estimativas
for i, price in enumerate(reg.predict(client_data)):
    print "Preço estimado para a casa do cliente {}: ${:,.2f}".format(i + 1, price)

clients = np.transpose(client_data)
predicts = reg.predict(client_data)

for i, feat in enumerate(['RM', 'LSTAT', 'PTRATIO']):
    plt.plot(features[feat], prices, 'ro', alpha=0.4)
    plt.plot(clients[i], predicts, 'ko', alpha=1)
    plt.xlabel(feat)
    plt.ylabel('MEDV')
    plt.show()
