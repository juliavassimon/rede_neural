import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
#validação cruzada
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier, KerasRegressor
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(): #função para criar a rede neural
    classificador = Sequential()
    #primeira camada oculta com 30 neuronios
    classificador.add(Dense(units=16, activation='relu', 
                            kernel_initializer='random_uniform', input_dim=30))
    #adicionar um droupout 
    classificador.add(Dropout(0.2))
    #segunda camada oculta
    classificador.add(Dense(units=16, activation='relu', 
                            kernel_initializer='random_uniform', input_dim=30))
    #adicionar um droupout 
    classificador.add(Dropout(0.2))
    #camada de saida
    classificador.add(Dense(units=1, activation='sigmoid'))

    #configuração/parametro para compilar
    #paremetros adam para otimizar melhor
    otimizador = tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=0.5)
    #classificador.compile(optimizer='adam', loss = 'binary_crossentropy',
                         #metrics=['binary_accuracy'])
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
    metrics=['binary_accuracy'])
    return classificador
#Criar a variável que será a rede neural fora da função
classificador = KerasClassifier(model = criarRede, epochs =100, batch_size=10)
#fazer os testes 
resultados = cross_val_score(estimator = classificador, X = previsores,
                             y= classe, cv =10, scoring = 'accuracy')
#saber a media %
media = resultados.mean()
#desvio padrão
desvio = resultados.std()























    


