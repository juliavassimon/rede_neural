import pandas as pd

previsores = pd.read_csv('entradas_breast.csv') #atributos previsores
classe = pd.read_csv('saidas_breast.csv') #respostas/classe pra fazer a previsao

from sklearn.model_selection import train_test_split
previsores_treinamentos, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
#primeira camada oculta
classificador.add(Dense(units=16, activation='relu', 
                        kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dense(units=16, activation='relu', 
                        kernel_initializer='random_uniform', input_dim=30))
#camada de saida
classificador.add(Dense(units=1, activation='sigmoid'))

#configuração/parametro para compilar
#paremetros adam para otimizar melhor
otimizador = tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=0.5)
#classificador.compile(optimizer='adam', loss = 'binary_crossentropy',
                     #metrics=['binary_accuracy'])
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
metrics=['binary_accuracy'])

classificador.fit(previsores_treinamentos, classe_treinamento,  
                  steps_per_epoch=426, epochs=100)
#criar variaveis para ver o peso
peso0 = classificador.layers[0].get_weights() 
print(peso0)
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()
#lembrando que está treinando e fazendo teste com esse base de dados
#fazer as previsoes com a base de dados de teste
previsoes = classificador.predict(previsores_teste) 
#avaliação com algoritmo com base de dados teste forma manual
#converter os valores da probabilidade em 0 ou 1
previsoes = (previsoes >0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matrix = confusion_matrix(classe_teste, previsoes)
# faz a avaliação de forma automatica, utilizando o keras
resultado = classificador.evaluate(previsores_teste, classe_teste)




















