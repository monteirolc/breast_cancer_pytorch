import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
# print(torch.__version__)
import torch.nn as nn

np.random.seed(123)
torch.manual_seed(123)

previsores = pd.read_csv('data/entradas_breast.csv')
classe = pd.read_csv('data/saidas_breast.csv')

np.unique(classe)
sns.countplot(classe['0'])


[previsores_treinamento, previsores_teste,
 classe_treinamento, classe_teste] = train_test_split(
    previsores, classe, test_size=0.25)


previsores_treinamento = torch.tensor(
    np.array(previsores_treinamento), dtype=torch.float)
classe_treinamento = torch.tensor(
    np.array(classe_treinamento), dtype=torch.float)

dataset = torch.utils.data.TensorDataset(
    previsores_treinamento, classe_treinamento)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=10, shuffle=True)

'''CONSTRUÇÃO DO MODELO - REDE NEURAL '''
# 30 -> 16 -> 16 -> 1
# (ENTRADAS + SAÍDAS) / 2 = (30 + 1)/ 2 ~= 16

classificador = nn.Sequential(
    nn.Linear(in_features=30, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    classificador.parameters(), lr=0.001, weight_decay=0.0001)

'''TREINAMENTO DO MODELO '''
for epoch in range(300):
    running_loss = 0.
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()  # zera o gradiente
        outputs = classificador(inputs)  # realiza os calcúlos dos somatórios
        loss = criterion(outputs, labels)  # calcúlo dos erros
        loss.backward()  # volta ao inicio
        optimizer.step()  # atualiza os pesos

        running_loss += loss.item()
    # print('Época %3d: perda %.5f' %
    # (epoch+1, running_loss/len(train_loader)))

''' AVALIAÇÃO DO MODELO'''

classificador.eval()  # põe o classificador em modo de avaliação
# transforma os previsores em array numpy e depois em tensor do pytorch
previsores_teste = torch.tensor(np.array(previsores_teste), dtype=torch.float)
previsoes = classificador.forward(previsores_teste)

previsoes = np.array(previsoes > 0.5)
# verificação da taxa de acerto
taxa_acerto = accuracy_score(classe_teste, previsoes)
print(taxa_acerto)


matriz = confusion_matrix(classe_teste, previsoes)
sns.jointplot(data=matriz)
sns.heatmap(matriz, annot=True)
