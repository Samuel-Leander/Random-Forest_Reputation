from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import seaborn as sns
import os

os.environ['OMP_NUM_THREADS'] = '1'

# Carregamento da base de dados
dataset = pd.read_csv('Numpy e Estatística/csv_result-ebay_confianca_completo.csv')

print(dataset.shape)
# print(dataset.head())

dataset['blacklist'] = dataset['blacklist'] == 'S'
# dataset.dropna(inplace = True) # O dataset não tem linhas vazias
sns.countplot(x=dataset['reputation']) # reputation é a coluna de número 75
plt.show()

X = dataset.iloc[:, 0:74].values
y = dataset.iloc[:, 74].values
# print(X.shape)
# print(y.shape)

# Base de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

print(np.unique(y, return_counts=True)) # retorna o número de cada elemento no dataset
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# Classificação com Random Forest
modelo1 = RandomForestClassifier()
modelo1.fit(X_train, y_train)
previsao1 = modelo1.predict(X_test)
ac1 = accuracy_score(previsao1, y_test) # precisão do modelo
print(ac1)

# Undersampling - TomekLinks
tml = TomekLinks(sampling_strategy='majority') # majority = 'apaga registros da classe majoritária'
X_under, y_under = tml.fit_resample(X,y)

# print(X_under.shape, y_under.shape) 
# print(np.unique(y_under, return_counts=True))

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_under, y_under, test_size=0.2, 
                                                            stratify=y_under, random_state = 1)
modelo2 = RandomForestClassifier()
modelo2.fit(X_train_u, y_train_u)
previsao2 = modelo2.predict(X_test_u)
ac2 = accuracy_score(previsao2, y_test_u)
print(ac2)

# Oversampling - SMOTE
smote = SMOTE(sampling_strategy='minority') # aumenta os registro da classe minoritária
X_over, y_over = smote.fit_resample(X,y)

# print(X_over.shape, y_over.shape) 
# print(np.unique(y_over, return_counts=True)

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_over, y_over, test_size=0.2, 
                                                            stratify=y_over, random_state = 1)
modelo3 = RandomForestClassifier()
modelo3.fit(X_train_o, y_train_o)
previsao3 = modelo3.predict(X_test_o)
ac3 = accuracy_score(previsao3, y_test_o)
print(ac3)