import numpy as np 
import pandas as pd 
import os, random

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore', message="DataFrame is highly fragmented")

import pickle

base_dir = "C:\\Users\\Nick\\Downloads\\gs\\files"

class_folder_names = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

image_paths = []

for cls in class_folder_names:
    cls_path = os.path.join(base_dir, cls)  # Junta corretamente o caminho do diretório
    if os.path.isdir(cls_path):  # Verifica se é um diretório
        for file_name in os.listdir(cls_path):
            if file_name.endswith('.jpg'):  # Verifica arquivos jpg
                image_paths.append(os.path.join(cls_path, file_name))

print("O total de imagens = ", len(image_paths))
print("-------------------------------")

image_paths[0:5]

classes = []

for image_path in image_paths:
    classes.append(image_path.split('\\')[-2])
    
classes[0:5]

inputs = []

for i in tqdm(image_paths):  # image_paths[0:2]
    image = load_img(i)
    img_array = img_to_array(image)
    inputs.append(img_array)

X = np.array(inputs)

le = LabelEncoder()
y = le.fit_transform(classes)
y = np.array(y)

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Dados de treinamento = ", X_train.shape,  y_train.shape)
print("Dados de teste = ", X_test.shape,  y_test.shape)

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

print("Dados de treinamento achatados = ", X_train_flattened.shape,  y_train.shape)
print("Dados de teste achatados = ", X_test_flattened.shape,  y_test.shape)

# Crie um dicionário para armazenar os índices de amostras para cada classe

unique_classes = np.unique(y_train)
class_indices = {class_id: np.where(y_train == class_id)[0] for class_id in unique_classes}
class_indices

# Plotando o total de amostras para cada classe

images_count = [len(class_indices[key]) for key in class_indices.keys()]

fig = px.bar(x=class_folder_names, y=images_count, color=class_folder_names)

fig.update_layout(xaxis_title='Doença', yaxis_title='Contagem', title="Total de amostras para cada classe")
fig.update_traces(texttemplate='%{y}', textposition='inside')

fig.show()

# Exibindo imagens de amostra para cada classe aleatoriamente.

plt.figure(figsize=(12, 5))

for i, (class_id, indices) in enumerate(class_indices.items()):
    random_index = np.random.choice(indices)
    random_image = X_train[random_index] 

    plt.subplot(1, len(unique_classes), i + 1)
    plt.imshow(random_image)
    plt.title(class_folder_names[class_id])
    plt.axis('off')

plt.show()

# Exibindo 7 imagens de amostra para cada classe aleatoriamente.

selected_classes = [0, 1, 2]
total_images_per_class = 7

plt.figure(figsize=(15, 5))

for c, selected_class in enumerate(selected_classes):
    
    # Obter índices de amostras para a classe selecionada
    indices_for_selected_class = np.where(y_train == selected_class)[0]
    random_indices = np.random.choice(indices_for_selected_class, total_images_per_class, replace=False)

    # Exibir imagens para a classe selecionada atual
    for i, idx in enumerate(random_indices):
        plt.subplot(3, 7, c * total_images_per_class + i + 1)
        plt.imshow(X_train[idx])
        plt.title(class_folder_names[selected_class])
        plt.axis('off')

plt.tight_layout()
plt.show()

# Redimensionar os dados da imagem para um array 2D
num_samples, height, width, channels = X_train.shape

X_train_reshaped = X_train.reshape(num_samples, height * width * channels)
X_train_flattened.shape

# Aplicar PCA para reduzir as dimensões para 2 para visualização

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_flattened)
X_train_pca.shape

# Visualizar os dados transformados pelo PCA

hover_text = [f"Índice: {index}" for index in range(len(X_train_pca))]

fig = px.scatter(x=X_train_pca[:, 0], y=X_train_pca[:, 1], color=y_train, hover_name=hover_text, symbol=y_train, title='Visualização PCA das Classes de Imagens')
fig.update_traces(marker=dict(size=15))
fig.update_layout(xaxis_title='Componente Principal 1', yaxis_title='Componente Principal 2')
fig.update_layout(coloraxis_showscale=False)

fig.show()

# Encontrando outliers ao analisar valores altos de soma de distâncias

pca_sums = np.sum(X_train_pca, axis=1)

outlier_indexes = []
for idx, row in enumerate(pca_sums):
    if row > 15000:
        print(row, idx)
        outlier_indexes.append(idx)

# Exibindo imagens de outliers

plt.figure(figsize=(15, 4))

for i, outlier in enumerate(outlier_indexes):
    outlier_image = X_train[outlier] 

    plt.subplot(1, len(outlier_indexes), i + 1)
    plt.imshow(outlier_image)
    title = class_folder_names[y_train[outlier]]
    plt.title(title)
    plt.axis('off')

plt.show()

pca_df = pd.DataFrame(data=X_train_pca, columns=['PC1', 'PC2'])
pca_df['Classe'] = y_train
pca_df['Índice'] = pca_df.index
pca_df.head()

# Criar gráficos de caixa lado a lado para PC1 e PC2 usando Plotly

fig = px.box(pca_df, x='Classe', y=['PC1', 'PC2'], points="all", facet_col="variable",
             title='Gráficos de Caixa PCA - Componentes Principais 1 e 2 por Classe', hover_data={'Índice': True})
fig.update_layout(width=1200, height=500)
fig.show()

models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
}

accuracies = {}

for name, model in models.items():
    
    model.fit(X_train_flattened, y_train)        
    predictions = model.predict(X_test_flattened)
    
    accuracy = accuracy_score(y_test, predictions)
    accuracies[name] = accuracy

    print(f"Acurácia do {name}: {accuracy}")

# Encontre o melhor modelo com base na precisão
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

print("Melhor modelo = ", best_model )

# Treine o melhor modelo
best_model.fit(X_train_flattened, y_train)
best_predictions = best_model.predict(X_test_flattened)

best_predictions

# Obter pesos

if best_model_name == 'Random Forest' or best_model_name == 'Decision Tree':
    coefficients = best_model.feature_importances_
    print(coefficients.shape)

elif best_model_name == 'Logistic Regression':
    coefficients = best_model.coef_.ravel()
    print(coefficients.shape)
else:
    coefficients = None

# Plotando a distribuição dos pesos

if coefficients is not None:
    fig = px.histogram(x=coefficients, nbins=50, labels={'x': 'Valor do Coeficiente'}, title='Distribuição dos Coeficientes (Pesos)')
    fig.update_layout(bargap=0.1)
    fig.update_traces(opacity=0.7)
    fig.show()

report = classification_report(y_test, best_predictions)
print(report)

# Calcular e exibir a matriz de confusão para o melhor modelo
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title(f'Matriz de Confusão para o Melhor Modelo com base na Acurácia ({best_model_name})')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# Salvar o modelo treinado
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Carregar o modelo salvo
with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

loaded_model

# Previsões no modelo carregado com X_test_flattened

predictions = loaded_model.predict(X_test_flattened)
predictions

# Previsão de uma única imagem no modelo carregado

random_image_path = random.choice(image_paths)
actual_class = random_image_path.split('\\')[-2]

print("Caminho da Imagem Aleatória = ", random_image_path)
print("Sua classe original = ", actual_class)

# Pré-processamento de imagem para previsão

random_image = load_img(random_image_path)
random_img_array = img_to_array(random_image)

flattened_img_array = random_img_array.reshape(1, -1)
flattened_img_array.shape

predictions = loaded_model.predict(flattened_img_array)

predicted_class = class_folder_names[predictions[0]]
actual_class = random_image_path.split('\\')[-2]

plt.imshow(random_image)
plt.title(f"Classe Prevista: {predicted_class}\nClasse Real: {actual_class}")
plt.axis('off')
plt.show()