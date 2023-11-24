Leticia Pfitzenmeier, RM 84906
Nicolas Garcia, RM 84635
Matheus de Araújo Ruck, RM 82360

# Classificação de Imagens de Doenças usando Machine Learning

Este repositório contém um projeto de classificação de imagens de doenças usando machine learning. O objetivo é explorar e demonstrar várias etapas do processo de construção de um modelo de aprendizado de máquina para classificar diferentes classes de imagens de doenças.

## Observação sobre ambiente de execução

O script `main.py` foi escrito para ser executado em um ambiente Windows. Caso deseje adaptá-lo para ser executado
em um ambiente Linux, as seguintes linhas devem ser alteradas: 51, 279 e 295. O delimitador deve ser alterado
de `'\\'` para `'/'`.

## Imagens

O conjunto de dados utilizado está localizado em `files/` e contém imagens de várias classes de doenças. O script lê e pré-processa essas imagens para serem usadas no treinamento e teste do modelo.

**IMPORTANTE:** a linha 30 do script `main.py` define a variável `base_dir`. O caminho é absoluto. Mude o valor
da variável quando baixar o projeto antes de executar.

```bash
https://www.kaggle.com/datasets/trainingdatapro/computed-tomography-ct-of-the-brain/download?datasetVersionNumber=1
```

## Exploração de Dados

O script inclui visualizações interativas e gráficos que fornecem insights sobre o conjunto de dados. Isso inclui gráficos de barras mostrando o total de amostras para cada classe e visualizações de imagens de amostra para cada classe.

## Análise de Componentes Principais (PCA)

O script utiliza a Análise de Componentes Principais (PCA) para reduzir as dimensões das imagens e visualizá-las em um espaço bidimensional. Isso ajuda a entender a distribuição dos dados no conjunto de treinamento.

## Modelagem e Avaliação

Vários modelos de aprendizado de máquina são treinados, incluindo Random Forest, Decision Tree, Logistic Regression, SVM e K-Nearest Neighbors. A acurácia de cada modelo é calculada e comparada para identificar o modelo mais eficaz.

## Análise de Pesos e Importância de Recursos

Para modelos como Random Forest, Decision Tree e Logistic Regression, a importância dos recursos (pesos) é analisada e visualizada. Isso fornece informações sobre quais características são mais relevantes para a classificação.

## Avaliação do Melhor Modelo

O modelo com a maior acurácia é selecionado, treinado novamente e avaliado usando métricas como precisão, recall e F1-score. Uma matriz de confusão é gerada para visualizar o desempenho do modelo.

## Salvando e Carregando o Modelo

O melhor modelo é salvo em um arquivo (`best_model.pkl`) usando a biblioteca `pickle`. O script também inclui uma seção para carregar o modelo salvo e realizar previsões, demonstrando como usar o modelo treinado para classificar novas imagens.

## Executando o Script

Para executar o script, certifique-se de ter todas as bibliotecas necessárias instaladas.

```bash
pip install numpy tqdm tensorflow scikit-learn keras seaborn matplotlib plotly jupyter ipywidgets pandas
```

O caminho do diretório do conjunto de dados deve ser ajustado conforme necessário. O script pode ser executado linha por linha em um ambiente Python, como Jupyter Notebook ou em um arquivo Python independente.

Este projeto fornece uma introdução abrangente ao processo de classificação de imagens usando machine learning, desde o pré-processamento de dados até a avaliação do modelo e a implementação prática para prever novas imagens.




