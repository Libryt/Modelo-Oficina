import pandas as pd
from sklearn import tree # Módulo para árvores de decisão
from sklearn.metrics import accuracy_score # Métrica de acurácia
from sklearn.model_selection import train_test_split # Função para dividir os dados
import matplotlib.pyplot as plt # Para plotar a árvore
from sklearn.tree import plot_tree # Função para plotar a árvore

# 1. Leitura dos dados
try:
    dados = pd.read_csv('oficina_Britt.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'oficina_Britt.csv' não foi encontrado. Verifique o caminho.")
    exit()

dados.columns = dados.columns.str.strip() # Remove espaços dos nomes das colunas

# 2. Definição da variável alvo (alvo) e das features (features_originais) antes do pré-processamento
alvo = dados['Avaliacao_Cliente']
features_originais = dados.drop('Avaliacao_Cliente', axis=1)

# 3. Aplicação do pd.get_dummies na feature categórica 'Servico'
features_processadas = pd.get_dummies(features_originais, columns=['Servico'], prefix='Servico', dtype=int)

# 4. Definição dos dados de treinamento e de teste (COM STRATIFY)
# Adicionado stratify=alvo para replicar a condição que resultava em ~47% de acurácia
X_treino, X_teste, alvo_treino, alvo_teste = train_test_split(
    features_processadas, alvo, test_size=0.3, random_state=1, stratify=alvo
)

# 5. Aprendizado do modelo de Árvore de Decisão
modelo_arvore = tree.DecisionTreeClassifier(random_state=1)
modelo_arvore.fit(X_treino, alvo_treino)

# 6. Mostrar desempenho em PORCENTAGEM
previsoes_no_teste = modelo_arvore.predict(X_teste)
print(f"Acurácia do modelo: {accuracy_score(alvo_teste, previsoes_no_teste):.2f}")
print("-" * 30)

# 7. Mostrar classificação para uma nova entrada do usuário
print("\n--- Previsão para Nova Entrada ---")
try:
    servicos_disponiveis = features_originais['Servico'].unique()
    if servicos_disponiveis.size > 0:
        servico_usuario = input(f"Digite o tipo de Serviço (ex: {servicos_disponiveis[0]}, disponíveis: {', '.join(servicos_disponiveis)}): ")
    else:
        servico_usuario = input("Digite o tipo de Serviço: ")

    pecas_usuario = float(input("Digite o valor das peças (ex.: 150.0): "))
    mao_obra_usuario = float(input("Digite o valor da mão de obra (ex.: 70.0): "))
    tempo_usuario = float(input("Digite o tempo de serviço em horas (ex.: 2.5): "))
    km_usuario = float(input("Digite a quilometragem do carro (ex.: 85000): "))
    ano_usuario = int(input("Digite o ano de fabricação do carro (ex.: 2018): "))

    nova_entrada_df = pd.DataFrame([[
        servico_usuario, pecas_usuario, mao_obra_usuario,
        tempo_usuario, km_usuario, ano_usuario
    ]], columns=features_originais.columns)

    nova_entrada_processada_df = pd.get_dummies(nova_entrada_df, columns=['Servico'], prefix='Servico', dtype=int)
    nova_entrada_reindexada_df = nova_entrada_processada_df.reindex(columns=X_treino.columns, fill_value=0)

    resultado_previsao = modelo_arvore.predict(nova_entrada_reindexada_df)
    print(f"\nPrevisão da Avaliação do Cliente para a nova entrada: {resultado_previsao[0]}")

except ValueError:
    print("\nErro: Entrada inválida. Por favor, verifique os tipos de dados e tente novamente.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado durante a previsão para nova entrada: {e}")

# 8. Visualização da Árvore de Decisão
print("\n--- Visualização da Árvore de Decisão ---")
try:
    feature_names = features_processadas.columns.tolist()
    class_names = sorted([str(c) for c in alvo.unique()])

    plt.figure(figsize=(20,10))
    plot_tree(modelo_arvore,
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              proportion=False,
              fontsize=7)
    plt.title("Árvore de Decisão Treinada")
    plt.show()

except Exception as e:
    print(f"\nOcorreu um erro ao tentar visualizar a árvore: {e}")
    print("Certifique-se de que a biblioteca matplotlib está instalada ('pip install matplotlib').")