import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline # Importar Pipeline

# 1. Leitura dos dados
try:
    dados = pd.read_csv('oficina_Britt.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'oficina_Britt.csv' não foi encontrado. Verifique o caminho.")
    exit()

dados.columns = dados.columns.str.strip()

# 2. Definição da variável alvo (alvo) e das features (features_originais)
alvo = dados['Avaliacao_Cliente']
features_originais = dados.drop('Avaliacao_Cliente', axis=1)

# 3. Aplicação do pd.get_dummies na feature categórica 'Servico'
# O Pipeline não lida nativamente com a transformação de colunas específicas para get_dummies
# de forma simples dentro de si para DataFrames, então mantemos este passo fora do pipeline
# ou usaríamos ColumnTransformer para uma abordagem mais complexa e integrada ao pipeline.
# Para este exemplo, manteremos o get_dummies fora para simplicidade,
# pois o pipeline será focado no escalonamento de features já numéricas.
features_processadas = pd.get_dummies(features_originais, columns=['Servico'], prefix='Servico', dtype=int)

# 4. Definição dos dados de treinamento e de teste
# Usamos stratify=alvo para manter a proporção das classes
X_treino, X_teste, alvo_treino, alvo_teste = train_test_split(
    features_processadas, alvo, test_size=0.3, random_state=1, stratify=alvo
)

# 5. Criação e Aprendizado do Pipeline SVM
# O Pipeline irá primeiro escalonar os dados (StandardScaler) e depois aplicar o SVC.
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Etapa de escalonamento
    ('svc', svm.SVC(kernel='linear', C=1.0)) # Etapa do classificador SVM
])

# Treina o pipeline. O scaler será ajustado (fit_transform) nos dados de treino
# e o SVC será treinado com os dados de treino já escalonados, tudo internamente.
svm_pipeline.fit(X_treino, alvo_treino)

# 6. Mostrar desempenho
# Ao usar predict com o pipeline, os dados de teste são automaticamente transformados (escalonados)
# antes da predição.
previsoes_no_teste_svm = svm_pipeline.predict(X_teste)
print(f"Acurácia do modelo SVM com Pipeline: {accuracy_score(alvo_teste, previsoes_no_teste_svm):.2f}")
print("-" * 30)

# 6.1 Relatório de Classificação
print("\nRelatório de Classificação SVM com Pipeline:\n")
try:
    target_names = [str(c) for c in sorted(alvo.unique())]
    print(classification_report(alvo_teste, previsoes_no_teste_svm, target_names=target_names, zero_division=0))
except Exception as e:
    print(f"Não foi possível gerar nomes de classe para o relatório, usando padrão: {e}")
    print(classification_report(alvo_teste, previsoes_no_teste_svm, zero_division=0))
print("-" * 30)


# 7. Mostrar classificação para uma nova entrada do usuário
print("\n--- Previsão para Nova Entrada (SVM com Pipeline) ---")
try:
    servicos_disponiveis = features_originais['Servico'].unique()

    servico_usuario = input(f"Digite o tipo de Serviço \n({', '.join(servicos_disponiveis)}): ")
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
    # Garante que a nova entrada tenha as mesmas colunas que os dados de treino (X_treino)
    # Isso é crucial pois o pipeline espera a mesma estrutura de features que foi usada no fit.
    nova_entrada_reindexada_df = nova_entrada_processada_df.reindex(columns=X_treino.columns, fill_value=0)

    # O pipeline aplica o escalonamento e a predição automaticamente.
    resultado_previsao_svm = svm_pipeline.predict(nova_entrada_reindexada_df)
    print(f"\nPrevisão da Avaliação do Cliente para a nova entrada (SVM com Pipeline): {resultado_previsao_svm[0]}")

except ValueError:
    print("\nErro: Entrada inválida. Por favor, verifique os tipos de dados e tente novamente.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")