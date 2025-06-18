import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
features_processadas = pd.get_dummies(features_originais, columns=['Servico'], prefix='Servico', dtype=int)

# 4. Definição dos dados de treinamento e de teste
X_treino, X_teste, alvo_treino, alvo_teste = train_test_split(
    features_processadas, alvo, test_size=0.3, random_state=1)

# 4.1 Escalonamento das features (IMPORTANTE para SVM)
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# 5. Aprendizado do modelo SVM

modelo_svm = svm.SVC(kernel='linear', C=1.0)
modelo_svm.fit(X_treino_scaled, alvo_treino)

# 6. Mostrar desempenho
previsoes_no_teste_svm = modelo_svm.predict(X_teste_scaled)
print(f"Acurácia do modelo SVM: {accuracy_score(alvo_teste, previsoes_no_teste_svm):.2f}")
print("-" * 30)

# 6.1 Relatório de Classificação
print("\nRelatório de Classificação SVM:\n")
try:
    target_names = [str(c) for c in sorted(alvo.unique())] # Usar alvo.unique() do dataset original para pegar todos os nomes de classe possíveis
    print(classification_report(alvo_teste, previsoes_no_teste_svm, target_names=target_names, zero_division=0))
except Exception as e:
    print(f"Não foi possível gerar nomes de classe para o relatório, usando padrão: {e}")
    print(classification_report(alvo_teste, previsoes_no_teste_svm, zero_division=0))
print("-" * 30)


# 7. Mostrar classificação para uma nova entrada do usuário
print("\n--- Previsão para Nova Entrada (SVM) ---")
try:
    servicos_disponiveis = features_originais['Servico'].unique()

    servico_usuario = input(f"Digite o tipo de Serviço: \n ({', '.join(servicos_disponiveis)}): ")
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

    nova_entrada_scaled = scaler.transform(nova_entrada_reindexada_df)

    resultado_previsao_svm = modelo_svm.predict(nova_entrada_scaled)
    print(f"\nPrevisão da Avaliação do Cliente para a nova entrada (SVM): {resultado_previsao_svm[0]}")

except ValueError:
    print("\nErro: Entrada inválida. Por favor, verifique os tipos de dados e tente novamente.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")