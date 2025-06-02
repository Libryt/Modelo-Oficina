import pandas as pd
import matplotlib.pyplot as plt # Lib para gráficos
from sklearn.tree import plot_tree # Lib para plotar árvore de decisão
from sklearn import tree, svm # Libs para modelos de Árvore de Decisão e SVM
from sklearn.preprocessing import StandardScaler # Lib para escalonar features
from sklearn.metrics import accuracy_score, classification_report # Libs para métricas de avaliação
from sklearn.model_selection import train_test_split # Lib para dividir dados

# --- 1. Carregamento e Preparação Inicial dos Dados ---
try:
    df = pd.read_csv('oficina_Britt.csv') # Carrega os dados do arquivo CSV
except FileNotFoundError:
    print("Erro: O arquivo 'oficina_Britt.csv' não foi encontrado.")
    exit() # Encerra se o arquivo não existir

df.columns = df.columns.str.strip() # Remove espaços extras dos nomes das colunas

# Define a variável alvo (y) e as features brutas (X_raw)
y = df['Avaliacao_Cliente']
X_raw = df.drop('Avaliacao_Cliente', axis=1)

# Aplica One-Hot Encoding na feature categórica 'Servico'
X = pd.get_dummies(X_raw, columns=['Servico'], prefix='Servico', dtype=int)

# Divide os dados em conjuntos de treino e teste, estratificando pelo alvo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Variáveis para armazenar modelos e scaler (evita reprocessamento)
dt_model = None # Modelo de Árvore de Decisão
svm_model = None # Modelo SVM
scaler = None # Objeto StandardScaler
X_train_scaled = None # Features de treino escalonadas
X_test_scaled = None # Features de teste escalonadas


# --- 2. Loop Principal do Menu Interativo ---
while True:
    print("\nMenu principal\nEscolha uma opção: \n1 - Árvore de Decisão\n2 - SVM\n3 - Encerrar programa")
    try:
        main_choice = int(input("Digite uma opção: ")) # Escolha do usuário no menu principal

        # --- Seção Árvore de Decisão ---
        if main_choice == 1:
            if dt_model is None: # Treina o modelo apenas se ainda não foi treinado
                dt_model = tree.DecisionTreeClassifier(random_state=1)
                dt_model.fit(X_train, y_train)

            while True: # Loop do submenu da Árvore de Decisão
                print("\nSeção Árvore de Decisão\nEscolha uma opção:")
                print("1 - Mostrar Desempenho\n2 - Mostrar Árvore\n3 - Fazer nova classificação\n4 - Voltar ao menu principal")
                try:
                    dt_choice = int(input("Digite uma opção: ")) # Escolha no submenu

                    if dt_choice == 1: # Mostrar Desempenho
                        dt_preds = dt_model.predict(X_test) # Previsões no conjunto de teste
                        print(f"\nAcurácia (Árvore de Decisão): {accuracy_score(y_test, dt_preds):.2f}")
                        print("-" * 30)
                        print("\nRelatório de Classificação (Árvore de Decisão):\n")
                        class_labels = [str(c) for c in sorted(y.unique())] # Nomes das classes para o relatório
                        print(classification_report(y_test, dt_preds, target_names=class_labels, zero_division=0))
                        print("-" * 30)

                    elif dt_choice == 2: # Mostrar Árvore
                        print("\n--- Visualização da Árvore de Decisão ---")
                        try:
                            feat_names = X.columns.tolist() # Nomes das features para o plot
                            class_labels_plot = sorted([str(c) for c in y.unique()]) # Nomes das classes para o plot

                            plt.figure(figsize=(20,10))
                            plot_tree(dt_model, feature_names=feat_names, class_names=class_labels_plot,
                                      filled=True, rounded=True, proportion=False, fontsize=7)
                            plt.title("Árvore de Decisão Treinada")
                            plt.show() # Exibe a árvore
                        except Exception as e:
                            print(f"\nOcorreu um erro ao visualizar a árvore: {e}")

                    elif dt_choice == 3: # Fazer Nova Classificação
                        print("\n--- Nova Classificação (Árvore de Decisão) ---")
                        try:
                            valid_services = X_raw['Servico'].unique() # Lista de serviços válidos
                            print(f"Serviços disponíveis: {', '.join(valid_services)}")
                            user_service = input(f"Digite o tipo de Serviço: ").strip()

                            if user_service not in valid_services: # Validação do serviço inserido
                                print(f"Erro: Serviço '{user_service}' inválido. Escolha entre: {', '.join(valid_services)}")
                                continue

                            # Coleta das demais features do usuário
                            user_parts = float(input("Valor das peças (ex.: 150.0): "))
                            user_labor = float(input("Valor da mão de obra (ex.: 70.0): "))
                            user_time = float(input("Tempo de serviço em horas (ex.: 2.5): "))
                            user_km = float(input("Quilometragem do carro (ex.: 85000): "))
                            user_year = int(input("Ano de fabricação (ex.: 2018): "))

                            # Cria DataFrame com a nova entrada
                            new_input_df = pd.DataFrame([[user_service, user_parts, user_labor, user_time, user_km, user_year]],
                                                        columns=X_raw.columns)
                            # Processa a nova entrada
                            new_input_processed = pd.get_dummies(new_input_df, columns=['Servico'], prefix='Servico', dtype=int)
                            # Reindexa para garantir as mesmas colunas do treino, preenchendo com 0 colunas faltantes
                            new_input_reindexed = new_input_processed.reindex(columns=X.columns, fill_value=0)

                            prediction = dt_model.predict(new_input_reindexed) # Realiza a previsão
                            print(f"\nPrevisão da Avaliação do Cliente: {prediction[0]}")

                        except ValueError:
                            print("\nErro: Entrada inválida. Verifique os tipos de dados.")
                        except Exception as e:
                             print(f"\nErro inesperado na classificação: {e}")

                    elif dt_choice == 4: # Voltar ao Menu Principal
                        break # Sai do submenu
                    else:
                        print("Opção inválida.")
                except ValueError:
                    print("Entrada inválida. Digite um número.")

        # --- Seção SVM ---
        elif main_choice == 2:
            if svm_model is None or scaler is None: # Prepara dados e treina SVM apenas se necessário
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train) # Ajusta o scaler e transforma dados de treino
                X_test_scaled = scaler.transform(X_test) # Transforma dados de teste

                svm_model = svm.SVC(random_state=1, kernel='linear', C=1.0)
                svm_model.fit(X_train_scaled, y_train) # Treina o modelo SVM

            while True: # Loop do submenu SVM
                print("\nSeção SVM\nEscolha uma opção:")
                print("1 - Mostrar Desempenho\n2 - Fazer nova classificação\n3 - Voltar ao menu principal")
                try:
                    svm_choice = int(input("Digite uma opção: ")) # Escolha no submenu SVM

                    if svm_choice == 1: # Mostrar Desempenho
                        svm_preds = svm_model.predict(X_test_scaled) # Previsões no teste escalonado
                        print(f"\nAcurácia (SVM): {accuracy_score(y_test, svm_preds):.2f}")
                        #print(f"\nAcurácia (SVM): {accuracy_score(y_test, svm_preds) * 100:.2f}%") CASO O PROFESSOR QUEIRA PORCENTAGEM
                        print("-" * 30)
                        print("\nRelatório de Classificação (SVM):\n")
                        class_labels = [str(c) for c in sorted(y.unique())] # Nomes das classes
                        print(classification_report(y_test, svm_preds, target_names=class_labels, zero_division=0))
                        print("-" * 30)

                    elif svm_choice == 2: # Fazer Nova Classificação
                        print("\n--- Previsão para Nova Entrada (SVM) ---")
                        try:
                            valid_services = X_raw['Servico'].unique() # Lista de serviços válidos
                            print(f"Serviços disponíveis: {', '.join(valid_services)}")
                            user_service = input(f"Digite o tipo de Serviço: ").strip()

                            if user_service not in valid_services: # Validação do serviço
                                print(f"Erro: Serviço '{user_service}' inválido. Escolha entre: {', '.join(valid_services)}")
                                continue

                            # Coleta das demais features
                            user_parts = float(input("Valor das peças (ex.: 150.0): "))
                            user_labor = float(input("Valor da mão de obra (ex.: 70.0): "))
                            user_time = float(input("Tempo de serviço em horas (ex.: 2.5): "))
                            user_km = float(input("Quilometragem do carro (ex.: 85000): "))
                            user_year = int(input("Ano de fabricação (ex.: 2018): "))

                            # Cria DataFrame com a nova entrada
                            new_input_df = pd.DataFrame([[user_service, user_parts, user_labor, user_time, user_km, user_year]],
                                                        columns=X_raw.columns)
                            # Processa a nova entrada (One-Hot Encoding)
                            new_input_processed = pd.get_dummies(new_input_df, columns=['Servico'], prefix='Servico', dtype=int)
                            # Reindexa para consistência com colunas de treino
                            new_input_reindexed = new_input_processed.reindex(columns=X.columns, fill_value=0)
                            # Escala a nova entrada usando o scaler ajustado anteriormente
                            new_input_scaled = scaler.transform(new_input_reindexed)

                            prediction = svm_model.predict(new_input_scaled) # Realiza a previsão
                            print(f"\nPrevisão da Avaliação do Cliente (SVM): {prediction[0]}")

                        except ValueError:
                            print("\nErro: Entrada inválida. Verifique os tipos de dados.")
                        except Exception as e:
                            print(f"\nErro inesperado na classificação: {e}")

                    elif svm_choice == 3: # Voltar ao Menu Principal
                        break # Sai do submenu SVM
                    else:
                        print("Opção inválida.")
                except ValueError:
                    print("Entrada inválida. Digite um número.")

        elif main_choice == 3: # Encerrar Programa
            print("Programa encerrado.")
            break # Sai do loop principal
        else:
            print("Opção inválida.")
    except ValueError:
        print("Entrada inválida. Digite um número para a opção do menu.")