import pandas as pd
import matplotlib.pyplot as plt # Lib para gráficos
from sklearn.tree import plot_tree # Lib para plotar árvore de decisão
from sklearn import tree, svm # Libs para modelos de Árvore de Decisão e SVM
from sklearn.preprocessing import StandardScaler # Lib para escalonar features
from sklearn.metrics import accuracy_score, classification_report # Libs para métricas de avaliação
from sklearn.model_selection import train_test_split # Lib para dividir dados

# --- 1. Carregamento + Teste de CSV + Preparação Inicial dos Dados ---
try:
    dataframe_oficina = pd.read_csv('oficina_Britt.csv') # Carrega os dados do arquivo CSV
except FileNotFoundError:
    print("Erro: O arquivo 'oficina_Britt.csv' não foi encontrado.")
    exit() # Encerra se o arquivo não existir

dataframe_oficina.columns = dataframe_oficina.columns.str.strip() # Remove espaços extras dos nomes das colunas

dataframe_oficina['Servico'] = dataframe_oficina['Servico'].astype(str).str.strip() # Converte para string e remove espaços extras das linhas da tabela

# Armazena serviços disponíveis
lista_servicos_disponiveis = dataframe_oficina['Servico'].unique().tolist()

# Define a variável alvo (y_alvo) e as features brutas (X_features_originais)
y_alvo = dataframe_oficina['Avaliacao_Cliente']
X_features_originais = dataframe_oficina.drop('Avaliacao_Cliente', axis=1)
# Armazena nomes das colunas ANTES do get_dummies
nomes_colunas_originais = X_features_originais.columns.tolist()

# Transforma variável categórica 'Servico' em colunas numéricas
X_features_codificadas = pd.get_dummies(X_features_originais.copy(), columns=['Servico'], prefix='Servico', dtype=int) #copy() cria copia do dataframe para não modificar documento
#original(procedimento de segurança)

# Divide os dados em conjuntos de treino e teste, estratificando pelo alvo
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_features_codificadas, y_alvo, test_size=0.3, random_state=1)

# Variáveis para armazenar modelos e escalonador(Procedimento para evitar treinar várias vezes na mesma execução de código)
modelo_arvore_decisao = None
modelo_svm_oficina = None 
escalonador_features = None
X_treino_escalonado = None
X_teste_escalonado = None

# --- 2. Loop Principal do Menu Interativo ---
while True:
    print("\nMenu principal\nEscolha uma opção: \n1 - Árvore de Decisão\n2 - SVM\n3 - Encerrar programa")
    try:
        opcao_menu_principal = int(input("Digite uma opção: "))

        # --- Seção Árvore de Decisão ---
        if opcao_menu_principal == 1:
            #IF para verificar se o codigo ja foi treinado ou não, se foi ele ignora o treinamento.
            if modelo_arvore_decisao is None:
                modelo_arvore_decisao = tree.DecisionTreeClassifier()
                modelo_arvore_decisao.fit(X_treino, y_treino)

            while True: # Loop do submenu da Árvore de Decisão
                print("\nSeção Árvore de Decisão\nEscolha uma opção:")
                print("1 - Mostrar Desempenho\n2 - Mostrar Árvore\n3 - Fazer nova classificação\n4 - Voltar ao menu principal")
                try:
                    opcao_submenu_arvore = int(input("Digite uma opção: "))

                    if opcao_submenu_arvore == 1: # Mostrar Desempenho da árvore
                        predicoes_arvore = modelo_arvore_decisao.predict(X_teste)
                        print(f"\nAcurácia (Árvore de Decisão): {(accuracy_score(y_teste, predicoes_arvore) * 100):.2f}%")
                        print("-" * 30)
                    elif opcao_submenu_arvore == 2: # ________Mostrar Árvore_________
                        print("\n--- Visualização da Árvore de Decisão ---")
                        try:
                            nomes_features_plot_arvore = X_treino.columns.tolist() # Separa os serviços em lista para mostrar servico + regra logica(Servico <= X)

                            plt.figure(figsize=(20,10))
                            plot_tree(modelo_arvore_decisao, feature_names=nomes_features_plot_arvore,
                                      filled=True, rounded=True, proportion=False, fontsize=7)
                            plt.title("Árvore de Decisão Treinada")
                            plt.show()
                        except Exception as erro_plot_arvore:
                            print(f"\nOcorreu um erro ao visualizar a árvore: {erro_plot_arvore}")

                    elif opcao_submenu_arvore == 3: # Fazer Nova Classificação
                        print("\n--- Nova Classificação (Árvore de Decisão) ---")
                        fluxo_classificacao_interrompido_arvore = False
                        try:
                            servico_digitado_usuario = ""
                            while True: # Loop para obter entrada de serviço válida
                                print(f"Serviços disponíveis: {', '.join(lista_servicos_disponiveis)}")
                                if not lista_servicos_disponiveis:
                                    print("Nenhum serviço disponível para classificação. Voltando ao menu anterior.")
                                    fluxo_classificacao_interrompido_arvore = True
                                    break
                                
                                entrada_servico_usuario_temp = input(f"Digite o tipo de Serviço (ou 'cancelar' para voltar): ").strip()

                                if entrada_servico_usuario_temp.lower() == 'cancelar':
                                    print("Classificação cancelada.")
                                    fluxo_classificacao_interrompido_arvore = True
                                    break
                                
                                nome_servico_correspondente_arvore = None
                                for item_servico_lista in lista_servicos_disponiveis:
                                    if entrada_servico_usuario_temp.lower() == item_servico_lista.lower():
                                        nome_servico_correspondente_arvore = item_servico_lista
                                        break
                                
                                if nome_servico_correspondente_arvore is None:
                                    print(f"Erro: Serviço '{entrada_servico_usuario_temp}' inválido ou não encontrado. Por favor, tente novamente ou digite 'cancelar'.")
                                else:
                                    servico_digitado_usuario = nome_servico_correspondente_arvore
                                    print(f"Serviço selecionado: {servico_digitado_usuario}")
                                    break
                            
                            if fluxo_classificacao_interrompido_arvore:
                                continue

                            try:
                                #input para o usuario
                                nome_coluna_pecas_arvore = nomes_colunas_originais[1]
                                valor_pecas_usuario_arvore = float(input(f"Valor das peças (para '{nome_coluna_pecas_arvore}', ex.: 150.0): "))

                                nome_coluna_mao_obra_arvore = nomes_colunas_originais[2]
                                valor_mao_obra_usuario_arvore = float(input(f"Valor da mão de obra (para '{nome_coluna_mao_obra_arvore}', ex.: 70.0): "))
                                
                                nome_coluna_horas_arvore = nomes_colunas_originais[3]
                                tempo_horas_usuario_arvore = float(input(f"Tempo de serviço em horas (para '{nome_coluna_horas_arvore}', ex.: 2.5): "))

                                nome_coluna_km_arvore = nomes_colunas_originais[4]
                                km_carro_usuario_arvore = float(input(f"Quilometragem do carro (para '{nome_coluna_km_arvore}', ex.: 85000): "))

                                nome_coluna_ano_carro_arvore = nomes_colunas_originais[5]
                                ano_carro_usuario_arvore = int(input(f"Ano de fabricação (para '{nome_coluna_ano_carro_arvore}', ex.: 2018): "))

                            except IndexError:
                                print("\nErro de configuração: O número de colunas em 'nomes_colunas_originais' não é o esperado.")
                                print(f"Verifique 'nomes_colunas_originais': {nomes_colunas_originais}")
                                continue

                            valores_nova_entrada_arvore = [[servico_digitado_usuario, valor_pecas_usuario_arvore, valor_mao_obra_usuario_arvore, tempo_horas_usuario_arvore, km_carro_usuario_arvore, ano_carro_usuario_arvore]]
                            dataframe_nova_entrada_arvore = pd.DataFrame(valores_nova_entrada_arvore, columns=nomes_colunas_originais)
                            
                            nova_entrada_codificada_arvore = pd.get_dummies(dataframe_nova_entrada_arvore, columns=['Servico'], prefix='Servico', dtype=int)
                            nova_entrada_reindexada_arvore = nova_entrada_codificada_arvore.reindex(columns=X_treino.columns, fill_value=0)

                            predicao_final_arvore = modelo_arvore_decisao.predict(nova_entrada_reindexada_arvore)
                            print(f"\nPrevisão da Avaliação do Cliente (Árvore): {predicao_final_arvore[0]}")

                        except ValueError:
                            print("\nErro: Entrada inválida para valor numérico. Tente a classificação novamente.")
                        except Exception as erro_classificacao_arvore:
                             print(f"\nErro inesperado durante a nova classificação (Árvore): {erro_classificacao_arvore}")

                    elif opcao_submenu_arvore == 4:
                        break 
                    else:
                        print("Opção inválida.")
                except ValueError:
                    print("Entrada inválida para opção do submenu. Digite um número.")

        # --- Seção SVM ---
        elif opcao_menu_principal == 2:
            if X_treino_escalonado is None:
                escalonador_features = StandardScaler()
                X_treino_escalonado = escalonador_features.fit_transform(X_treino)
                X_teste_escalonado = escalonador_features.transform(X_teste)
                modelo_svm_oficina = svm.SVC(kernel='linear', C=1.0)
                modelo_svm_oficina.fit(X_treino_escalonado, y_treino)

            while True: # Loop do submenu SVM
                print("\nSeção SVM\nEscolha uma opção:")
                print("1 - Mostrar Desempenho\n2 - Fazer nova classificação\n3 - Voltar ao menu principal")
                try:
                    opcao_submenu_svm = int(input("Digite uma opção: "))

                    if opcao_submenu_svm == 1: # Mostrar Desempenho
                        predicoes_svm = modelo_svm_oficina.predict(X_teste_escalonado)
                        print(f"\nAcurácia (SVM): {(accuracy_score(y_teste, predicoes_svm) * 100):.2f}%")
                        print("-" * 30)
                        print("\nRelatório de Classificação (SVM):\n")
                        rotulos_classes_relatorio_svm = [str(classe) for classe in sorted(y_alvo.unique())]
                        print(classification_report(y_teste, predicoes_svm, target_names=rotulos_classes_relatorio_svm, zero_division=0))
                        print("-" * 30)

                    elif opcao_submenu_svm == 2: # Fazer Nova Classificação
                        print("\n--- Previsão para Nova Entrada (SVM) ---")
                        fluxo_classificacao_interrompido_svm = False
                        try:
                            servico_digitado_usuario_svm = ""
                            while True: # Loop para obter entrada de serviço válida
                                print(f"Serviços disponíveis: {', '.join(lista_servicos_disponiveis)}")
                                if not lista_servicos_disponiveis:
                                    print("Nenhum serviço disponível para classificação. Voltando ao menu anterior.")
                                    fluxo_classificacao_interrompido_svm = True
                                    break
                                
                                entrada_servico_usuario_temp_svm = input(f"Digite o tipo de Serviço (ou 'cancelar' para voltar): ").strip()

                                if entrada_servico_usuario_temp_svm.lower() == 'cancelar':
                                    print("Classificação cancelada.")
                                    fluxo_classificacao_interrompido_svm = True
                                    break
                                
                                nome_servico_correspondente_svm = None
                                for item_servico_lista_svm in lista_servicos_disponiveis:
                                    if entrada_servico_usuario_temp_svm.lower() == item_servico_lista_svm.lower():
                                        nome_servico_correspondente_svm = item_servico_lista_svm
                                        break
                                
                                if nome_servico_correspondente_svm is None:
                                    print(f"Erro: Serviço '{entrada_servico_usuario_temp_svm}' inválido ou não encontrado. Por favor, tente novamente ou digite 'cancelar'.")
                                else:
                                    servico_digitado_usuario_svm = nome_servico_correspondente_svm
                                    print(f"Serviço selecionado: {servico_digitado_usuario_svm}")
                                    break
                            
                            if fluxo_classificacao_interrompido_svm:
                                continue

                            try:
                                #input para usuario
                                nome_coluna_pecas_svm = nomes_colunas_originais[1]
                                valor_pecas_usuario_svm = float(input(f"Valor das peças (para '{nome_coluna_pecas_svm}', ex.: 150.0): "))

                                nome_coluna_mao_obra_svm = nomes_colunas_originais[2]
                                valor_mao_obra_usuario_svm = float(input(f"Valor da mão de obra (para '{nome_coluna_mao_obra_svm}', ex.: 70.0): "))
                                
                                nome_coluna_horas_svm = nomes_colunas_originais[3]
                                tempo_horas_usuario_svm = float(input(f"Tempo de serviço em horas (para '{nome_coluna_horas_svm}', ex.: 2.5): "))

                                nome_coluna_km_svm = nomes_colunas_originais[4]
                                km_carro_usuario_svm = float(input(f"Quilometragem do carro (para '{nome_coluna_km_svm}', ex.: 85000): "))

                                nome_coluna_ano_carro_svm = nomes_colunas_originais[5]
                                ano_carro_usuario_svm = int(input(f"Ano de fabricação (para '{nome_coluna_ano_carro_svm}', ex.: 2018): "))
                                
                            except IndexError:
                                print("\nErro de configuração: O número de colunas em 'nomes_colunas_originais' não é o esperado.")
                                print(f"Verifique 'nomes_colunas_originais': {nomes_colunas_originais}")
                                continue
                                
                            valores_nova_entrada_svm = [[servico_digitado_usuario_svm, valor_pecas_usuario_svm, valor_mao_obra_usuario_svm, tempo_horas_usuario_svm, km_carro_usuario_svm, ano_carro_usuario_svm]]
                            dataframe_nova_entrada_svm = pd.DataFrame(valores_nova_entrada_svm, columns=nomes_colunas_originais)
                            
                            nova_entrada_codificada_svm = pd.get_dummies(dataframe_nova_entrada_svm, columns=['Servico'], prefix='Servico', dtype=int)
                            nova_entrada_reindexada_svm = nova_entrada_codificada_svm.reindex(columns=X_treino.columns, fill_value=0)
                            nova_entrada_escalonada_svm = escalonador_features.transform(nova_entrada_reindexada_svm)

                            predicao_final_svm = modelo_svm_oficina.predict(nova_entrada_escalonada_svm)
                            print(f"\nPrevisão da Avaliação do Cliente (SVM): {predicao_final_svm[0]}")

                        except ValueError:
                            print("\nErro: Entrada inválida para valor numérico. Tente a classificação novamente.")
                        except Exception as erro_classificacao_svm:
                            print(f"\nErro inesperado durante a nova classificação SVM: {erro_classificacao_svm}")
                            
                    elif opcao_submenu_svm == 3:
                        break
                    else:
                        print("Opção inválida.")
                except ValueError:
                    print("Entrada inválida para opção do submenu. Digite um número.")

        elif opcao_menu_principal == 3:
            print("Programa encerrado.")
            break
        else:
            print("Opção inválida.")
    except ValueError:
        print("Entrada inválida para opção do menu principal. Digite um número.")