### Setup
Para poder rodar os scripts, pode ser necessário instalar as dependências.
Na raiz do projeto, rode:
```pip install -r requirements.txt```

isso deve instalar as dependências listadas no arquivo:
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0

### Como executar?
O script principal é o main.py. Por padrão o script irá rodar utilizando Label Encoding no dataset, mas é possível passar um parâmetro para usar OHE. Formas de utilização:

```python3 main.py``` 
ou 
```python3 main.py le```
ou 
```python3 main.py ohe```

le = Label Encoding

ohe = One Hot Encoding

Após a execução do script, serão gerados alguns arquivos dentro da pasta resultados. Arquivos gerados:
- resultados_comparacao_<encoding>.csv
- Real Vs Presvisto e matriz de confusão dos algoritmos:
    - Decision_tree.png
    - Random_forest.png
    - Naive Bayes.png
- analise_outliers.png
- comparacao_algortimos.png
- distribuicao_classes.png
- importancia_features.png

### Estrutura e Organização

#### algorithms/
Contém as implementações de todos os algoritmos de Machine Learning utilizados no projeto.
__init__.py

Responsabilidade: Inicializador do pacote algorithms
Funcionalidades:

Torna o diretório um módulo Python
Facilita imports dos modelos


#### base_model.py

Responsabilidade: Define a classe abstrata BaseModel que serve como template para todos os algoritmos
Funcionalidades:

Interface padronizada com métodos train() e predict()
Cálculo automático de métricas (acurácia, precisão, recall, F1-score)
Geração de matrizes de confusão
Estrutura comum para todos os modelos



Modelos Implementados

- decision_tree.py: Implementação de Árvore de Decisão
- naive_bayes.py: Implementação de Naive Bayes (classificador probabilístico)
- random_forest.py: Implementação de Random Forest (ensemble de árvores)

Cada modelo contém:

Implementação específica do algoritmo usando scikit-learn
Configuração de hiperparâmetros otimizados
Métodos de treinamento e predição
Extração de importância de features (quando aplicável)

## Arquivos na Raiz
#### data_loader.py

Responsabilidade: Carregamento e exploração inicial do dataset
Funcionalidades:

Leitura do arquivo CSV
Análise exploratória dos dados (EDA)
Estatísticas descritivas
Criação da classe target (Performance) baseada em G3
Verificação de dados faltantes


#### data_preprocessor.py

Responsabilidade: Pré-processamento e transformação dos dados
Funcionalidades:

Suporte a três tipos de encoding:

Label Encoding (LE): Converte categorias em números ordinais
One-Hot Encoding (OHE): Cria variáveis dummy para cada categoria

Divisão em conjuntos de treino e teste
Estratificação da variável target
Codificação da variável target


#### model_trainer.py

Responsabilidade: Orquestração do processo de treinamento
Funcionalidades:

Instanciação de todos os modelos
Execução do treinamento para cada algoritmo
Coleta de métricas de desempenho
Comparação entre modelos
Armazenamento das predições
Extração de importância das features



#### visualization.py

Responsabilidade: Criação de gráficos e análises visuais
Funcionalidades:

Gráfico de distribuição de classes
Comparação de desempenho entre algoritmos
Análise de outliers (método IQR)
Gráfico de predições vs valores reais
Visualização da importância das features
Matrizes de confusão



#### main.py

Responsabilidade: Arquivo principal que orquestra todo o pipeline
Funcionalidades:

Processamento de argumentos da linha de comando
Execução sequencial de todas as etapas:

Carregamento dos dados
Análise exploratória
Identificação de outliers
Criação da classe target
Pré-processamento
Treinamento dos modelos
Geração de visualizações
Exportação dos resultados





## Diretórios de Resultados
#### resultados/

Conteúdo: Resultados gerados pelo script são armazenados nesta pasta apenas. Sempre que o script é executado, os dados são sobrescritos
Arquivos:
Gráficos PNG (distribuição, comparações, outliers)
CSV com métricas dos modelos


#### resultados-label-encoding/

Conteúdo: Armazena exemplos de resultados gerados usando Label Encoding
Arquivos:

Gráficos PNG (distribuição, comparações, outliers)
CSV com métricas dos modelos



#### resultados-one-hot-encoding/

Conteúdo: Armazena exemplos de resultados gerados usando One-Hot Encoding
Arquivos:

Gráficos PNG (distribuição, comparações, outliers)
CSV com métricas dos modelos





*As pastas resultados-label-encoding e resultados-one-hot-encoding não são atualizadas pelo script. A única pasta que é atualizada é a resultados/ *

