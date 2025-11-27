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
```
student-grade-prediction-ml/
│
├── algorithms/                      # Implementações dos algoritmos de ML
│   ├── __init__.py                 # Inicializador do módulo
│   ├── base_model.py               # Classe base abstrata para todos os modelos
│   ├── decision_tree.py            # Implementação do Decision Tree
│   ├── naive_bayes.py              # Implementação do Naive Bayes
│   └── random_forest.py            # Implementação do Random Forest
│
├── dataset/                         # Datasets utilizados
│   └── (arquivos .csv)             # Dataset de matemática
│
├── resultados-label-encoding/       # Amostra de resultados do script usando Label encoding
│   └── (logs, gráficos e CSVs gerados)   # Outputs do encoding LE
│
├── resultados-one-hot-encoding/     # Amostra de resultados do script usando One-Hot Encoding
│   └── (logs, gráficos e CSVs gerados)   # Outputs do encoding OHE
│
├── resultados/                      # Pasta onde o script armazena as imagens e CSVs gerados
│   └── (gráficos e CSVs gerados)               # Exemplo de output
│
├── data_loader.py                   # Carregamento e exploração inicial dos dados
├── data_preprocessor.py             # Pré-processamento e preparação dos dados
├── encoding_columns.py              # Configuração de colunas para encoding
├── main.py                          # Arquivo principal de execução
├── model_trainer.py                 # Orquestração do treinamento e avaliação
├── visualization.py                 # Geração de gráficos e análises visuais
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Documentação do projeto
```


As pastas de `resultados-one-hot-encoding/`  e `resultados-label-encoding/` não são populadas pelo script automaticamente. Ela foram geradas rodando o script para cada tipo de encoding uma vez e copiando os dados salvos na pasta `resultados/`. Todos os dados da pasta `resultados` são sobrescritos quando o script é executado.

