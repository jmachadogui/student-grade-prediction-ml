### Setup
Para poder rodar os scripts, pode ser necessário instalar as dependências.
Na raiz do projeto, rode:
```pip install -r requirements.txt```

isso deve instalar as dependências listadas no arquivo:
-pandas>=1.3.0
-numpy>=1.21.0
-matplotlib>=3.4.0
-seaborn>=0.11.0
-scikit-learn>=0.24.0

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

student-grade-prediction-ml/
├── algorithms/
│   ├── __init__.py
│   ├── base_model.py
│   ├── decision_tree.py
│   ├── naive_bayes.py
│   └── random_forest.py
├── dataset/
│   └── student-mat.csv
├── resultados-label-encoding/
│   └── (outputs do Label Encoding)
├── resultados-one-hot-encoding/
│   └── (outputs do One-Hot Encoding)
├── resultados/
│   └── exemplo.txt
├── data_loader.py
├── data_preprocessor.py
├── encoding_columns.py
├── main.py
├── model_trainer.py
├── visualization.py
├── requirements.txt
└── README.md

*As pastas resultados-label-encoding e resultados-one-hot-encoding não são atualizadas pelo script. A única pasta que é atualizada é a resultados/ *

