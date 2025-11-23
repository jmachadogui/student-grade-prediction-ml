import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

# Importar módulos personalizados
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from visualization import Visualizer

warnings.filterwarnings('ignore')

# Configuração de visualização
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def main():
    print("="*60)
    print("ANÁLISE DO STUDENT PERFORMANCE DATASET")
    print("="*60)

    # 1. CARREGAR E EXPLORAR DADOS
    loader = DataLoader('./dataset/student-mat.csv')
    df = loader.load_data()
    loader.explore_dataset(df)

    # 2. CRIAR ATRIBUTO CLASSE
    df = loader.create_target_class(df)

    # 3. PREPARAR DADOS
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

    # 4. TREINAR E AVALIAR MODELOS
    trainer = ModelTrainer()
    resultados = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

    # 5. VISUALIZAR RESULTADOS
    visualizer = Visualizer()
    visualizer.plot_class_distribution(df)
    visualizer.plot_comparison(resultados)

    # 6. SALVAR RESULTADOS
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv('resultados/resultados_comparacao.csv', index=False)
    print("\n✓ Resultados salvos em: resultados_comparacao.csv")

    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA!")
    print("="*60)

if __name__ == "__main__":
    main()
