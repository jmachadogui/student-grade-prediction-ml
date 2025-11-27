import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from visualization import Visualizer

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    encoding_type = 'le'  # default
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['le', 'ohe']:
            encoding_type = arg
        else:
            print("Uso: python3 main.py [le|ohe]")
            print("  le  - Label Encoding (padrão)")
            print("  ohe - One-Hot Encoding")
            sys.exit(1)
    
    print("="*60)
    print("ANÁLISE DO STUDENT PERFORMANCE DATASET")
    print(f"Encoding selecionado: {'Label Encoding' if encoding_type == 'le' else 'One-Hot Encoding'}")
    print("="*60)
    
    # 1. CARREGAR E EXPLORAR DADOS
    loader = DataLoader('./dataset/student-mat.csv')
    df = loader.load_data()
    loader.explore_dataset(df)
    
    # 2. IDENTIFICAR OUTLIERS (antes de criar a classe)
    visualizer = Visualizer()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers_info = visualizer.plot_outliers(df, numeric_columns, method='iqr')
    
    # 3. CRIAR ATRIBUTO CLASSE
    df = loader.create_target_class(df)
    
    # 4. PREPARAR DADOS com o encoding escolhido
    preprocessor = DataPreprocessor(encoding_type=encoding_type)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # 5. TREINAR E AVALIAR MODELOS
    trainer = ModelTrainer()
    resultados = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 6. VISUALIZAÇÕES
    visualizer.plot_class_distribution(df)
    visualizer.plot_comparison(resultados)
    
    # 7. GRÁFICO DE PREDIÇÕES VS REAL
    predictions = trainer.get_predictions()
    visualizer.plot_real_vs_pred(predictions, y_test)
    
    # 8. GRÁFICO DE IMPORTÂNCIA DAS FEATURES
    feature_importance = trainer.get_feature_importance()
    visualizer.plot_feature_importance(feature_importance, top_n=15)
    
    # 9. SALVAR RESULTADOS
    df_resultados = pd.DataFrame(resultados)
    output_filename = f'resultados_comparacao_{encoding_type}.csv'
    df_resultados.to_csv(output_filename, index=False)
    print(f"\n✓ Resultados salvos em: {output_filename}")
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA!")
    print("="*60)
    print("\nArquivos gerados:")
    print("  • distribuicao_classes.png")
    print("  • comparacao_algoritmos.png")
    print("  • predicoes_vs_real.png")
    print("  • importancia_features.png")
    print("  • analise_outliers.png")
    print(f"  • {output_filename}")

if __name__ == "__main__":
    main()
