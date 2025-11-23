import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self):
        self.fig_size = (12, 6)

    def plot_class_distribution(self, df):
        """Plota a distribuição das classes"""
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        df['Performance'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
        plt.title('Distribuição das Classes')
        plt.xlabel('Performance')
        plt.ylabel('Quantidade')
        plt.xticks(rotation=0)

        plt.subplot(1, 2, 2)
        df['G3'].hist(bins=20, color='skyblue', edgecolor='black')
        plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Limiar (nota 10)')
        plt.title('Distribuição das Notas Finais (G3)')
        plt.xlabel('Nota')
        plt.ylabel('Frequência')
        plt.legend()

        plt.tight_layout()
        plt.savefig('resultados/distribuicao_classes.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico salvo: distribuicao_classes.png")
        plt.close()

    def plot_comparison(self, resultados):
        """Plota a comparação entre os modelos"""
        df_resultados = pd.DataFrame(resultados)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Gráfico 1: Acurácia
        axes[0, 0].bar(df_resultados['Algoritmo'], df_resultados['Acurácia'], color='steelblue')
        axes[0, 0].set_title('Acurácia por Algoritmo')
        axes[0, 0].set_ylabel('Acurácia')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(df_resultados['Acurácia']):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')

        # Gráfico 2: Precisão
        axes[0, 1].bar(df_resultados['Algoritmo'], df_resultados['Precisão'], color='orange')
        axes[0, 1].set_title('Precisão por Algoritmo')
        axes[0, 1].set_ylabel('Precisão')
        axes[0, 1].set_ylim([0, 1])
        for i, v in enumerate(df_resultados['Precisão']):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')

        # Gráfico 3: Recall e F1-Score
        x = np.arange(len(df_resultados))
        width = 0.35
        axes[1, 0].bar(x - width/2, df_resultados['Recall'], width, label='Recall', color='green')
        axes[1, 0].bar(x + width/2, df_resultados['F1-Score'], width, label='F1-Score', color='purple')
        axes[1, 0].set_title('Recall e F1-Score por Algoritmo')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df_resultados['Algoritmo'])
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])

        # Gráfico 4: Tempo de treinamento
        axes[1, 1].bar(df_resultados['Algoritmo'], df_resultados['Tempo (s)'], color='crimson')
        axes[1, 1].set_title('Tempo de Treinamento por Algoritmo')
        axes[1, 1].set_ylabel('Tempo (segundos)')
        for i, v in enumerate(df_resultados['Tempo (s)']):
            axes[1, 1].text(i, v + 0.001, f'{v:.4f}s', ha='center')

        plt.tight_layout()
        plt.savefig('resultados/comparacao_algoritmos.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico comparativo salvo: comparacao_algoritmos.png")
        plt.close()

    def plot_feature_importance(self, importance_dict, top_n=10):
        """Plota a importância das features (para modelos baseados em árvore)"""
        if importance_dict is None:
            print("Este modelo não possui importância de features.")
            return

        # Ordenar por importância
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]

        features, importances = zip(*top_features)

        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color='skyblue')
        plt.xlabel('Importância')
        plt.title(f'Top {top_n} Features Mais Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico de importância das features salvo: feature_importance.png")
        plt.close()
