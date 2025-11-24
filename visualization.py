import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
    
    def plot_predictions_vs_real(self, models_predictions, y_test):
        """Plota predições vs valores reais para cada modelo"""
        n_models = len(models_predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        target_names = ['Reprovado', 'Aprovado']
        
        for idx, (model_name, y_pred) in enumerate(models_predictions.items()):
            # Criar matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            
            # Plotar matriz de confusão
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=target_names, 
                       yticklabels=target_names,
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{model_name}\nMatriz de Confusão')
            axes[idx].set_ylabel('Real')
            axes[idx].set_xlabel('Predito')
            
            # Adicionar acurácia no título
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            axes[idx].text(0.5, -0.15, f'Acurácia: {accuracy:.2%}', 
                          ha='center', va='top', transform=axes[idx].transAxes,
                          fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.savefig('resultados/predicoes_vs_real.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico salvo: predicoes_vs_real.png")
        plt.close()
    
    def plot_feature_importance(self, models_importance, top_n=15):
        """Plota a importância das features para modelos baseados em árvore"""
        # Filtrar apenas modelos com importância
        valid_models = {k: v for k, v in models_importance.items() if v is not None}
        
        if not valid_models:
            print("\n⚠️  Nenhum modelo com importância de features disponível.")
            return
        
        n_models = len(valid_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance_dict) in enumerate(valid_models.items()):
            # Ordenar por importância
            sorted_features = sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            
            features, importances = zip(*top_features)
            
            # Plotar
            y_pos = np.arange(len(features))
            axes[idx].barh(y_pos, importances, color='skyblue')
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(features)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Importância')
            axes[idx].set_title(f'{model_name}\nTop {top_n} Features')
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('resultados/importancia_features.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico salvo: importancia_features.png")
        plt.close()
    
    def plot_outliers(self, df, numeric_columns, method='iqr'):
        """Identifica e plota outliers usando IQR ou Z-score"""
        print("\n" + "="*60)
        print("ANÁLISE DE OUTLIERS")
        print("="*60)
        
        n_cols = len(numeric_columns)
        n_rows = (n_cols + 2) // 3  # 3 colunas por linha
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        outliers_summary = {}
        
        for idx, col in enumerate(numeric_columns):
            data = df[col].dropna()
            
            if method == 'iqr':
                # Método IQR (Interquartile Range)
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
            else:
                # Método Z-score
                mean = data.mean()
                std = data.std()
                z_scores = np.abs((data - mean) / std)
                outliers = data[z_scores > 3]
            
            # Armazenar informações
            outliers_summary[col] = {
                'quantidade': len(outliers),
                'percentual': (len(outliers) / len(data)) * 100,
                'valores': outliers.tolist()
            }
            
            # Plotar boxplot
            axes[idx].boxplot(data, vert=True)
            axes[idx].set_title(f'{col}\n{len(outliers)} outliers ({outliers_summary[col]["percentual"]:.1f}%)')
            axes[idx].set_ylabel('Valor')
            axes[idx].grid(axis='y', alpha=0.3)
        
        # Remover subplots vazios
        for idx in range(len(numeric_columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('resultados/analise_outliers.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráfico salvo: analise_outliers.png")
        plt.close()
        
        # Imprimir resumo
        print(f"\nResumo de Outliers (método: {method.upper()}):")
        print("-" * 60)
        for col, info in outliers_summary.items():
            print(f"\n{col}:")
            print(f"  Quantidade: {info['quantidade']}")
            print(f"  Percentual: {info['percentual']:.2f}%")
            if info['quantidade'] > 0 and info['quantidade'] <= 10:
                print(f"  Valores: {info['valores']}")
        
        return outliers_summary
