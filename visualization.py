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
    
    def plot_real_vs_pred(self, models_predictions, y_test):
        """Gera uma imagem individual para cada modelo com gráficos grandes."""
        target_names = ['Reprovado', 'Aprovado']
        samples = np.arange(len(y_test))

        for model_name, y_pred in models_predictions.items():
            
            # Cria figura grande (2 linhas, 1 coluna)
            fig, axes = plt.subplots(2, 1, figsize=(12, 14))
            ax1, ax2 = axes

            # --------------------------------------------------
            # 1. GRÁFICO REAL VS PREVISTO (GRANDE)
            # --------------------------------------------------
            ax1.plot(samples, y_test, 'o-', 
                     label='Real', alpha=0.8, linewidth=3, markersize=7)

            ax1.plot(samples, y_pred, 's-', 
                     label='Previsto', color='#2ecc71', alpha=0.8, linewidth=3, markersize=6)

            ax1.fill_between(
                samples, y_test, y_pred,
                where=(y_test != y_pred),
                color='red', alpha=0.2, label='Erro'
            )

            ax1.set_title(f"{model_name} — Real vs Previsto", fontsize=20, weight='bold')
            ax1.set_xlabel("Amostras de Teste", fontsize=14)
            ax1.set_ylabel("Classe", fontsize=14)
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(target_names, fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', labelsize=12)

            # --------------------------------------------------
            # 2. MATRIZ DE CONFUSÃO (GRANDE)
            # --------------------------------------------------
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                cbar=False, annot_kws={'size': 18}, ax=ax2
            )

            ax2.set_title("Matriz de Confusão", fontsize=20, weight='bold')
            ax2.set_xlabel("Previsto", fontsize=14)
            ax2.set_ylabel("Real", fontsize=14)
            ax2.tick_params(axis='both', labelsize=12)

            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            errors = np.sum(y_test != y_pred)

            ax2.text(
                0.5, -0.15,
                f"Acurácia: {accuracy:.2%}   |   Erros: {errors}/{len(y_test)}",
                ha='center', va='top', transform=ax2.transAxes,
                fontsize=14, weight='bold'
            )

            # --------------------------------------------------
            # SALVAR FIGURA
            # --------------------------------------------------
            filename = f"resultados/{model_name.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ Imagem salva: {filename}")

    def plot_predictions_vs_real(self, models_predictions, y_test):
        """Plota predições vs valores reais para cada modelo"""
        n_models = len(models_predictions)

        # Criar figura com 2 linhas: linha superior com gráficos de linha, inferior com matrizes
        fig = plt.figure(figsize=(10*n_models, 12))
        gs = fig.add_gridspec(2, n_models, hspace=0.35, wspace=0.3)

        target_names = ['Reprovado', 'Aprovado']

        for idx, (model_name, y_pred) in enumerate(models_predictions.items()):
            # GRÁFICO SUPERIOR: Linha Real vs Previsto
            ax1 = fig.add_subplot(gs[0, idx])

            # Criar índice de amostras
            samples = np.arange(len(y_test))

            # Plotar linha real
            ax1.plot(samples, y_test, 'o-', label='Real', 
                    color='#2ecc71', linewidth=3, markersize=6, alpha=0.8)

            # Plotar linha prevista
            ax1.plot(samples, y_pred, 's-', label='Previsto', 
                    color='#e74c3c', linewidth=3, markersize=5, alpha=0.8)

            ax1.set_title(f'{model_name}\nReal vs Previsto', fontsize=16, weight='bold')
            ax1.set_xlabel('Amostras de Teste', fontsize=14)
            ax1.set_ylabel('Classe', fontsize=14)
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(target_names, fontsize=12)
            ax1.tick_params(axis='both', labelsize=12)
            ax1.legend(loc='upper right', fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Adicionar linha de fundo para destacar diferenças
            ax1.fill_between(samples, y_test, y_pred, 
                           where=(y_test != y_pred), 
                           color='red', alpha=0.2, label='Erro')

            # GRÁFICO INFERIOR: Matriz de Confusão
            ax2 = fig.add_subplot(gs[1, idx])

            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=target_names, 
                       yticklabels=target_names,
                       ax=ax2, cbar=False, annot_kws={'size': 16})
            ax2.set_title(f'Matriz de Confusão', fontsize=15, weight='bold')
            ax2.set_ylabel('Real', fontsize=14)
            ax2.set_xlabel('Previsto', fontsize=14)
            ax2.tick_params(axis='both', labelsize=12)

            # Adicionar métricas
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            errors = np.sum(y_test != y_pred)
            ax2.text(0.5, -0.15, 
                    f'Acurácia: {accuracy:.2%} | Erros: {errors}/{len(y_test)}', 
                    ha='center', va='top', transform=ax2.transAxes,
                    fontsize=13, weight='bold')

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
