import pandas as pd
from algorithms.decision_tree import DecisionTreeModel
from algorithms.random_forest import RandomForestModel
from algorithms.naive_bayes import NaiveBayesModel

class ModelTrainer:
    def __init__(self):
        self.models = []
        self.results = []

    def initialize_models(self):
        """Inicializa os modelos de machine learning"""
        self.models = [
            DecisionTreeModel(random_state=42),
            RandomForestModel(n_estimators=100, random_state=42),
            NaiveBayesModel()
        ]

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Treina e avalia todos os modelos"""
        print("\n" + "="*60)
        print("TREINAMENTO E AVALIAÇÃO DE ALGORITMOS")
        print("="*60)

        self.initialize_models()
        self.results = []

        for model in self.models:
            # Treinar
            model.train(X_train, y_train)

            # Avaliar
            resultado = model.evaluate(X_test, y_test)
            self.results.append(resultado)

        # Exibir comparação
        self._print_comparison()

        return self.results

    def _print_comparison(self):
        """Imprime a comparação entre os modelos"""
        print("\n" + "="*60)
        print("COMPARAÇÃO DE RESULTADOS")
        print("="*60)

        df_resultados = pd.DataFrame(self.results)
        print("\nTabela Comparativa:")
        print(df_resultados.to_string(index=False))

        # Identificar melhor e pior
        melhor_idx = df_resultados['F1-Score'].idxmax()
        pior_idx = df_resultados['F1-Score'].idxmin()

        print(f"\n{'='*60}")
        print("ANÁLISE FINAL")
        print(f"{'='*60}")

        print(f"\n🏆 MELHOR ALGORITMO: {df_resultados.loc[melhor_idx, 'Algoritmo']}")
        print(f"   Acurácia:  {df_resultados.loc[melhor_idx, 'Acurácia']:.4f}")
        print(f"   Precisão:  {df_resultados.loc[melhor_idx, 'Precisão']:.4f}")
        print(f"   Recall:    {df_resultados.loc[melhor_idx, 'Recall']:.4f}")
        print(f"   F1-Score:  {df_resultados.loc[melhor_idx, 'F1-Score']:.4f}")
        print(f"   Tempo:     {df_resultados.loc[melhor_idx, 'Tempo (s)']:.4f}s")

        print(f"\n❌ PIOR ALGORITMO: {df_resultados.loc[pior_idx, 'Algoritmo']}")
        print(f"   Acurácia:  {df_resultados.loc[pior_idx, 'Acurácia']:.4f}")
        print(f"   Precisão:  {df_resultados.loc[pior_idx, 'Precisão']:.4f}")
        print(f"   Recall:    {df_resultados.loc[pior_idx, 'Recall']:.4f}")
        print(f"   F1-Score:  {df_resultados.loc[pior_idx, 'F1-Score']:.4f}")
        print(f"   Tempo:     {df_resultados.loc[pior_idx, 'Tempo (s)']:.4f}s")

    def get_best_model(self):
        """Retorna o melhor modelo baseado no F1-Score"""
        if not self.results:
            return None

        df_resultados = pd.DataFrame(self.results)
        melhor_idx = df_resultados['F1-Score'].idxmax()
        return self.models[melhor_idx]
