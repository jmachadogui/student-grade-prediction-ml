import time
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, classification_report)

class BaseModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.training_time = 0
        self.results = {}

    def train(self, X_train, y_train):
        """Treina o modelo e mede o tempo"""
        print(f"\n{'='*60}")
        print(f"Treinando: {self.name}")
        print(f"{'='*60}")

        inicio = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - inicio

        print(f"✓ Treinamento concluído em {self.training_time:.4f}s")

    def evaluate(self, X_test, y_test):
        """Avalia o modelo e retorna métricas"""
        y_pred = self.model.predict(X_test)

        acuracia = accuracy_score(y_test, y_pred)
        precisao = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        self.results = {
            'Algoritmo': self.name,
            'Acurácia': acuracia,
            'Precisão': precisao,
            'Recall': recall,
            'F1-Score': f1,
            'Tempo (s)': self.training_time
        }

        self._print_results(y_test, y_pred)

        return self.results

    def _print_results(self, y_test, y_pred):
        """Imprime os resultados da avaliação"""
        print(f"\nResultados:")
        print(f"  Acurácia:  {self.results['Acurácia']:.4f} ({self.results['Acurácia']*100:.2f}%)")
        print(f"  Precisão:  {self.results['Precisão']:.4f}")
        print(f"  Recall:    {self.results['Recall']:.4f}")
        print(f"  F1-Score:  {self.results['F1-Score']:.4f}")
        print(f"  Tempo:     {self.results['Tempo (s)']:.4f} segundos")

        print(f"\nRelatório por classe:")
        target_names = ['Reprovado', 'Aprovado']
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    def predict(self, X):
        """Faz predições"""
        return self.model.predict(X)
