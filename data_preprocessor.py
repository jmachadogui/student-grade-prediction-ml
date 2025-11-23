from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def prepare_data(self, df):
        """Prepara os dados para modelagem"""
        print("\n" + "="*60)
        print("PREPARAÇÃO DOS DADOS")
        print("="*60)

        # Selecionar features (remover G1, G2, G3 e Performance)
        features_to_drop = ['G3', 'Performance']
        X = df.drop(columns=features_to_drop)
        y = df['Performance']

        # Codificar variáveis categóricas
        X_encoded = X.copy()
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])

        # Codificar target
        y_encoded = self.label_encoder.fit_transform(y)

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y_encoded
        )

        print(f"\n✓ Dados preparados:")
        print(f"   Conjunto de treino: {X_train.shape[0]} exemplos")
        print(f"   Conjunto de teste: {X_test.shape[0]} exemplos")
        print(f"   Features utilizadas: {X_encoded.shape[1]}")

        return X_train, X_test, y_train, y_test
