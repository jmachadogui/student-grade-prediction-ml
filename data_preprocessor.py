from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

class DataPreprocessor:
    def __init__(self, test_size=0.3, random_state=42, encoding_type='le'):
        """
        Args:
            test_size: Proporção do conjunto de teste
            random_state: Seed para reprodutibilidade
            encoding_type: Tipo de encoding ('le' para Label Encoding ou 'ohe' para One-Hot Encoding)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.encoding_type = encoding_type.lower()
        self.label_encoder = LabelEncoder()
        
        if self.encoding_type not in ['le', 'ohe']:
            raise ValueError("encoding_type deve ser 'le' ou 'ohe'")
    
    def prepare_data(self, df):
        """Prepara os dados para modelagem"""
        print("\n" + "="*60)
        print("PREPARAÇÃO DOS DADOS")
        print("="*60)
        print(f"Tipo de encoding: {'Label Encoding' if self.encoding_type == 'le' else 'One-Hot Encoding'}")
        
        features_to_drop = ['G3', 'Performance']
        X = df.drop(columns=features_to_drop)
        y = df['Performance']
        
        # Codificar variáveis categóricas
        categorical_columns = X.select_dtypes(include='object').columns.tolist()
        
        if self.encoding_type == 'le':
            X_encoded = self._label_encoding(X, categorical_columns)
        else:  # ohe
            X_encoded = self._onehot_encoding(X, categorical_columns)
        
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
    
    def _label_encoding(self, X, categorical_columns):
        """Aplica Label Encoding nas colunas categóricas"""
        X_encoded = X.copy()
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
        return X_encoded
    
    def _onehot_encoding(self, X, categorical_columns):
        print("ONE HOT ENCODING SEND USADO")
        """Aplica One-Hot Encoding nas colunas categóricas"""
        X_encoded = X.copy()
        
        # Aplicar One-Hot Encoding
        X_encoded = pd.get_dummies(X_encoded, columns=categorical_columns, drop_first=True)
        
        return X_encoded
