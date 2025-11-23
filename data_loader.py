import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Carrega o dataset"""
        df = pd.read_csv(self.file_path, sep=';')
        print(f"\n✓ Dataset carregado com sucesso!")
        print(f"Dimensões: {df.shape[0]} instâncias x {df.shape[1]} atributos")
        return df

    def explore_dataset(self, df):
        """Explora e caracteriza o dataset"""
        print("\n" + "="*60)
        print("CARACTERIZAÇÃO DO DATASET")
        print("="*60)

        # a. Quantidade de atributos
        print(f"\na) Quantidade de atributos: {df.shape[1]}")

        # b. Tipo de cada atributo
        print(f"\nb) Tipos de atributos:")
        for col, tipo in df.dtypes.items():
            print(f"   {col}: {tipo}")

        # c. Intervalo ou conjunto de valores
        print(f"\nc) Intervalo/Conjunto de valores:")
        print("\nAtributos Numéricos:")
        print(df.describe())

        print("\nAtributos Categóricos (valores únicos):")
        for col in df.select_dtypes(include='object').columns:
            valores_unicos = df[col].unique()
            print(f"   {col}: {valores_unicos}")

        # d. Valores faltantes
        print(f"\nd) Valores faltantes por atributo:")
        valores_faltantes = df.isnull().sum()
        total_faltantes = valores_faltantes.sum()
        if total_faltantes > 0:
            print(valores_faltantes[valores_faltantes > 0])
        else:
            print("   ✓ Não há valores faltantes no dataset!")

        # e. Atributo mais desbalanceado
        print(f"\ne) Análise de desbalanceamento:")
        desbalanceamentos = {}
        for col in df.columns:
            distribuicao = df[col].value_counts()
            if len(distribuicao) > 1:
                desbalanceamento = (distribuicao.max() - distribuicao.min()) / len(df)
                desbalanceamentos[col] = desbalanceamento

        atributo_mais_desbalanceado = max(desbalanceamentos, key=desbalanceamentos.get)
        print(f"   Atributo mais desbalanceado: {atributo_mais_desbalanceado}")
        print(f"   Distribuição:")
        print(df[atributo_mais_desbalanceado].value_counts())

        # f. Quantidade de instâncias
        print(f"\nf) Quantidade de instâncias/exemplos: {df.shape[0]}")

    def create_target_class(self, df):
        """Cria o atributo classe/target"""
        print("\n" + "="*60)
        print("CRIAÇÃO DO ATRIBUTO CLASSE/TARGET")
        print("="*60)

        df['Performance'] = df['G3'].apply(lambda x: 'Aprovado' if x >= 10 else 'Reprovado')

        print(f"\nAtributo classe criado: Performance")
        print(f"Baseado em: G3 (nota final)")
        print(f"Classes:")
        print(df['Performance'].value_counts())
        print(f"\nDistribuição percentual:")
        print(df['Performance'].value_counts(normalize=True) * 100)

        return df
