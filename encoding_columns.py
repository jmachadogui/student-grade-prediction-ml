
from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(df: pd.DataFrame, save_csv: Optional[str] = None):
    le = LabelEncoder()
    mappings = []

    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

        for encoded, original in enumerate(le.classes_):
            mappings.append({
                "column": col,
                "original_value": original,
                "encoded_value": encoded
            })

    mapping_df = pd.DataFrame(mappings)

    if save_csv:
        mapping_df.to_csv(save_csv, index=False)

    return df, mapping_df

df = pd.read_csv('./dataset/student-mat.csv', sep=";")
encode_categorical_columns(df, save_csv="encoding_mapping_2.csv")
