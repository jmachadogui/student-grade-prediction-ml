
from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(df: pd.DataFrame, save_csv: Optional[str] = None):
    le = LabelEncoder()
    mappings = []

    # Process only object (string/categorical) columns
    for col in df.select_dtypes(include='object').columns:
        # Fit encoder and transform the column
        df[col] = le.fit_transform(df[col])

        # Capture the mapping for this column
        for encoded, original in enumerate(le.classes_):
            mappings.append({
                "column": col,
                "original_value": original,
                "encoded_value": encoded
            })

    # Build the mapping DataFrame
    mapping_df = pd.DataFrame(mappings)

    # Save CSV if requested
    if save_csv:
        mapping_df.to_csv(save_csv, index=False)

    return df, mapping_df

df = pd.read_csv('./dataset/student-mat.csv', sep=";")
encode_categorical_columns(df, save_csv="encoding_mapping_2.csv")
