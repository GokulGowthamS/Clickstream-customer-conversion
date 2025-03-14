import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv("F:\\Guvi Projects\\Clickstream - Customer Conversion\\Preprocessed_Data\\cleaned_data.csv")

drop_columns = ['year', 'session_id', 'page2_clothing_model']
data.drop(columns=drop_columns, errors='ignore', inplace=True)

corr = data.corr()

pair_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
corr65 = pair_corr[abs(pair_corr[0]) > 0.65]
corr65.columns = ['Primary', 'Secondary', 'Score']

groups = corr65.groupby(['Primary']).agg({'Secondary': 'count'}).sort_values('Secondary', ascending=False).index
columns_to_drop = list(groups)

data.drop(columns=columns_to_drop, axis=1, errors='ignore', inplace=True)

numeric_features = ['browsing_depth', 'avg_price', 'unique_products', 'weekend']
categorical_features = ['page1_main_category', 'colour', 'location', 'model_photography', 'page', 'country']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X = data.drop(columns=['high_price_preference'], errors='ignore')  # Exclude target variable
preprocessor.fit(X)

processed_csv_path = "F:\\Guvi Projects\\Clickstream - Customer Conversion\\Preprocessed_Data"

os.makedirs(processed_csv_path, exist_ok=True)

processed_csv_file = os.path.join(processed_csv_path, "processed_data.csv")

data.to_csv(processed_csv_file, index=False)

print(f"Processed data saved successfully at '{processed_csv_file}'")

preprocessed_data_path = "F:\\Guvi Projects\\Clickstream - Customer Conversion\\Pickles"

os.makedirs(preprocessed_data_path, exist_ok=True)

data_file = os.path.join(preprocessed_data_path, "preprocessed_data.pkl")

with open(data_file, "wb") as file:
    pickle.dump(preprocessor, file)

print(f"Preprocessing model saved successfully at '{data_file}'")


