import pandas as pd
import numpy as np
import io
from sklearn.linear_model import Ridge


df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')

columns = df_train.columns
df_train = df_train.drop_duplicates(subset=[elem for elem in columns if elem != "selling_price"])
df_train = df_train.reset_index(drop=True)

df_train["max_power"] = df_train["max_power"].replace(" bhp", np.nan)
for column in ["mileage", "engine", "max_power"]:
    df_train[column] = df_train[column].apply(lambda x: float(x.split(" ")[0]) if type(x) is str else x)
df_train.drop(columns=["torque"], inplace=True)

buffer = io.StringIO()
df_train.info(buf=buffer)
info = buffer.getvalue()
numeric_columns = [column[5:18].strip() for column in info.split("\n")[5:-3] if column[-7:].strip() in ("int64", "float64")]
for column in numeric_columns:
    median = df_train[column].median()
    df_train[column] = df_train[column].fillna(median)

df_train["engine"] = df_train["engine"].astype(int)
df_train["seats"] = df_train["seats"].astype(int)

y_train = df_train[["selling_price"]]
X_train = df_train[[value for value in df_train.columns.values if value not in ("name", "selling_price")]]

category_columns = ["fuel", "seller_type", "transmission", "owner", "seats"]
X_train = pd.get_dummies(X_train, columns=category_columns, drop_first=True)

alpha = 7.054802310718643
ridge = Ridge(alpha=alpha).fit(X_train, y_train)
ridge_columns = X_train.columns.values.tolist()

ridge_input_dict_template = dict(zip(ridge_columns, [0]*len(ridge_columns)))


