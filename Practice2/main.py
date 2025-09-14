import pandas as pd
from sklearn.datasets import load_wine


data = load_wine(as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)


def load_wine_data():
    print(df)
    return df


def load_clean_wine_data():
    if df.isna().any().any():
        df_cleaned = df.dropna()
        print(f"Rows deleted: {len(df_cleaned)}")
    print(df.info())
    print(df.info())


if __name__ == '__main__':

    # Выгрузка многомерных данных
    res1 = load_wine_data()

    # Вывод информации о данных, используя info(), info()
    load_clean_wine_data()
