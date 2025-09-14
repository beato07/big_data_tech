import pandas as pd
from sklearn.datasets import load_wine


def import_load_wine_dataset():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print(df.info())
    return df


if __name__ == '__main__':
    res1 = import_load_wine_dataset()
